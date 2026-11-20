"""
Isaac Lab environment for the 2D-trained DQN navigation policy, written to match the 2D env
(dqn_2d_standalone_oscillation/dqn_2d_env.py) wherever the 2D env defines something.

What comes from Isaac (physics):
- the ARL quadrotor with the cascaded velocity -> attitude -> thrust controller of the earlier Isaac env
  (vel_ctrl_kp 4, max_accel 6 m/s^2), holding a fixed flight altitude; the DQN commands forward speed and
  yaw rate only, so the drone has real inertia and response lag;
- one forward depth camera (73 deg HFOV, the OAK-D Lite's stereo field of view), whose depth image is
  reduced to the policy's 5 distance readings by depth_bins.depth_to_bins;
- contacts.

What comes from the 2D env, through a "book" instance of DroneNav2DEnv that mirrors the drone's pose every
step, so the definitions cannot drift apart:
- start, target and obstacle layouts (8 cylinders of radius 0.5 m, one on the start-target path),
- the target offset, previous-action, pose-change and similarity terms of the state,
- the reward, the reach/out-of-bounds/timeout rules and the outcome names.

One environment step is one 0.1 s decision: the chosen command is held for `decimation` physics steps of
sim.dt (10 x 0.01 s), the camera renders once, then the state and reward are computed.

State (19 values, as a 2D checkpoint trained with --camera_count 1 --obs_velocity 1 expects):
  5 depth readings | previous action (7) | pose change over 2 s (3) | target dx, dy | velocity forward, left
"""
from __future__ import annotations

import math
import os
import sys

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_mul, wrap_to_pi
from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG
from isaaclab_contrib.assets import Multirotor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dqn_2d_standalone_oscillation"))
sys.path.insert(0, HERE)

from dqn_config import DQNConfig, SPEED_FRACTIONS  # noqa: E402
from dqn_2d_env import (  # noqa: E402
    DT, DRONE_RADIUS, MAX_OBSTACLES, MAX_SENSE_RANGE, NUM_CYLINDERS, OUT_OF_BOUNDS_DIST, REACHED_THRESHOLD,
    SENSOR_HALF_FOV, CAMERA_BINS, CYLINDER_RADII, DroneNav2DEnv)
from obstacle_geometry import obstacle_surface_distance  # noqa: E402
from depth_bins import depth_to_bins, pixel_rays, quat_mul, quat_rotate  # noqa: E402

SENSORS = ("camera", "geometry")
# Camera mount in the body frame, ROS convention (x right, y down, z forward = body x).
CAMERA_ROT_ROS = (0.5, -0.5, 0.5, -0.5)


@configclass
class NavEnvCfg(DirectRLEnvCfg):
    # 100 Hz physics and control, one 0.1 s decision per env step (the 2D env's DT).
    sim: SimulationCfg = SimulationCfg(dt=0.01, render_interval=10)
    decimation: int = 10
    episode_length_s: float = 60.0  # the 2D env's EPISODE_LENGTH_S; the book enforces the same 600 steps
    observation_space: int = 19
    action_space: int = 3           # raw [vx, vz, yaw_rate] as discrete_action_to_continuous() produces
    state_space: int = 0
    rerender_on_reset: bool = True  # the first state after a reset reads a fresh depth image
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=30.0, replicate_physics=True)

    # Flight altitude, held by a proportional climb-rate correction inside the controller.
    flight_height: float = 1.0
    altitude_hold_kp: float = 2.0
    max_z_vel: float = 1.0
    # Controller of the earlier Isaac env (a PX4-like velocity controller), unchanged.
    vel_ctrl_kp: float = 4.0
    max_accel: float = 6.0
    att_ctrl_kp: float = 12.0
    att_ctrl_kd: float = 2.5
    yaw_rate_ctrl_kp: float = 0.5
    max_yaw_rate_delta: float = 0.4  # rad/s change allowed per physics step

    # Depth camera: 73 deg HFOV pinhole (24 mm focal length, 35.5 mm aperture), mounted forward of the
    # center so the airframe stays out of view.
    camera_width: int = 80
    camera_height: int = 60
    camera_forward_offset: float = 0.25
    camera_near_clip: float = 0.05
    camera_far_clip: float = MAX_SENSE_RANGE
    height_band: float = 0.5        # hit points kept within this many metres of the drone's height
    ground_z: float = 0.15          # hit points at or below this height are ground
    camera_self_radius: float = 0.3  # hit points this close to the drone's center are its own airframe

    obstacle_height: float = 4.0
    # Crash rules beyond the 2D one (center within DRONE_RADIUS of a surface).
    contact_force_threshold: float = 1.0  # N on any robot body
    min_height: float = 0.3
    max_height: float = 2.5
    max_tilt_cos: float = 0.5       # body z axis below 60 deg from vertical


class NavEnv(DirectRLEnv):
    cfg: NavEnvCfg

    def __init__(self, cfg: NavEnvCfg, dqn_cfg: DQNConfig, num_obstacles: int = 8, sensor: str = "camera",
                 render_mode=None, **kwargs):
        if sensor not in SENSORS:
            raise ValueError(f"sensor must be one of {SENSORS}, got {sensor!r}")
        if sensor == "camera" and dqn_cfg.camera_count != 1:
            raise ValueError("the Isaac env has one camera: use --camera_count 1, or --sensor geometry")
        if not 0 <= num_obstacles <= NUM_CYLINDERS:
            raise ValueError(f"num_obstacles must be in [0, {NUM_CYLINDERS}]")
        self.dqn_cfg = dqn_cfg
        self.num_obstacles = num_obstacles
        self.sensor = sensor
        self.book = None  # created after the scene exists; _reset_idx is a no-op until then
        cfg.observation_space = CAMERA_BINS * dqn_cfg.camera_count + 7 + 3 + 2 + (2 if dqn_cfg.obs_velocity else 0)
        super().__init__(cfg, render_mode, **kwargs)

        n, dev = self.num_envs, self.device
        self.book = DroneNav2DEnv(num_envs=n, cfg=dqn_cfg, device=dev, num_obstacles=num_obstacles,
                                  distance_mode="ground_truth")
        self.state_dim = cfg.observation_space
        self.max_steps = self.book.max_steps

        self._vx_cmd = torch.zeros(n, device=dev)
        self._yaw_rate_cmd = torch.zeros(n, device=dev)
        self._applied_yaw_rate = torch.zeros(n, device=dev)
        self._chosen_prev_action = torch.zeros(n, 7, device=dev)
        self._actions = torch.zeros(n, 3, device=dev)
        self._speeds = torch.tensor(SPEED_FRACTIONS, device=dev) * dqn_cfg.forward_speed

        alloc = torch.tensor([
            [1.0, 1.0, 1.0, 1.0],
            [-0.13, -0.13, 0.13, 0.13],
            [-0.13, 0.13, 0.13, -0.13],
            [-0.07, 0.07, -0.07, 0.07],
        ], device=dev)
        self._alloc_inv = torch.linalg.inv(alloc)
        self._mass = 1.24
        self._g = 9.81
        self._rays = None
        # Filled every step for training and diagnostics.
        self.pre_reset_state = {}
        self._pre_state = torch.zeros(n, self.state_dim, device=dev)
        self._reward_inputs = None
        self.last_geometry_readings = None
        self.last_camera_readings = None
        self.physics_trace = None  # set to a list to record (time, env 0 state) every physics step

    # ── scene ────────────────────────────────────────────────────────────────────────────────────────
    def _setup_scene(self):
        robot_cfg = ARL_ROBOT_1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        robot_cfg.actuators["thrusters"].dt = self.cfg.sim.dt
        robot_cfg.init_state.pos = (0.0, 0.0, self.cfg.flight_height)
        self._robot = Multirotor(robot_cfg)
        self.scene.articulations["robot"] = self._robot

        self._camera = TiledCamera(TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/base_link/depth_camera",
            offset=TiledCameraCfg.OffsetCfg(pos=(self.cfg.camera_forward_offset, 0.0, 0.0),
                                            rot=CAMERA_ROT_ROS, convention="ros"),
            data_types=["distance_to_camera"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=35.5,
                                             clipping_range=(self.cfg.camera_near_clip, self.cfg.camera_far_clip)),
            width=self.cfg.camera_width, height=self.cfg.camera_height,
        ))
        self.scene.sensors["depth_camera"] = self._camera

        self._contact = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", history_length=3,
                                                       update_period=0.0, track_air_time=False))
        self.scene.sensors["contact_sensor"] = self._contact

        self._obstacles = []
        for k in range(MAX_OBSTACLES):
            obj = RigidObject(RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Obstacle{k}",
                spawn=sim_utils.CylinderCfg(
                    radius=CYLINDER_RADII[k], height=self.cfg.obstacle_height, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.55, 0.25)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -20.0 - 5.0 * k)),
            ))
            self.scene.rigid_objects[f"obstacle{k}"] = obj
            self._obstacles.append(obj)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        light = sim_utils.DomeLightCfg(intensity=750.0, color=(1.0, 1.0, 1.0))
        light.func("/World/skyLight", light)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    # ── control ──────────────────────────────────────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor):
        """Decode the raw action exactly as the 2D env does, and record the chosen action for the state."""
        self._actions = actions.clone()
        cfg = self.dqn_cfg
        self._vx_cmd = ((torch.tanh(actions[:, 0]) + 1.0) / 2.0) * cfg.max_xy_vel
        self._yaw_rate_cmd = torch.tanh(actions[:, 2]) * cfg.max_yaw_rate
        turn = torch.sign(self._yaw_rate_cmd)
        speed = (self._vx_cmd.unsqueeze(-1) - self._speeds).abs().argmin(dim=-1)
        self._chosen_prev_action = torch.cat([
            torch.stack([turn == 0, turn > 0, turn < 0], dim=-1).float(),
            torch.nn.functional.one_hot(speed, len(SPEED_FRACTIONS)).float()], dim=-1)

    def _apply_action(self):
        """Cascaded velocity -> attitude -> thrust controller, as in the earlier Isaac env: forward speed
        along the heading, a climb rate that holds flight_height, and a slew-limited yaw rate."""
        d, c = self._robot.data, self.cfg
        yaw_rate = torch.clamp(self._yaw_rate_cmd, self._applied_yaw_rate - c.max_yaw_rate_delta,
                               self._applied_yaw_rate + c.max_yaw_rate_delta)
        self._applied_yaw_rate = yaw_rate
        vz_cmd = torch.clamp(c.altitude_hold_kp * (c.flight_height - d.root_pos_w[:, 2]), -c.max_z_vel, c.max_z_vel)

        quat = d.root_quat_w
        yaw = wrap_to_pi(euler_xyz_from_quat(quat)[2])
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        v_cmd = torch.stack([self._vx_cmd * cy, self._vx_cmd * sy, vz_cmd], dim=-1)
        a_fb = torch.clamp((v_cmd - d.root_lin_vel_w) * c.vel_ctrl_kp, -c.max_accel, c.max_accel)
        a_des = a_fb.clone()
        a_des[:, 2] += self._g
        f_des = self._mass * a_des
        thrust = torch.norm(f_des, dim=-1).clamp(min=0.5, max=22.0)

        zb = f_des / (torch.norm(f_des, dim=-1, keepdim=True) + 1e-8)
        xc = torch.stack([cy, sy, torch.zeros_like(yaw)], dim=-1)
        yb = torch.cross(zb, xc, dim=-1)
        yb = yb / (torch.norm(yb, dim=-1, keepdim=True) + 1e-8)
        xb = torch.cross(yb, zb, dim=-1)
        xb = xb / (torch.norm(xb, dim=-1, keepdim=True) + 1e-8)
        q_des = _rotmat_to_quat(torch.stack([xb, yb, zb], dim=-1))

        q_conj = torch.cat([quat[:, :1], -quat[:, 1:]], dim=-1)
        q_err = quat_mul(q_conj, q_des)
        sign = torch.where(q_err[:, :1] < 0, -torch.ones_like(q_err[:, :1]), torch.ones_like(q_err[:, :1]))
        att_err = 2.0 * sign * q_err[:, 1:]
        w = d.root_ang_vel_b
        max_torque = 0.13 * 3.0
        tau_roll = torch.clamp(c.att_ctrl_kp * att_err[:, 0] - c.att_ctrl_kd * w[:, 0], -max_torque, max_torque)
        tau_pitch = torch.clamp(c.att_ctrl_kp * att_err[:, 1] - c.att_ctrl_kd * w[:, 1], -max_torque, max_torque)
        tau_yaw = c.yaw_rate_ctrl_kp * (yaw_rate - w[:, 2])
        wrench = torch.stack([thrust, tau_roll, tau_pitch, tau_yaw], dim=-1)
        self._robot.set_thrust_target(torch.clamp(wrench @ self._alloc_inv.T, 0.0, 6.0))

        if self.physics_trace is not None:
            vel = d.root_lin_vel_w
            self.physics_trace.append(torch.stack([
                vel[:, 0] * cy + vel[:, 1] * sy,     # forward speed
                -vel[:, 0] * sy + vel[:, 1] * cy,    # left speed
                w[:, 2],                             # yaw rate
                d.root_pos_w[:, 2],                  # height
                self._vx_cmd, self._yaw_rate_cmd,
            ], dim=-1).detach().cpu())

    # ── state, reward, termination ───────────────────────────────────────────────────────────────────
    def _pose(self):
        d = self._robot.data
        pos = d.root_pos_w - self.scene.env_origins
        yaw = wrap_to_pi(euler_xyz_from_quat(d.root_quat_w)[2])
        vel = d.root_lin_vel_w[:, :2]
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        vel_body = torch.stack([vel[:, 0] * cy + vel[:, 1] * sy, -vel[:, 0] * sy + vel[:, 1] * cy], dim=-1)
        return pos, yaw, vel_body

    def _camera_pose(self):
        """The camera's world pose (ROS convention) from the robot's current pose and the fixed mount.
        The camera sensor's own pos_w / quat_w_ros are not refreshed every step (they kept the spawn pose)."""
        d = self._robot.data
        n = d.root_pos_w.shape[0]
        off = torch.tensor([self.cfg.camera_forward_offset, 0.0, 0.0], device=self.device).expand(n, 3)
        rot = torch.tensor(CAMERA_ROT_ROS, device=self.device).expand(n, 4)
        return d.root_pos_w + quat_rotate(d.root_quat_w, off), quat_mul(d.root_quat_w, rot)

    def _camera_readings(self, pos, yaw):
        data = self._camera.data
        depth = data.output["distance_to_camera"]
        if self._rays is None:
            self._rays = pixel_rays(data.intrinsic_matrices, depth.shape[1], depth.shape[2])
        drone_w = pos + self.scene.env_origins
        cam_pos, cam_quat = self._camera_pose()
        return depth_to_bins(depth, self._rays, cam_pos, cam_quat, drone_w, yaw, SENSOR_HALF_FOV,
                             CAMERA_BINS, MAX_SENSE_RANGE, self.cfg.height_band, self.cfg.ground_z,
                             self.dqn_cfg.sensor_min_range, far_clip=self.cfg.camera_far_clip,
                             self_radius=self.cfg.camera_self_radius)

    def _build_state(self, pos, yaw, vel_body):
        """The book's 2D state at this pose (its geometry readings, previous action, pose change, target
        offset), with the readings replaced by the camera's in camera mode and the velocity appended."""
        bk = self.book
        bk.pos, bk.yaw = pos[:, :2].clone(), yaw.clone()
        obs, dist = bk._get_obs()
        state = obs["dqn_state"]
        nr = bk.num_readings
        self.last_geometry_readings = state[:, :nr].clone()
        if self.sensor == "camera" and self.num_obstacles > 0:
            cam = self._camera_readings(pos, yaw)
            self.last_camera_readings = cam
            state = torch.cat([cam, state[:, nr:]], dim=-1)
        if self.dqn_cfg.obs_velocity:
            state = torch.cat([state, vel_body], dim=-1)
        return state, dist

    def _get_dones(self):
        bk = self.book
        pos, yaw, vel_body = self._pose()
        pos_before = bk.pos.clone()
        dist_before = torch.norm(bk.target - pos_before, dim=-1)

        bk.pos, bk.yaw = pos[:, :2].clone(), yaw.clone()
        bk.step_count += 1
        bk._update_pose_similarity()
        bk.prev_action = self._chosen_prev_action.clone()
        state, dist = self._build_state(pos, yaw, vel_body)

        if self.num_obstacles > 0:
            surface = obstacle_surface_distance(pos[:, :2], bk.obstacle_xy[:, :NUM_CYLINDERS], bk._circle_r,
                                                bk.obstacle_xy[:, NUM_CYLINDERS:], bk._box_half)
        else:
            surface = torch.full_like(dist, math.inf)
        forces = torch.norm(self._contact.data.net_forces_w_history, dim=-1)
        contact = (forces > self.cfg.contact_force_threshold).flatten(1).any(dim=1)
        up = _body_z_world(self._robot.data.root_quat_w)[:, 2]
        crashed = ((surface < DRONE_RADIUS) | contact | (pos[:, 2] < self.cfg.min_height)
                   | (pos[:, 2] > self.cfg.max_height) | (up < self.cfg.max_tilt_cos))
        reached = dist < REACHED_THRESHOLD
        out_of_bounds = dist > OUT_OF_BOUNDS_DIST
        timed_out = bk.step_count >= bk.max_steps
        terminated = reached | crashed
        truncated = (out_of_bounds | timed_out) & ~terminated

        bk._last_reached_flag = reached
        bk._last_ground_crash = (pos[:, 2] < self.cfg.min_height)
        bk._last_any_contact = crashed
        bk._last_out_of_bounds = out_of_bounds
        self._pre_state = state
        self.pre_reset_state = {
            "pos": pos[:, :2].clone(), "yaw": yaw.clone(), "target": bk.target.clone(),
            "obstacle_xy": bk.obstacle_xy.clone(), "obstacle_active": bk.obstacle_active.clone(),
            "speed": vel_body.norm(dim=-1), "height": pos[:, 2].clone(), "contact": contact.clone(),
            "surface_dist": surface.clone(),
        }
        travel_speed = (pos[:, :2] - pos_before).norm(dim=-1) / DT
        self._reward_inputs = (dist, reached, crashed, travel_speed, surface,
                               state[:, :bk.num_readings].min(dim=-1).values, dist_before)
        return terminated, truncated

    def _get_rewards(self):
        dist, reached, crashed, travel, surface, sensor_dist, dist_before = self._reward_inputs
        reward, self.last_reward_breakdown = self.book._compute_reward(
            dist, self._actions, reached, crashed, vx_signed=travel, surface_dist=surface,
            sensor_dist=sensor_dist, dist_before=dist_before, return_breakdown=True)
        return reward

    def _get_observations(self):
        pos, yaw, vel_body = self._pose()
        # An env reset on this step starts from the pose _reset_idx wrote, at rest; take it from the book
        # rather than from simulator buffers that may not show the write until the next physics step.
        fresh = self.book.step_count == 0
        if fresh.any():
            pos = pos.clone()
            pos[:, :2] = torch.where(fresh[:, None], self.book.pos, pos[:, :2])
            pos[:, 2] = torch.where(fresh, torch.full_like(pos[:, 2], self.cfg.flight_height), pos[:, 2])
            yaw = torch.where(fresh, self.book.yaw, yaw)
            vel_body = torch.where(fresh[:, None], torch.zeros_like(vel_body), vel_body)
        state, _ = self._build_state(pos, yaw, vel_body)
        return {"policy": state}

    # ── reset ────────────────────────────────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if self.book is None:
            return
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device)
        env_ids = env_ids.to(self.device).long()
        n = len(env_ids)
        bk = self.book
        bk._sample_positions(env_ids)  # start, heading, target, obstacles; clears the book's step state

        origins = self.scene.env_origins[env_ids]
        root = self._robot.data.default_root_state[env_ids].clone()
        root[:, 0:2] = bk.pos[env_ids] + origins[:, 0:2]
        root[:, 2] = self.cfg.flight_height + origins[:, 2]
        half = bk.yaw[env_ids] / 2
        root[:, 3:7] = torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half),
                                    torch.sin(half)], dim=-1)
        root[:, 7:] = 0.0
        self._robot.write_root_state_to_sim(root, env_ids=env_ids)
        if hasattr(self._robot, "actuators") and "thrusters" in self._robot.actuators:
            self._robot.actuators["thrusters"].curr_thrust[env_ids] = self._mass * self._g / 4.0

        for k, obj in enumerate(self._obstacles):
            st = torch.zeros(n, 13, device=self.device)
            active = bk.obstacle_active[env_ids, k]
            st[:, 0:2] = torch.where(active[:, None], bk.obstacle_xy[env_ids, k], torch.zeros_like(st[:, 0:2]))
            st[:, 0:2] += origins[:, 0:2]
            st[:, 2] = torch.where(active, torch.full_like(st[:, 2], self.cfg.obstacle_height / 2),
                                   torch.full_like(st[:, 2], -20.0 - 5.0 * k))
            st[:, 3] = 1.0
            obj.write_root_state_to_sim(st, env_ids=env_ids)

        self._vx_cmd[env_ids] = 0.0
        self._yaw_rate_cmd[env_ids] = 0.0
        self._applied_yaw_rate[env_ids] = 0.0
        self._chosen_prev_action[env_ids] = 0.0


class NavEnvWrapper:
    """The 2D env's interface (reset, step, last_outcomes, step_count, state_dim, max_steps,
    pre_reset_state) over NavEnv, so the 2D training and evaluation code drives it unchanged.

    step() returns the state each env reached (pre-reset, the bootstrap target for a cut-off episode)
    and, in info["reset_obs"], the state to act from next (the new episode's first state where one ended).
    """

    def __init__(self, env: NavEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.state_dim = env.state_dim
        self.max_steps = env.max_steps
        self.cfg = env.dqn_cfg
        self.num_obstacles = env.num_obstacles
        # For eval_inertia.gap_width
        self._circle_r = env.book._circle_r
        self._box_half = env.book._box_half

    @property
    def step_count(self):
        return self.env.book.step_count

    @property
    def pre_reset_state(self):
        return self.env.pre_reset_state

    @property
    def target(self):
        return self.env.book.target

    @property
    def pos(self):
        return self.env.book.pos

    def reset(self):
        obs, _ = self.env.reset()
        return {"dqn_state": obs["policy"]}, {}

    def step(self, actions_continuous):
        obs, reward, terminated, truncated, _ = self.env.step(actions_continuous)
        return ({"dqn_state": self.env._pre_state}, reward, terminated, truncated,
                {"reset_obs": {"dqn_state": obs["policy"]}})

    def last_outcomes(self):
        return self.env.book.last_outcomes()

    def close(self):
        self.env.close()


def make_env(dqn_cfg: DQNConfig, num_envs: int, num_obstacles: int, sensor: str = "camera", seed: int = 0,
             **cfg_overrides) -> NavEnvWrapper:
    """Build the wrapped env. Call after the Isaac app has been launched."""
    cfg = NavEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    for k, v in cfg_overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"NavEnvCfg has no field {k}")
        setattr(cfg, k, v)
    return NavEnvWrapper(NavEnv(cfg, dqn_cfg, num_obstacles=num_obstacles, sensor=sensor))


def _body_z_world(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1)


def _rotmat_to_quat(R):
    """(N, 3, 3) rotation matrices -> (N, 4) unit quaternions (w, x, y, z), Shepperd's method."""
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    q = torch.zeros(R.shape[0], 4, device=R.device)
    c1 = tr > 0
    s = torch.sqrt(torch.clamp(tr + 1.0, min=1e-8)) * 2
    q[c1] = torch.stack([0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s], dim=-1)[c1]
    c2 = ~c1 & (m00 >= m11) & (m00 >= m22)
    s = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-8)) * 2
    q[c2] = torch.stack([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s], dim=-1)[c2]
    c3 = ~c1 & ~c2 & (m11 >= m22)
    s = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=1e-8)) * 2
    q[c3] = torch.stack([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s], dim=-1)[c3]
    c4 = ~c1 & ~c2 & ~c3
    s = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=1e-8)) * 2
    q[c4] = torch.stack([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s], dim=-1)[c4]
    return q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
