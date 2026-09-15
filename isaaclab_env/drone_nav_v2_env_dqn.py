# DroneRanger Stage 2: Direct RL environment with depth camera obstacle avoidance
from __future__ import annotations
import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab_contrib.assets import Multirotor
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg, ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_apply, quat_mul, euler_xyz_from_quat, wrap_to_pi, subtract_frame_transforms
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG


@configclass
class DroneNavV2SimpleObsEnvCfg(DirectRLEnvCfg):
    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1.0/100.0, render_interval=2)
    decimation: int = 2
    episode_length_s: float = 20.0

    # Spaces: "policy" group is state-only (21). Depth image is a separate
    # obs group ("depth") consumed directly by the CNN model; it is NOT
    # included in cfg.observation_space (that field only registers the
    # gym space for the "policy" key — DirectRLEnv.step() does not
    # restrict which keys _get_observations() may return).
    # BERKELEY-PAPER-STYLE REDESIGN: state space reduced from 21 to 7
    # values, matching "Vision-Based Deep RL for Autonomous Drone Flight"
    # (Saran, UC Berkeley EECS-2023-280, real-world validated on DJI
    # Matrice 300 + ZED2i). World-frame position error (x,y,z), roll,
    # pitch, heading error, yaw rate -- simpler than our previous body-
    # frame-transformed 21-dim vector, and this exact 7-dim design was
    # proven to work end-to-end (sim + real hardware).
    observation_space: int = 15   # SimpleObs experiment: 12-dim state + 3-dim nearest-visible-obstacle vector (was 12, depth-image-based)
    # example (root_lin_vel_b[3] + root_ang_vel_b[3] + projected_gravity_b[3] +
    # desired_pos_b[3]), replacing the old 7-dim world-frame design.
    # ARCHITECTURE CHANGE: action space switched from raw per-motor thrust
    # to a velocity command (matches real deployment, which uses PX4 in
    # OFFBOARD velocity mode, not direct motor control). The 4 network
    # outputs now mean [vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd] rather than
    # 4 motor thrusts. Attitude stabilization is handled by a simulated
    # velocity->attitude->thrust controller (mirrors PX4's cascaded control
    # architecture) in _apply_action(), NOT learned by the policy -- exactly
    # matching how attitude stabilization will be PX4's job on real hardware,
    # not the network's job.
    # Reduced from 4 to 3: dropped lateral (vy) velocity entirely,
    # matching the reference paper's constraint that the drone may
    # ONLY fly in the direction its camera is pointing (no strafing).
    # This guarantees the camera always looks where the drone is
    # capable of moving -- to go left, it must first yaw left, then
    # move forward. Simpler action space, proven on real hardware.
    action_space: int = 3  # [vx_cmd (forward), vz_cmd, yaw_rate_cmd]
    state_space: int = 13   # critic-only: 12-dim state (v2) + normalized min_obstacle_dist

    # Depth image resolution (kept in sync with the TiledCameraCfg below)
    depth_image_size: tuple = (84, 84)

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, env_spacing=40.0, replicate_physics=True
    )

    # Reward scales
    # RESTORED to original values. Reward redesign (progress-based shaping +
    # downweighted stability terms) was reverted because the actual root
    # cause of the "drone doesn't fly" behavior was traced to a physical
    # bug: the robot's default spawn point (0,0,0) coincided with the
    # ground plane, so it was spawning embedded in/on the ground rather
    # than in free air. That has been fixed separately (see init_state.pos
    # below and the reset() Z-position handling). We're now re-testing
    # whether the ORIGINAL reward design works correctly once the drone can
    # Industry-standard reward design (based on SkyDreamer, LCLA, Indoor Path Planning papers)
    # ══════════════════════════════════════════════════════════════════
    # REWARD REDESIGN: replicating "Vision-Based Deep RL for Autonomous
    # Drone Flight" (Saran, UC Berkeley EECS-2023-280) -- real-hardware-
    # validated on DJI Matrice 300 + ZED2i depth camera. Weights below
    # preserve the EXACT RELATIVE RATIOS from the paper's Table 2.2/2.3
    # (each weight / their terminal "reached goal" weight of 2.0),
    # rescaled so OUR terminal "reached" reward = 100 (an arbitrary but
    # consistent round number for our own bookkeeping) instead of their
    # 2.0 -- NOT a blind copy of their absolute numbers, which were
    # calibrated for their own reward/gamma/episode-length convention.
    # NOTE: the paper's "heading" term's piecewise bonus table entry
    # appeared to have PDF-extraction corruption (bonus increasing with
    # LARGER heading error makes no intuitive sense); we use a corrected,
    # sensible interpretation instead (small error = bonus, large error =
    # penalty) -- flagged here for transparency.
    rew_distance: float = 0.3      # (d0-d)/d0 progress-toward-goal term; paper's 6e-3/2.0*100
    rew_heading: float = 0.025     # exp(-16*heading_err^2) + small alignment bonus/penalty; paper's 5e-4/2.0*100
    # Reverted to paper's exact ratio (0.0125 = 2.5e-4/2.0*100). The 0.05
    # value was tuned while the critic was blind to obstacle distance
    # (state_space=0, critic reused "policy" 7-dim only) -- raising the
    # penalty was likely compensating for bad advantage estimates near
    # obstacles rather than a genuinely-needed stronger signal. Reverting
    # concurrently with adding critic-side obstacle-distance input
    # (state_space=8, see below) so the two changes can be evaluated together.
    rew_obstacle: float = 0.10     # SPEED-COUPLED FORM, threshold widened 1.0m->2.0m (was w=0.05,
    # then squared-distance-only forms at w=0.15/0.04, both of which caused
    # training divergence for reasons that were never fully root-caused --
    # offline analysis on real training data showed neither of those attempts
    # ever actually out-weighed rew_vel_progress in total magnitude (max 23.7%
    # of rew_vel_progress's total), disproving the original "reward imbalance"
    # explanation for the divergence. Separately, collision analysis showed
    # vx_cmd stays pinned near max speed all the way into obstacles regardless
    # of proximity -- this term was a pure position penalty with no speed
    # dependence, so "drifting slowly into 0.9m" and "ramming in at 2.5m/s"
    # were penalized identically. This form multiplies proximity intrusion by
    # current speed, so danger is priced by both closeness AND velocity --
    # cheap to be close while slow, expensive to be close while fast. Offline
    # replay on real collision data showed this structural change alone (same
    # w=0.0125) already roughly doubled the term's share of total reward
    # magnitude vs rew_vel_progress (3.4% -> 7.9%); w=0.05 was chosen as a
    # moderate middle ground (~32% of rew_vel_progress's total) -- enough to
    # matter without repeating the earlier large jumps that preceded (but were
    # not conclusively the cause of) divergence.
    rew_unnecessary_vz: float = 0.05   # penalty on |vz_cmd| when min_obstacle_dist > 1.0m
    # ── Height-alignment progress reward (added): step-to-step potential-based
    # shaping on |height - target_z|, NOT a historical-min potential like
    # rew_distance. Deliberately per-step (telescoping over a round trip nets
    # to zero) so repeated legitimate climb-for-obstacle / descend-back cycles
    # are never penalized or "used up" -- unlike a historical-min formulation,
    # which would only reward the FIRST return to a given height and block
    # reward on subsequent obstacle-driven round trips. Added after observing
    # (via play.py) that the drone overshoots target height on final approach
    # and corrects via slow, oscillatory vz_cmd (rapid -1/+1 swings) instead
    # of a clean single descent, because previously nothing gave vz a direct
    # signal toward the target's actual height -- rew_unnecessary_vz only
    # gates on obstacle proximity and was confirmed (via logged data) to be
    # near-zero during these overshoot episodes since they often occur close
    # to obstacles, not in open space. ──
    rew_height_progress: float = 0.15  # weight on (prev_height_diff - height_diff) each step
    rew_vel_goal: float = 0.002    # reward velocity component toward goal; paper's 4e-5/2.0*100
    rew_heading_rate: float = 0.015  # penalize |yaw_rate|; paper's 3e-4/2.0*100
    rew_vertical: float = 0.004     # penalize squared error in z vs goal height; paper's 8e-5/2.0*100

    # ── v2 reward redesign: exp-shaped dense terms (adopted from internal
    # early-design doc: distance/ang_vel/lin_vel/yaw/flat-orientation exp
    # rewards + action_rate/action_magnitude L2 penalties), rescaled down
    # by ~50x from the doc's raw per-step weights so that even a "perfect"
    # dense-reward episode cannot out-accumulate the terminal rew_reached
    # (100) -- avoids the "loiter near goal" degenerate solution seen
    # before. Old rew_heading/rew_vel_goal/rew_heading_rate/rew_vertical
    # above are kept defined (not deleted) but no longer referenced in
    # _get_rewards() -- superseded by the terms below, which operate on
    # the v2 body-frame state (desired_pos_b, ang_vel_b, lin_vel_b,
    # gravity_b) instead of world-frame heading_error.
    rew_distance_exp: float = 0.5      # exp(-k * dist), doc's 25.0 / 50
    distance_exp_k: float = 0.5        # decay rate for the exp above
    rew_ang_vel_exp: float = 0.04      # exp(-k * |ang_vel|^2), doc's 10.0 / 50
    # REWEIGHTED (was 0.2): reward reconstruction from play.py logs showed this
    # term contributed 89-104 reward per 1000-step episode, 3-4x the next-largest
    # term and dwarfing every direction-aware term combined (rew_distance,
    # rew_height_progress, rew_yaw_aligned). Because it rewards low angular
    # velocity unconditionally (any steady motion, regardless of direction),
    # this magnitude imbalance let it fund both the "climb to 14m" and "orbit
    # near target" failure modes -- steady uncontrolled climbing or circling
    # farmed nearly as much reward as flying correctly toward the goal. Tried
    # gating it on progress (ang_vel_progress_gate_k, see .bak_pre_gate_revert)
    # but any v_along-based gate condition has edge cases (3D-composite v_along
    # gets fooled by horizontal progress masking vertical climb; gating also
    # suppressed this term during legitimate low-speed turn-to-align, causing a
    # stationary-spin regression). Reverted the gate and instead just cut the
    # weight down to be in line with the other dense terms, so it can no longer
    # dominate the reward sum in any scenario -- no conditional logic, so no new
    # edge cases. Original anti-jitter intent is preserved at reduced strength.
    ang_vel_exp_k: float = 0.5
    # ── Progress-gate for rew_ang_vel (added): rew_ang_vel rewards LOW
    # angular velocity unconditionally, with no requirement that the drone
    # be making progress toward the target. Confirmed via play.py + offline
    # reward reconstruction that this was the dominant reward source (89-104
    # of ~100-130 total reward per 1000-step episode, vs single digits for
    # every other dense term) for BOTH the "climbs to 14m" and "orbits near
    # target without entering" failure modes -- any steady, low-angular-
    # velocity motion farms this term regardless of direction (e.g. a
    # constant-rate climb is just as "stable" as flying straight at the
    # goal). Gated by clamp(v_along, max=0) -> exp, so gate=1 whenever
    # v_along >= 0 (no bonus for speed, just no penalty) and decays smoothly
    # toward 0 as the drone moves away from the target (v_along < 0),
    # closing the exploit without reintroducing the earlier "do nothing to
    # avoid turning cost" failure mode this term was originally added for. ──
    ang_vel_progress_gate_k: float = 2.0  # decay rate for gate when v_along < 0
    rew_lateral_vel_exp: float = 0.05  # exp(-k * |v_lateral|^2), doc's 2.5 / 50
    # ── v2 fix: reward for velocity component TOWARD the goal (reuses
    # v_along, already computed for rew_lateral_vel). Added because
    # rew_ang_vel_exp + rew_lateral_vel_exp together made "do nothing"
    # nearly free while turning/moving cost reward every step -- policy
    # learned extremely slow turning + crawling forward speed whenever
    # heading wasn't yet aligned (confirmed via play.py: turns and closes
    # distance ~normally ONLY when no turn is needed at all). This term
    # rewards actual task progress (closing distance) directly, regardless
    # of whether the drone is still turning, so "doing something toward
    # the goal" is no longer strictly worse than "doing nothing."
    # RE-ENABLED (see _get_rewards return statement): the original reason
    # this was pulled from the return value was a backward-flight exploit
    # (see the "BUGFIX (v2)" note in _apply_action) -- the policy flew
    # backward to farm rew_ang_vel_exp without paying the turning cost.
    # That exploit is now closed at the source: vx_cmd is squashed to
    # [0, 1] (non-negative only), so backward flight is physically
    # impossible regardless of what this term rewards. Weight lowered
    # from the old 0.1 to 0.03 (max ~0.075/step at max_xy_vel=2.5) so it
    # nudges the policy toward facing/closing on the target without
    # dominating rew_distance or the terminal rewards.
    rew_vel_progress: float = 0.03
    lateral_vel_exp_k: float = 0.5
    rew_yaw_aligned: float = 0.04      # cos(angle to target), doc's 2.0 / 50
    rew_flat: float = 0.02             # -(gx^2+gy^2), doc's 1.0 / 50
    rew_action_rate: float = 0.001     # -|actions - prev_actions|^2, doc's 0.05 / 50
    rew_action_magnitude: float = 0.001  # -|actions|^2, doc's 0.05 / 50

    rew_reached: float = 100.0      # terminal: reached goal (dist < 0.75m), paper's own threshold
    rew_crashed: float = -50.0      # terminal: collision, paper's -1.0/2.0*100
    rew_timedout: float = -4.0      # terminal: episode timeout, paper's -0.08/2.0*100
    rew_out_of_bounds: float = -8.0 # terminal: flew too far from goal, paper's -0.16/2.0*100
    max_dist_from_start: float = 40.0  # meters; triggers out-of-bounds termination (widened from 20.0)

    # ── Velocity command limits (must match real PX4 OFFBOARD config!) ──
    max_xy_vel: float = 2.5     # m/s, forward speed limit (paper trained at 1 m/s;
    """kept our existing value since our obstacle corridor is shorter than
    their 80x80m world -- revisit if training struggles, paper showed
    training speed and evaluation speed must roughly match)."""
    max_z_vel: float = 1.0      # m/s, vertical speed limit
    max_yaw_rate: float = 1.57  # rad/s (~90 deg/s)

    # -- Altitude-hold safety net (method 2, see chat notes) --
    # Root cause of turn-induced altitude loss: thrust_mag in _apply_action
    # is sized correctly in theory (it's the norm of the full 3D desired
    # accel vector, which *should* compensate for tilt), but only if
    # attitude tracks q_des instantaneously. The attitude PD controller
    # (att_ctrl_kp/kd, torque-limited) has finite bandwidth, so during fast
    # yaw-rate changes the actual body z-axis lags q_des and the realized
    # vertical thrust component is briefly short of mg -- this deficit
    # accumulates into real altitude loss the faster/more often the
    # commanded direction changes. This was previously only addressed via
    # rew_action_rate, a soft *training-time* penalty that discourages
    # thrashy actions but cannot prevent a not-yet-well-trained (or
    # collapsed) policy from actually commanding them.
    # This is a position-feedback correction added directly to vz_cmd in
    # _apply_action -- independent of the RL policy entirely, active
    # even if the policy always sends vz_cmd=0 (as this DQN task's action
    # table does). It measures how far current altitude has drifted from
    # altitude_hold_target and adds a proportional climb-rate correction,
    # so lost altitude gets actively recovered rather than just not-lost.
    # Set altitude_hold_kp=0.0 to fully disable and restore old behavior.
    altitude_hold_target: float = 1.0   # meters; matches this task's fixed spawn height
    altitude_hold_kp: float = 2.0       # (m/s) per meter of altitude error

    # -- Yaw-rate slew-rate limit (method 1, see chat notes) --
    # rew_action_rate only discourages thrashy switching as a soft
    # training-time penalty -- it cannot stop the physical controller from
    # actually receiving a hard switch. This is the source-level fix:
    # physically clamp how much yaw_rate_cmd is allowed to change per
    # control step, regardless of what the policy outputs.
    max_yaw_rate_delta: float = 0.4     # rad/s change allowed per control step

    # ── Simulated velocity controller gains (fixed classical PD control,
    # NOT learned -- analogous to PX4's own onboard controller) ──
    vel_ctrl_kp: float = 4.0
    max_accel: float = 6.0
    att_ctrl_kp: float = 12.0
    att_ctrl_kd: float = 2.5
    yaw_rate_ctrl_kp: float = 0.5

    min_height: float = -0.1   # tightened: touching near ground now counts as crash
    target_pos_range_x: tuple = (-5.0, 5.0)
    target_pos_range_y: tuple = (-5.0, 5.0)
    # DQN 2D experiment: fixed to match the drone's own init height (1.0m),
    # not randomized. The 3D "reached" check (dist = norm(desired_pos_b))
    # includes the z-axis, but this DQN's action space has no vz control
    # (vz_cmd locked to 0 in all 4 discrete actions) -- a randomized target
    # height would make many episodes physically unreachable regardless of
    # how well the horizontal navigation is learned, since the drone can
    # never close a height gap it has no way to act on.
    target_pos_range_z: tuple = (1.0, 1.0)

    # ── Curriculum learning parameters ────────────────────────────────────
    # Manually bumped between resume runs (not auto-advanced inside a single
    # training session, since training happens across many separate resumed
    # launches, not one continuous run — see project notes). Increase these
    # once play.py confirms the current difficulty level is handled well.
    # Curriculum stage 1: fewer, more widely-spaced obstacles. Bump these up
    # manually (e.g. 3/2.0, then 5/1.3 = original full difficulty) once
    # play.py confirms the current stage is handled well.
    # Curriculum progress: 0 obstacles (pure nav, 90-96% success) -> 3
    # obstacles with target forced past the corridor (60-82% success,
    # confirmed via play.py to require real navigation, not just a lucky
    # skip). Bumping to all 5 with zigzag lateral placement (see `lateral`
    # in _reset_idx) to prevent the observed "swing wide around one side"
    # shortcut and force genuine weaving between obstacles.
    num_active_obstacles: int = 0   # DQN Stage 0 v2: pure navigation with the new 3-action space (forward/turn-left/turn-right, decoupled). Deliberately 0 for this stage -- must validate pure nav converges cleanly with the new action space before introducing the obstacle again (Stage 1). Do not revert to 1 until Stage 0 v2 is confirmed.
    # the current reward set (rew_unnecessary_vz + rew_distance potential +
    # rew_height_progress + rew_ang_vel_exp=0.04) performs excellently in pure
    # navigation (reached_target ~97-100%, no climb, no orbit-tangent, smooth
    # turning) -- better than or equal to the original v1 baseline. Re-enabling
    # obstacles to observe whether/how the climb and orbit-tangent issues
    # re-emerge, resuming from the clean no-obstacle checkpoint rather than
    # the original pre-reward-changes baseline.
    """non-negative fix + rew_vel_progress) is now validated as solid via
    play.py (no backward flight, no crawling-turn timeout, reached_target
    ~98-100%). Reintroducing obstacles one at a time, resuming from that
    checkpoint rather than retraining from scratch -- the CNN will need to
    adapt to actually seeing a nearby obstacle in the depth image for the
    first time (previously always empty/far), so some readjustment period
    is expected even though the underlying nav/turning skills should
    transfer."""
    obstacle_min_spacing: float = 2.5   # meters between consecutive active
    """obstacles along the start->target path (was a fixed 1.3m gap before
    curriculum learning was introduced)."""


class DroneNavV2SimpleObsEnv(DirectRLEnv):
    cfg: DroneNavV2SimpleObsEnvCfg

    def __init__(self, cfg: DroneWithObstacleEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # NOTE: Multirotor has no joints — thruster indexing is handled internally
        # by set_thrust_target(thruster_ids=None) which defaults to "all thrusters",
        # so no find_joints()/find_bodies() call is needed here.

        # Buffers (num_envs is available after super().__init__)
        self._target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, 3, device=self.device)  # BUGFIX: was 4 (stale from pre-3-dim action space era, never used until v2's rew_action_rate)
        self._prev_yaw_rate_cmd = torch.zeros(self.num_envs, device=self.device)  # method 1: slew-rate limit state, see cfg.max_yaw_rate_delta
        self._actions = torch.zeros(self.num_envs, 3, device=self.device)  # BUGFIX: was 4 (stale from pre-3-dim action space era)
        self._prev_distance = torch.zeros(self.num_envs, device=self.device)
        # Tracks the closest distance-to-target ever achieved this episode.
        # Used to make rew_distance a potential-based (progress-only) reward
        # instead of a per-step function of current distance -- this closes
        # the "orbit near target, never enter, accumulate more dense reward
        # than one-time rew_reached" exploit observed via play.py (drone
        # circling at 1-2m radius for 700+ steps, netting 200+ reward vs
        # rew_reached=100). Reset alongside _prev_distance in _reset_idx.
        self._min_dist_so_far = torch.zeros(self.num_envs, device=self.device)
        # Previous step's |height - target_z|, for step-to-step potential-based
        # height-alignment shaping (see rew_height_progress notes above).
        self._prev_height_diff = torch.zeros(self.num_envs, device=self.device)
        # d0: initial distance to target at episode start, used by the
        # (d0-d)/d0 progress reward (Berkeley paper design). Set in
        # _reset_idx() right after each env's target is (re)placed.
        self._d0 = torch.ones(self.num_envs, device=self.device)
        self._prev_vel_error = torch.zeros(self.num_envs, 3, device=self.device)

        # ── Simulated velocity controller setup ─────────────────────────────
        # Reduced 4x4 allocation matrix (rows [Fz, taux, tauy, tauz] from
        # ARL_ROBOT_1_CFG.allocation_matrix rows 2-5; rows 0-1 (fx,fy) are
        # all-zero since each thruster's force is purely along body +z).
        # Maps [f0,f1,f2,f3] (per-motor thrust) -> [Fz, taux, tauy, tauz].
        # We need the INVERSE: given a desired [Fz, taux, tauy, tauz] from
        # the controller below, solve for the 4 motor thrusts that produce it.
        alloc_reduced = torch.tensor([
            [1.0, 1.0, 1.0, 1.0],
            [-0.13, -0.13, 0.13, 0.13],
            [-0.13, 0.13, 0.13, -0.13],
            [-0.07, 0.07, -0.07, 0.07],
        ], device=self.device)
        self._alloc_inv = torch.linalg.inv(alloc_reduced)  # (4,4)
        self._drone_mass = 1.24  # kg, from robot_physx_view.get_masses() (see project notes)
        self._gravity = 9.81

        self._last_depth = torch.ones(self.num_envs, 1, 84, 84, device=self.device)  # (N,1,H,W)
        # Distance-to-target from the previous step, used for potential-based
        # progress reward. Initialized to 0 here; _reset_idx() sets the real
        # value right after each env's first reset, before any reward is computed.

        # ── Ground-plane detection setup (v2: classify by hit-point height) ──
        # v1 (matching actual depth reading against the theoretical distance
        # a ray would need to travel to hit z=0) was reverted: it frequently
        # misclassified real obstacles as ground, because at low flight
        # altitude a downward-angled ray's theoretical ground distance often
        # coincides numerically with an actual nearby obstacle's distance --
        # exactly the rays most in need of correct obstacle detection.
        # Confirmed via testing: near-100% obstacle collision rate, bizarre
        # sideways-circling flight.
        #
        # v2: instead of comparing distances, compute the ABSOLUTE HEIGHT of
        # the point each ray actually hits (drone_height + actual_dist *
        # ray_dz) and check if THAT is near zero. A real obstacle's surface
        # sits well above z=0 (it's a vertical column/box), so its hit height
        # is unambiguously far from zero regardless of what the ground's
        # theoretical distance along that same ray happens to be. Verified
        # via isolated test: an obstacle at 2.5m gave hit height 0.81m
        # (correctly NOT ground), while true ground gave hit height ~0.0m
        # (correctly IS ground) -- no distance-based ambiguity.
        img_h, img_w = 84, 84
        _focal_length = 24.0
        _horizontal_aperture = 35.5
        _hfov = 2.0 * math.atan(_horizontal_aperture / (2.0 * _focal_length))
        _vfov = _hfov

        _rows = torch.arange(img_h, device=self.device).float()
        _cols = torch.arange(img_w, device=self.device).float()
        _row_angles = (_rows - (img_h - 1) / 2.0) / img_h * _vfov
        _col_angles = (_cols - (img_w - 1) / 2.0) / img_w * _hfov
        _row_grid, _col_grid = torch.meshgrid(_row_angles, _col_angles, indexing="ij")

        _dir_x = torch.tan(_col_grid)
        _dir_y = torch.tan(_row_grid)
        _dir_z = torch.ones_like(_dir_x)
        _rays_cam = torch.stack([_dir_x, _dir_y, _dir_z], dim=-1)
        _rays_cam = _rays_cam / torch.norm(_rays_cam, dim=-1, keepdim=True)
        self._camera_rays = _rays_cam.reshape(-1, 3)  # (H*W, 3)

        self._camera_mount_quat = torch.tensor([0.5, -0.5, 0.5, -0.5], device=self.device)
        self._ground_height_tolerance = 0.2  # meters; hit points below this height are classified as ground
        # ── Simplified-observation experiment: persist HFOV/VFOV so the
        # nearest-visible-obstacle computation in _get_observations() can
        # reuse the exact same field-of-view definition as the depth
        # camera, keeping the two observation modes as comparable as
        # possible (same visibility rule, different information channel). ──
        self._hfov = _hfov
        self._vfov = _vfov
        # ── DQN: 8x8 ray grid for lateral (left/right) cameras, reusing the
        # same HFOV/VFOV as the front camera (identical physical camera
        # model, just mounted at a different yaw). Shared by both lateral
        # cameras since only the mount quaternion differs between them;
        # used for ground-height filtering, mirroring the front camera's
        # own approach below in _get_observations().
        _lat_res = 8
        _lat_rows = torch.arange(_lat_res, device=self.device).float()
        _lat_cols = torch.arange(_lat_res, device=self.device).float()
        _lat_row_angles = (_lat_rows - (_lat_res - 1) / 2.0) / _lat_res * self._vfov
        _lat_col_angles = (_lat_cols - (_lat_res - 1) / 2.0) / _lat_res * self._hfov
        _lat_row_grid, _lat_col_grid = torch.meshgrid(_lat_row_angles, _lat_col_angles, indexing="ij")
        _lat_dir_x = torch.tan(_lat_col_grid)
        _lat_dir_y = torch.tan(_lat_row_grid)
        _lat_dir_z = torch.ones_like(_lat_dir_x)
        _lat_rays_cam = torch.stack([_lat_dir_x, _lat_dir_y, _lat_dir_z], dim=-1)
        _lat_rays_cam = _lat_rays_cam / torch.norm(_lat_rays_cam, dim=-1, keepdim=True)
        self._camera_rays_lateral = _lat_rays_cam.reshape(-1, 3)  # (64, 3)
        # ── DQN: 8x8 ray grid for lateral (left/right) cameras, reusing the
        # same HFOV/VFOV as the front camera (identical physical camera
        # model, just mounted at a different yaw). Shared by both lateral
        # cameras since only the mount quaternion differs between them;
        # used for ground-height filtering, mirroring the front camera's
        # own approach below in _get_observations().
        _lat_res = 8
        _lat_rows = torch.arange(_lat_res, device=self.device).float()
        _lat_cols = torch.arange(_lat_res, device=self.device).float()
        _lat_row_angles = (_lat_rows - (_lat_res - 1) / 2.0) / _lat_res * self._vfov
        _lat_col_angles = (_lat_cols - (_lat_res - 1) / 2.0) / _lat_res * self._hfov
        _lat_row_grid, _lat_col_grid = torch.meshgrid(_lat_row_angles, _lat_col_angles, indexing="ij")
        _lat_dir_x = torch.tan(_lat_col_grid)
        _lat_dir_y = torch.tan(_lat_row_grid)
        _lat_dir_z = torch.ones_like(_lat_dir_x)
        _lat_rays_cam = torch.stack([_lat_dir_x, _lat_dir_y, _lat_dir_z], dim=-1)
        _lat_rays_cam = _lat_rays_cam / torch.norm(_lat_rays_cam, dim=-1, keepdim=True)
        self._camera_rays_lateral = _lat_rays_cam.reshape(-1, 3)  # (64, 3)
        # ── Simplified-observation experiment: persistent buffer for each
        # obstacle's world-frame position. Populated per-env in _reset_idx()
        # (obstacles are static rigid bodies -- position only changes on
        # reset). N_OBS_HARDCODED must match len(obstacle_names) in
        # _reset_idx() (currently 5: 3 cylinders + 2 boxes) -- update both
        # together if the obstacle roster ever changes. ──
        N_OBS_HARDCODED = 5
        self._obstacle_world_pos = torch.zeros(self.num_envs, N_OBS_HARDCODED, 3, device=self.device)


        # Visual marker so the (otherwise invisible) target position can be
        # seen in the viewport during non-headless runs / play.py.
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/TargetMarker",
            markers={
                "target": sim_utils.SphereCfg(
                    radius=0.3,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
        self._target_marker = VisualizationMarkers(marker_cfg)
        self._target_marker.set_visibility(True)

        # ── Obstacle collision detection (distance-based approximation) ──────
        # NOTE: obstacles are static rigid bodies with real PhysX collision
        # (the drone WILL physically bounce off them), but until now nothing
        # in _get_dones()/_get_rewards() checked for obstacle contact — only
        # ground height was checked. This meant the drone could smash into an
        # obstacle, get knocked tumbling in any direction, and the episode
        # would simply continue as if nothing happened. This is a
        # distance-based approximation (bounding-sphere check) rather than a
        # true contact sensor: simpler to set up, and sufficient for now
        # since this signal doesn't need to match real-drone sim-to-real
        # consistency requirements (the real drone has no "collision boolean"
        # input — it only ever sees the depth image). Radii are generous
        # (obstacle bounding radius + ~0.3m for the drone's own prop span) so
        # we err toward ending the episode a bit early rather than missing a
        # real collision.
        drone_radius = 0.3
        # Obstacle base positions — randomized each reset in _reset_idx()
        self._obstacle_local_pos = torch.tensor([
            [2.5, 0.0, 2.0],    # ObstacleCyl0 (initial, will be randomized)
            [-2.5, 1.5, 2.0],   # ObstacleCyl1
            [1.5, -3.0, 2.5],   # ObstacleCyl2
            [-1.5, -2.0, 1.5],  # ObstacleBox0
            [3.5, 2.5, 2.0],    # ObstacleBox1
        ], device=self.device)
        self._obstacle_radius = torch.tensor([
            0.8 + drone_radius,               # Cyl0 (cylinder radius 0.8, enlarged for DQN Stage 1 -- was 0.3)
            0.3 + drone_radius,               # Cyl1 (cylinder radius 0.3)
            0.4 + drone_radius,               # Cyl2 (cylinder radius 0.4)
            0.707 + drone_radius,             # Box0 (1.0x1.0 half-diagonal ~0.707)
            1.06 + drone_radius,              # Box1 (1.5x1.5 half-diagonal ~1.06)
        ], device=self.device)

    def _setup_scene(self):
        # ── Robot ─────────────────────────────────────────────────────────────
        # NOTE: ThrusterCfg.dt must match this env's physics substep (sim.dt),
        # per the pattern established in track_position_state_based's
        # no_obstacle_env_cfg.py (self.scene.robot.actuators["thrusters"].dt =
        # self.sim.dt). ARL_ROBOT_1_CFG intentionally leaves dt unset (MISSING)
        # since it is shared across environments with potentially different
        # physics step sizes, so we set it here explicitly before constructing
        # the Articulation.
        robot_cfg = ARL_ROBOT_1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        robot_cfg.actuators["thrusters"].dt = self.cfg.sim.dt
        # DIAGNOSTIC: spawn 1m above ground to rule out the robot's default
        # (0,0,0) spawn point being embedded in / colliding with the ground
        # plane (also at z=0), which would physically prevent any movement
        # regardless of reward design. Both the reward-hacking run AND this
        # run showed the drone "emerging from underground and not moving" —
        # suggesting this may be a physical collision issue, not a learned
        # hovering strategy.
        robot_cfg.init_state.pos = (0.0, 0.0, 1.0)
        self._robot = Multirotor(robot_cfg)
        self.scene.articulations["robot"] = self._robot

        # ── Depth camera ──────────────────────────────────────────────────────
        self._camera = TiledCamera(TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/base_link/depth_camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.25, 0.0, 0.0),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
            data_types=["distance_to_camera"],
            # BUGFIX: horizontal_aperture=20.955 with focal_length=24.0 yields
            # only ~47 degrees of horizontal FOV -- far narrower than the real
            # deployment camera (OAK-D Lite stereo/depth pair, OV7251 sensors,
            # official Luxonis spec: 73 degree HFOV). This mismatch violates
            # the project's core sim-to-real requirement (training perception
            # must match the real drone's camera) and, combined with a narrow
            # rew_proximity detection cone (see obstacle_closeness below),
            # meant many laterally-offset obstacles were never perceived by
            # either the simulated FOV or the avoidance reward until far too
            # late. horizontal_aperture=35.5 with focal_length=24.0 reproduces
            # ~73 degrees HFOV, matching the real OAK-D Lite.
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0,
                horizontal_aperture=35.5, clipping_range=(0.1, 10.0),
            ),
            width=84, height=84,
        ))
        # ── DQN experiment: two additional cameras facing left/right (±90 deg
        # yaw relative to the forward camera), for a real 3-sensor Three-Beam-
        # style perception matching the paper's design. Using real cameras
        # (not raw ray-cast queries) so this stays sim-to-real viable -- ray-
        # cast has no physical hardware analog. Small 8x8 resolution since
        # only the global-minimum distance is used, not spatial detail;
        # cuts render cost vs. reusing 84x84. Direction (+90=left, -90=right)
        # follows this project's yaw convention (heading_angle uses standard
        # cos/sin construction, ROS-style body frame) but has NOT yet been
        # empirically verified -- swap the two mount quats below if a smoke
        # test shows them reversed.
        _yaw_left = torch.tensor([math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)], device=self.device)
        _yaw_right = torch.tensor([math.cos(-math.pi/4), 0.0, 0.0, math.sin(-math.pi/4)], device=self.device)
        _cam_mount_front = torch.tensor([0.5, -0.5, 0.5, -0.5], device=self.device)
        self._camera_mount_quat_left = quat_mul(_yaw_left, _cam_mount_front)
        self._camera_mount_quat_right = quat_mul(_yaw_right, _cam_mount_front)

        self._camera_left = TiledCamera(TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/base_link/depth_camera_left",
            offset=TiledCameraCfg.OffsetCfg(
                # Lateral offset (0.25m along body y) to clear the airframe,
                # mirroring the forward camera's 0.25m x-offset for the same
                # reason. Smoke test showed d_left/d_right=0.18m at pos=(0,0,0)
                # -- almost certainly self-occlusion (identical on both
                # sides, too close to be a real obstacle, hard-case disabled
                # this run so no forced near-obstacle start).
                pos=(0.0, 0.25, 0.0),
                rot=tuple(self._camera_mount_quat_left.tolist()),
                convention="ros",
            ),
            data_types=["distance_to_camera"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0,
                horizontal_aperture=35.5, clipping_range=(0.1, 10.0),
            ),
            width=8, height=8,
        ))
        self._camera_right = TiledCamera(TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/base_link/depth_camera_right",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, -0.25, 0.0),
                rot=tuple(self._camera_mount_quat_right.tolist()),
                convention="ros",
            ),
            data_types=["distance_to_camera"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0,
                horizontal_aperture=35.5, clipping_range=(0.1, 10.0),
            ),
            width=8, height=8,
        ))
        self.scene.sensors["depth_camera"] = self._camera



        # ── Contact sensor (detects real physical collisions) ─────────────────
        self._contact_sensor = ContactSensor(ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*",
            history_length=3,
            update_period=0.0,
            track_air_time=False,
        ))
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # ── Obstacles ─────────────────────────────────────────────────────────
        def make_cylinder(name, radius, height, pos, color):
            obj = RigidObject(RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/{name}",
                spawn=sim_utils.CylinderCfg(
                    radius=radius, height=height, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            ))
            self.scene.rigid_objects[name] = obj
            return obj

        def make_box(name, size, pos, color):
            obj = RigidObject(RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/{name}",
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            ))
            self.scene.rigid_objects[name] = obj
            return obj

        # Positions rescaled to fit within the new ±5m target range (previously
        # spanned roughly x:-3..7, y:-6..5 — a ~10m-wide field, mismatched
        # against the ±15m target range and now against the ±5m one too).
        # Scaled down by ~0.5x to keep obstacles genuinely in the drone's
        # likely flight path to randomly-sampled targets within ±5m.
        make_cylinder("ObstacleCyl0", 0.3, 4.0, (2.5, 0.0, 2.0), (0.2, 0.6, 0.2))
        make_cylinder("ObstacleCyl1", 0.3, 4.0, (-2.5, 1.5, 2.0), (0.2, 0.6, 0.2))
        make_cylinder("ObstacleCyl2", 0.4, 5.0, (1.5, -3.0, 2.5), (0.3, 0.5, 0.3))
        make_box("ObstacleBox0", (1.0, 1.0, 3.0), (-1.5, -2.0, 1.5), (0.5, 0.4, 0.3))
        make_box("ObstacleBox1", (1.5, 1.5, 4.0), (3.5, 2.5, 2.0), (0.5, 0.4, 0.3))

        # ── Ground + lights ───────────────────────────────────────────────────
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=750.0, color=(1.0, 1.0, 1.0))
        light_cfg.func("/World/skyLight", light_cfg)

        # Clone + filter
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()

    def _apply_action(self):
        """Simulated PX4-style velocity controller (geometric control).

        REDESIGNED to match "Vision-Based Deep RL for Autonomous Drone
        Flight" (Saran, UC Berkeley EECS-2023-280): action space reduced
        from 4 to 3 values -- forward velocity (body x-axis, i.e. the
        direction the camera points), vertical velocity (world z), and
        yaw rate. NO lateral (y) velocity is commanded at all -- to move
        left, the drone must first yaw left, then move forward. This is
        the exact constraint the paper uses, validated on real DJI
        Matrice 300 hardware, specifically because a forward-only-facing
        camera can only usefully perceive obstacles in the direction of
        travel; allowing free lateral strafing would let the drone move
        into directions its camera cannot see.

        The velocity->attitude->thrust cascade itself (geometric control,
        Lee et al. 2010 / Faessler et al. differential flatness,
        independently verified via standalone test script) is unchanged
        from the previous version.
        """
        d = self._robot.data
        cfg = self.cfg
        squashed = torch.tanh(self._actions)
        # BUGFIX (v2): vx_cmd was allowed to go negative (tanh squashes to
        # [-1,1]), which the policy learned to exploit -- flying BACKWARD
        # (heading_error near +/-180 deg while distance still decreases)
        # to farm rew_ang_vel_exp (reward for near-zero angular velocity)
        # without paying the cost of turning to face the goal, since
        # rew_yaw_aligned's weight (0.04) was too small to discourage this.
        # Observed via play.py: 5/6 sampled episodes flew backward for
        # most of the episode. This also violates the core safety
        # assumption inherited from the reference paper -- the
        # forward-only camera must point in the direction of travel to see
        # obstacles; a negative vx defeats that entirely. Fixed at the
        # source (not just via reward shaping) by remapping vx_cmd's
        # squashing function to [0, 1] instead of [-1, 1], making backward
        # flight physically impossible rather than merely discouraged.
        vx_cmd = ((torch.tanh(self._actions[:, 0]) + 1.0) / 2.0) * cfg.max_xy_vel      # forward only, non-negative, body x-axis
        vz_cmd = squashed[:, 1] * cfg.max_z_vel
        yaw_rate_cmd = squashed[:, 2] * cfg.max_yaw_rate

        # -- Yaw-rate slew-rate limit (method 1) -- hard physical clamp,
        # applied regardless of what the policy/argmax picked.
        yaw_rate_cmd = torch.clamp(
            yaw_rate_cmd,
            self._prev_yaw_rate_cmd - cfg.max_yaw_rate_delta,
            self._prev_yaw_rate_cmd + cfg.max_yaw_rate_delta,
        )
        self._prev_yaw_rate_cmd = yaw_rate_cmd.clone()

        # -- Altitude-hold safety net (method 2, see cfg.altitude_hold_kp
        # comment above for full rationale) -- purely additive position
        # feedback on top of whatever vz the policy commanded; with
        # altitude_hold_kp=0 this is an exact no-op (old behavior).
        if cfg.altitude_hold_kp != 0.0:
            altitude_error = cfg.altitude_hold_target - d.root_pos_w[:, 2]
            vz_cmd = vz_cmd + cfg.altitude_hold_kp * altitude_error
            vz_cmd = torch.clamp(vz_cmd, -cfg.max_z_vel, cfg.max_z_vel)

        quat = d.root_quat_w
        euler = euler_xyz_from_quat(quat)
        yaw_now = wrap_to_pi(euler[2])
        cos_yaw = torch.cos(yaw_now)
        sin_yaw = torch.sin(yaw_now)


        # Forward-only velocity command (body x-axis) rotated into world frame.
        # No vy component at all (matches reference paper's constraint).
        vx_world = vx_cmd * cos_yaw
        vy_world = vx_cmd * sin_yaw
        v_cmd_world = torch.stack([vx_world, vy_world, vz_cmd], dim=-1)  # (N,3)

        v_actual_world = d.root_lin_vel_w
        vel_error = v_cmd_world - v_actual_world
        a_fb = torch.clamp(vel_error * cfg.vel_ctrl_kp, -cfg.max_accel, cfg.max_accel)

        gravity_vec = torch.zeros(self.num_envs, 3, device=self.device)
        gravity_vec[:, 2] = -self._gravity
        a_des_world = a_fb - gravity_vec
        f_des_world = self._drone_mass * a_des_world
        thrust_mag = torch.norm(f_des_world, dim=-1).clamp(min=0.5, max=22.0)

        zb_des = f_des_world / (torch.norm(f_des_world, dim=-1, keepdim=True) + 1e-8)
        x_c_des = torch.stack([cos_yaw, sin_yaw, torch.zeros_like(yaw_now)], dim=-1)
        yb_des = torch.cross(zb_des, x_c_des, dim=-1)
        yb_des = yb_des / (torch.norm(yb_des, dim=-1, keepdim=True) + 1e-8)
        xb_des = torch.cross(yb_des, zb_des, dim=-1)
        xb_des = xb_des / (torch.norm(xb_des, dim=-1, keepdim=True) + 1e-8)
        R_des = torch.stack([xb_des, yb_des, zb_des], dim=-1)
        q_des = self._rotmat_to_quat(R_des)

        quat_conj = torch.cat([quat[:, 0:1], -quat[:, 1:4]], dim=-1)
        q_err = quat_mul(quat_conj, q_des)
        sign = torch.sign(q_err[:, 0:1])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        att_err = 2.0 * sign * q_err[:, 1:4]

        ang_vel = d.root_ang_vel_b
        tau_roll = cfg.att_ctrl_kp * att_err[:, 0] - cfg.att_ctrl_kd * ang_vel[:, 0]
        tau_pitch = cfg.att_ctrl_kp * att_err[:, 1] - cfg.att_ctrl_kd * ang_vel[:, 1]
        max_torque = 0.13 * 3.0
        tau_roll = torch.clamp(tau_roll, -max_torque, max_torque)
        tau_pitch = torch.clamp(tau_pitch, -max_torque, max_torque)
        tau_yaw = cfg.yaw_rate_ctrl_kp * (yaw_rate_cmd - ang_vel[:, 2])

        wrench = torch.stack([thrust_mag, tau_roll, tau_pitch, tau_yaw], dim=-1)
        thrust_per_motor = torch.matmul(wrench, self._alloc_inv.T)
        thrust_per_motor = torch.clamp(thrust_per_motor, 0.0, 6.0)
        self._robot.set_thrust_target(thrust_per_motor)

    @staticmethod
    def _rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
        """Batched rotation matrix (N,3,3) -> quaternion (N,4) in (w,x,y,z).
        Shepperd's method. Independently verified via standalone test script."""
        m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
        m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
        m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
        trace = m00 + m11 + m22
        N = R.shape[0]
        q = torch.zeros(N, 4, device=R.device)

        cond1 = trace > 0
        s1 = torch.sqrt(torch.clamp(trace + 1.0, min=1e-8)) * 2
        q[cond1, 0] = 0.25 * s1[cond1]
        q[cond1, 1] = (m21[cond1] - m12[cond1]) / s1[cond1]
        q[cond1, 2] = (m02[cond1] - m20[cond1]) / s1[cond1]
        q[cond1, 3] = (m10[cond1] - m01[cond1]) / s1[cond1]

        cond2 = (~cond1) & (m00 >= m11) & (m00 >= m22)
        s2 = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-8)) * 2
        q[cond2, 0] = (m21[cond2] - m12[cond2]) / s2[cond2]
        q[cond2, 1] = 0.25 * s2[cond2]
        q[cond2, 2] = (m01[cond2] + m10[cond2]) / s2[cond2]
        q[cond2, 3] = (m02[cond2] + m20[cond2]) / s2[cond2]

        cond3 = (~cond1) & (~cond2) & (m11 >= m22)
        s3 = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=1e-8)) * 2
        q[cond3, 0] = (m02[cond3] - m20[cond3]) / s3[cond3]
        q[cond3, 1] = (m01[cond3] + m10[cond3]) / s3[cond3]
        q[cond3, 2] = 0.25 * s3[cond3]
        q[cond3, 3] = (m12[cond3] + m21[cond3]) / s3[cond3]

        cond4 = (~cond1) & (~cond2) & (~cond3)
        s4 = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=1e-8)) * 2
        q[cond4, 0] = (m10[cond4] - m01[cond4]) / s4[cond4]
        q[cond4, 1] = (m02[cond4] + m20[cond4]) / s4[cond4]
        q[cond4, 2] = (m12[cond4] + m21[cond4]) / s4[cond4]
        q[cond4, 3] = 0.25 * s4[cond4]

        return q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)

    def _get_observations(self) -> dict:
        """v2 design: 12-value body-frame state (adopted from IsaacLab's
        official quadcopter direct-RL example) + depth image. State =
        root_lin_vel_b[3] + root_ang_vel_b[3] + projected_gravity_b[3] +
        desired_pos_b[3]. Body-frame velocity gives the policy explicit
        motion feedback it previously lacked; projected_gravity_b encodes
        full 3D attitude without the +/-180 deg discontinuity a raw yaw
        angle has; desired_pos_b encodes goal direction+distance jointly
        in one frame-consistent quantity, replacing the separately-computed
        world-frame pos_error/heading_error of the original 7-dim design.
        """
        d = self._robot.data
        pos = d.root_pos_w - self.scene.env_origins        # (N,3)
        quat = d.root_quat_w                                # (N,4)

        lin_vel_b = d.root_lin_vel_b                        # (N,3)
        ang_vel_b = d.root_ang_vel_b                         # (N,3)
        gravity_b = d.projected_gravity_b                    # (N,3)

        desired_pos_b, _ = subtract_frame_transforms(
            pos, quat, self._target_pos
        )  # (N,3), target position expressed in the drone's own body frame

        # Still needed below for the depth-based obstacle-distance ray cast
        # and for the debug print (kept for continuity with the old logs).
        euler = euler_xyz_from_quat(quat)
        yaw = wrap_to_pi(euler[2])
        target_vec = self._target_pos - pos
        heading_error = wrap_to_pi(torch.atan2(target_vec[:, 1], target_vec[:, 0]) - yaw)

        state = torch.cat([
            lin_vel_b,     # 3
            ang_vel_b,     # 3
            gravity_b,     # 3
            desired_pos_b, # 3
        ], dim=-1)  # (N, 12)

        depth = self._camera.data.output["distance_to_camera"].clone()
        depth[depth == float("inf")] = 10.0
        depth = torch.clamp(depth / 10.0, 0.0, 1.0)          # (N, 84, 84, 1)
        depth_chw = depth.permute(0, 3, 1, 2).contiguous()   # (N, 1, 84, 84)
        self._last_depth = depth_chw.clone()
        # ── DQN: ground-filtered nearest-obstacle distance for the left/
        # right cameras, mirroring the front camera's is_ground logic. Each
        # camera's rays must be rotated using ITS OWN mount quaternion (not
        # the front's), since ground-hit detection depends on each ray's
        # true world-frame direction, which differs by camera.
        def _lateral_min_dist(depth_raw_hw, mount_quat):
            combined = quat_mul(quat, mount_quat.unsqueeze(0).expand(self.num_envs, -1))
            rays_world = quat_apply(
                combined.unsqueeze(1).expand(-1, self._camera_rays_lateral.shape[0], -1).reshape(-1, 4),
                self._camera_rays_lateral.unsqueeze(0).expand(self.num_envs, -1, -1).reshape(-1, 3),
            ).reshape(self.num_envs, -1, 3)
            ray_dz = rays_world[..., 2]
            actual = depth_raw_hw.reshape(self.num_envs, -1)
            hit_height = pos[:, 2:3] + actual * ray_dz
            is_ground = torch.abs(hit_height) < self._ground_height_tolerance
            masked = torch.where(is_ground, torch.full_like(actual, 10.0), actual)
            return masked.min(dim=-1).values

        depth_left_raw = self._camera_left.data.output["distance_to_camera"].clone()
        depth_left_raw[depth_left_raw == float("inf")] = 10.0
        depth_right_raw = self._camera_right.data.output["distance_to_camera"].clone()
        depth_right_raw[depth_right_raw == float("inf")] = 10.0
        d_left = _lateral_min_dist(depth_left_raw[..., 0], self._camera_mount_quat_left)
        d_right = _lateral_min_dist(depth_right_raw[..., 0], self._camera_mount_quat_right)


        # ── Nearest-obstacle distance (ground-excluded), moved here from
        # _get_rewards() so it's computed once per step and shared by both
        # the reward (obstacle penalty) and the critic's privileged input. ──
        camera_mount_quat = self._camera_mount_quat
        combined_quat = quat_mul(quat, camera_mount_quat.unsqueeze(0).expand(self.num_envs, -1))
        rays_world = quat_apply(
            combined_quat.unsqueeze(1).expand(-1, self._camera_rays.shape[0], -1).reshape(-1, 4),
            self._camera_rays.unsqueeze(0).expand(self.num_envs, -1, -1).reshape(-1, 3),
        ).reshape(self.num_envs, -1, 3)
        cam_height = pos[:, 2:3]
        ray_dz = rays_world[..., 2]
        _center = self._last_depth[:, 0, :, :]
        actual_dist = _center.reshape(self.num_envs, -1) * 10.0
        hit_height = cam_height + actual_dist * ray_dz
        is_ground = torch.abs(hit_height) < self._ground_height_tolerance
        masked_depth_m = torch.where(
            is_ground, torch.full_like(actual_dist, 10.0), actual_dist
        )
        min_dist_m = masked_depth_m.min(dim=-1).values
        self._last_min_obstacle_dist = min_dist_m.clone()  # cached, meters, unnormalized
        # ── DQN: assemble the paper-mirroring 5-dim state
        # [d_front, d_left, d_right, dx, dy]. d_front reuses this file's
        # existing ground-filtered front-camera distance (min_dist_m,
        # already in meters). d_left/d_right were computed earlier in this
        # function (lines ~822-823, before this point). dx/dy are the
        # target's body-frame x/y components, reusing desired_pos_b
        # computed earlier in this function (z ignored per the same-height
        # assumption for this experiment).
        dqn_state = torch.cat([
            min_dist_m.unsqueeze(-1),      # d_front
            d_left.unsqueeze(-1),
            d_right.unsqueeze(-1),
            desired_pos_b[:, 0:1],          # dx (body frame)
            desired_pos_b[:, 1:2],          # dy (body frame)
        ], dim=-1)  # (N, 5)
        self._last_dqn_state = dqn_state.clone()
        # ── Simplified-observation experiment: nearest-visible-obstacle
        # relative position (body frame, 3-dim), ground-truth geometry
        # instead of the depth camera/CNN. Only active obstacles counted.
        # Visibility reuses the depth camera's HFOV/VFOV, widened per
        # obstacle by atan(radius/distance) so a partially-visible obstacle
        # counts as detected, matching the camera's pixel-level detection.
        # Default (10,0,0) meters when nothing visible. ──
        n_active_obs = self.cfg.num_active_obstacles
        n_obs_total = self._obstacle_world_pos.shape[1]
        best_rel_b = torch.zeros(self.num_envs, 3, device=self.device)
        best_rel_b[:, 0] = 10.0
        best_dist = torch.full((self.num_envs,), 10.0, device=self.device)
        for oi in range(min(n_active_obs, n_obs_total)):
            obs_pos_i = self._obstacle_world_pos[:, oi, :]
            rel_b, _ = subtract_frame_transforms(pos, quat, obs_pos_i)
            rel_cam = quat_apply_inverse(
                self._camera_mount_quat.unsqueeze(0).expand(self.num_envs, -1), rel_b
            )
            dist_i = torch.norm(rel_cam, dim=-1)
            forward = rel_cam[:, 2]
            h_angle = torch.atan2(rel_cam[:, 0], rel_cam[:, 2].clamp(min=1e-6))
            v_angle = torch.atan2(rel_cam[:, 1], rel_cam[:, 2].clamp(min=1e-6))
            radius_margin = torch.atan2(self._obstacle_radius[oi], dist_i.clamp(min=1e-3))
            visible = (
                (forward > 0.0)
                & (torch.abs(h_angle) < (self._hfov / 2.0 + radius_margin))
                & (torch.abs(v_angle) < (self._vfov / 2.0 + radius_margin))
            )
            is_nearer = visible & (dist_i < best_dist)
            best_rel_b = torch.where(is_nearer.unsqueeze(-1), rel_b, best_rel_b)
            best_dist = torch.where(is_nearer, dist_i, best_dist)
        self._nearest_visible_obstacle_b = best_rel_b
        simple_state = torch.cat([state, best_rel_b], dim=-1)  # (N, 15)

        # ── Critic-only privileged observation: base 12-dim state + nearest-
        # obstacle distance (normalized by /10.0 to roughly match other
        # position-scale quantities in the state). ──
        critic_state = torch.cat([
            state,
            (min_dist_m / 10.0).unsqueeze(-1),
        ], dim=-1)  # (N, 13)

        if self.episode_length_buf[0] % 50 == 0:
            dist_dbg = torch.norm(target_vec[0]).item()
            print(f"[step {self.episode_length_buf[0].item():4d}] "
                  f"heading_error={math.degrees(heading_error[0].item()):6.1f}\u00b0 dist={dist_dbg:.2f}m")

        self._prev_actions = self._actions.clone()
        self._target_marker.visualize(self._target_pos + self.scene.env_origins)
        return {"policy": simple_state, "dqn_state": dqn_state, "depth": depth_chw, "critic": critic_state}

    def _get_rewards(self) -> torch.Tensor:
        """v2 reward redesign: exp-shaped dense terms operating on the
        body-frame state (desired_pos_b, lin_vel_b, ang_vel_b, gravity_b),
        adapted from an internal early-design doc. All dense terms are
        scaled down (~50x from the doc's raw weights) so a "perfect"
        dense-reward episode cannot out-accumulate the terminal
        rew_reached -- see cfg class notes.
        """
        d = self._robot.data
        cfg = self.cfg
        pos = d.root_pos_w - self.scene.env_origins
        quat = d.root_quat_w
        lin_vel_b = d.root_lin_vel_b
        ang_vel_b = d.root_ang_vel_b
        gravity_b = d.projected_gravity_b

        desired_pos_b, _ = subtract_frame_transforms(pos, quat, self._target_pos)
        dist = torch.norm(desired_pos_b, dim=-1)

        # ── Distance: potential-based, progress-only reward (v2 fix for the
        # "orbits near target without finishing" exploit). Previously this
        # was exp(-k*dist)*weight computed fresh every step, which paid out
        # repeatedly for merely BEING close, so a policy that circled at a
        # fixed 1-2m radius could accumulate far more reward (verified via
        # play.py: 200+ over 700 steps) than the one-time rew_reached=100 for
        # actually finishing. Now the reward equals the INCREASE in
        # exp(-k*dist) relative to the closest distance ever achieved this
        # episode (a potential function) -- reward is paid out once per unit
        # of real progress and is exactly 0 while lingering at a distance
        # already visited, no matter how long. Total achievable reward from
        # this term is bounded per-episode regardless of path length. ──
        # ── DQN Stage 0 MINIMAL REWARD v5: switched from potential-based
        # (record-breaking only) distance shaping to a dense per-step
        # negative-distance reward, matching the field's most common,
        # well-validated baseline for point-to-goal RL tasks (OpenAI Gym
        # Reacher: reward_dist = -norm(fingertip - target), every step,
        # not a record/delta). Rationale: potential-based shaping was
        # originally adopted to prevent circling/reward-farming, but that
        # risk was tied to non-potential progress terms (rew_vel_progress
        # etc.) that have since been removed entirely from this minimal
        # reward. With only distance + terminal reached/crashed left,
        # potential-based shaping's main remaining effect was making the
        # per-step signal SPARSE (zero unless a new personal-best distance
        # is set), likely too sparse for a randomly-initialized DQN with
        # epsilon-driven exploration to bootstrap learning from. Dense
        # -k*dist gives non-zero, informative feedback every single step
        # regardless of history, AND makes "wander far from target"
        # costly every step it persists -- this directly subsumes what
        # rew_out_of_bounds was patched in for (see v4 comment), without
        # needing a separate terminal-only penalty: 1000 steps stuck near
        # the 20m out-of-bounds threshold accumulates -0.01*20*1000=-200,
        # already worse than a single -50 crash, so wandering off is no
        # longer "free" the way it was under the v3/v4 potential-based +
        # terminal-oob-only design. k=0.01 chosen so per-step cost stays
        # small relative to terminal rewards (reached=+100, crashed=-50)
        # while accumulated wandering cost dominates over an episode. ──
        DENSE_DISTANCE_K = 0.01
        rew_distance = -DENSE_DISTANCE_K * dist
        new_min_dist = torch.minimum(self._min_dist_so_far, dist)
        self._min_dist_so_far = new_min_dist  # kept for other diagnostics/logging that reference it
        # ── DQN orbit fix: rew_vel_progress and rew_yaw_aligned (below)
        # previously paid out on instantaneous state alone (velocity/heading
        # toward target), without requiring genuine progress -- unlike
        # rew_distance above, which was already made potential-based to fix
        # the same class of exploit (see the v2 fix comment). Diagnosed via
        # diagnose_orbit_dqn.py on real rollouts: 30 episodes, 0 reached,
        # timeout episodes showed rew_distance~0 (correctly not farmable)
        # but rew_vel_progress+rew_yaw_aligned ranging -22 to +25, driving
        # total reward independent of real progress -- confirming the
        # policy could profit from circling or even passive drifting.
        # Gate: only pay these two terms on steps that actually set a new
        # closest-distance record (reuses rew_distance's own progress
        # signal rather than inventing a new potential function for
        # heading, which is harder to define cleanly than distance). ──

        # ── Angular velocity: penalize (via exp reward) spinning/instability,
        # regardless of yaw component specifically -- uses full 3-axis
        # ang_vel_b instead of the old single yaw_rate. NOTE: previously gated
        # by v_along (progress toward goal) via ang_vel_progress_gate_k, but
        # that gate used the 3D-composite v_along, which stayed positive during
        # horizontal progress even while the drone climbed uncontrolled in z (the
        # gate was fooled by horizontal progress masking vertical error), so it
        # failed to fix the climb-away issue it was added for. It also likely
        # suppressed this term during legitimate low-speed turn-to-align phases
        # (where v_along is naturally near/below zero), causing a new stationary-
        # spin-in-place regression. Reverted to the original unconditional form;
        # ang_vel_progress_gate_k in cfg is now unused (kept, not deleted, in case
        # a future attempt wants a better-designed progress gate). ──
        ang_vel_sq = torch.sum(ang_vel_b**2, dim=-1)
        rew_ang_vel = torch.exp(-cfg.ang_vel_exp_k * ang_vel_sq) * cfg.rew_ang_vel_exp
        target_dir_b = desired_pos_b / (dist.unsqueeze(-1) + 1e-8)
        v_along = torch.sum(lin_vel_b * target_dir_b, dim=-1, keepdim=True)
        # ── Progress reward: dense, direct incentive for velocity toward
        # the goal (not gated on new-record distance like rew_distance's
        # potential-based sibling attempt was) -- gives DQN a denser
        # bootstrap signal than distance-only shaping. Can go negative if
        # the drone is moving away, which is an intentional real penalty,
        # not just an absence of reward. See cfg.rew_vel_progress note on
        # why this is safe to re-enable now (backward-flight exploit
        # closed at the source in _apply_action). ──
        rew_progress = v_along.squeeze(-1) * cfg.rew_vel_progress
        # ── Lateral velocity: only penalize the velocity component
        # PERPENDICULAR to the goal direction, not speed toward the goal
        # itself. This is the key fix vs. a naive "penalize all velocity"
        # term (which would fight the distance reward and reward standing
        # still) -- full-speed flight straight at the goal is NOT penalized. ──
        v_lateral = lin_vel_b - v_along * target_dir_b
        lateral_speed_sq = torch.sum(v_lateral**2, dim=-1)
        rew_lateral_vel = torch.exp(-cfg.lateral_vel_exp_k * lateral_speed_sq) * cfg.rew_lateral_vel_exp

        # ── v2 fix: direct reward for closing distance, active even
        # while the drone hasn't finished turning to face the goal. Uses
        # v_along directly (not squared/exp'd) -- can be negative if the
        # drone is moving away, which is fine (a real penalty for making
        # things worse, not just an absence of reward). ──
        # ── Obstacle-proximity decay on rew_vel_progress: squared decay,
        # active within 2.0m of nearest obstacle, floor=0.05 (retains a small
        # residual incentive so the drone isn't rewarded for stopping dead
        # next to an obstacle). Chosen via offline dwell-tradeoff analysis on
        # real play.py data: squared decay at far=2.0m was the only candidate
        # (among linear/squared x far=1.0/1.5/2.0) that flipped the average
        # net reward during obstacle-dwell periods negative; floor=0.05 stays
        # safely below the ~0.12 threshold where the net flips back positive. ──
        velprog_decay_floor = 0.05
        velprog_decay_far = 2.0
        velprog_decay_raw = torch.clamp(self._last_min_obstacle_dist / velprog_decay_far, min=0.0, max=1.0)
        velprog_decay_scale = velprog_decay_floor + (1.0 - velprog_decay_floor) * (velprog_decay_raw ** 2)

        # ── Yaw alignment: cosine of the angle between the body-frame
        # forward axis and the horizontal direction to the goal, using
        # desired_pos_b directly (no separate world-frame heading_error
        # computation, no +/-180 deg discontinuity). ──
        horiz_dist = torch.norm(desired_pos_b[:, :2], dim=-1).clamp(min=1e-6)
        yaw_cos = desired_pos_b[:, 0] / horiz_dist  # +1 = facing goal, -1 = facing away

        # ── DQN v2 action space: re-added dense yaw-alignment reward.
        # Root cause of turn-left/turn-right being nearly indistinguishable
        # in Q-values (diagnosed via dqn_diagnose.py --mode=trace): with
        # forward and turning now fully decoupled into separate actions,
        # a pure-turn action produces NO change in dist, so rew_distance
        # gives zero signal about whether a turn was correct -- "which way
        # to turn" had no direct reward at all, only a delayed, diluted
        # signal via bootstrapping once a later forward action happened to
        # close distance. This differs from the old v1 orbiting exploit:
        # that required BOTH rew_yaw_aligned AND rew_vel_progress together
        # (an instantaneous-velocity-toward-target term that let circling
        # farm reward without ever truly progressing). rew_vel_progress is
        # NOT being reintroduced here. rew_distance is dense (-k*dist,
        # not potential-based/record-only), so standing still while facing
        # the target still accumulates a steady negative cost every step
        # (unlike v1's potential-based rew_distance, which gave zero cost
        # for lingering) -- "align but don't move" remains a losing
        # strategy under this combination. Weight kept an order of
        # magnitude below rew_distance's effective per-step scale. ──
        # ── DQN v2: gated yaw-alignment reward. Raising the flat weight
        # from 0.02->0.06 fixed the "gives up turning too early" problem
        # but created the opposite exploit: once roughly facing the
        # target, oscillating left/right near that heading kept farming
        # rew_yaw_aligned indefinitely without ever committing to move
        # forward (0% reached, confirmed via trace: dy stabilized near
        # -5.4 for 30+ steps while action flip-flopped left/right).
        # Fix: zero out this reward once heading is already well-aligned
        # (yaw_cos > 0.9, ~25 degrees), so "hover near correct heading"
        # stops being a farmable steady-state -- the only way to keep
        # earning reward past that point is rew_distance, which requires
        # genuinely closing distance via the forward action.
        YAW_ALIGNED_THRESHOLD = 0.9
        rew_yaw_aligned = torch.where(
            yaw_cos > YAW_ALIGNED_THRESHOLD,
            torch.zeros_like(yaw_cos),
            yaw_cos * 0.06,
        )

        # ── Flat orientation: penalize non-level attitude via the
        # horizontal components of gravity projected into the body frame
        # (both are 0 when perfectly level; gravity_b = [0,0,-1] level). ──
        rew_flat = -(gravity_b[:, 0]**2 + gravity_b[:, 1]**2) * cfg.rew_flat

        # ── Obstacle penalty: speed-coupled proximity penalty. See cfg note
        # on rew_obstacle for full history/rationale. ──
        min_dist_m = self._last_min_obstacle_dist
        proximity_intrusion = torch.clamp(2.0 - min_dist_m, min=0.0)  # 0 to 2, 0=at 2.0m boundary, 2=touching
        speed = torch.norm(lin_vel_b, dim=-1)
        obstacle_penalty = -(proximity_intrusion * speed)
        rew_obstacle = obstacle_penalty * cfg.rew_obstacle

        # ── Unnecessary-vz penalty: penalize |vz_cmd| only when no obstacle
        # is within the same 1.0m threshold rew_obstacle uses. Gated (not an
        # unconditional height-tracking penalty) so vertical maneuvering for
        # obstacle avoidance remains completely free -- see cfg note above. ──
        vz_squashed = torch.tanh(self._actions[:, 1])
        no_obstacle_nearby = (min_dist_m > 1.0).float()
        rew_unnecessary_vz = -torch.abs(vz_squashed) * no_obstacle_nearby * cfg.rew_unnecessary_vz

        # ── Height-alignment progress: step-to-step potential-based shaping,
        # NOT gated on obstacle proximity (unlike rew_unnecessary_vz above).
        # Telescopes to zero over any round-trip climb/descend cycle, so
        # legitimate repeated obstacle-driven altitude changes are never
        # penalized net -- only rewards/penalizes the INSTANTANEOUS step's
        # progress toward/away from target height. See cfg note for context. ──
        height_diff = torch.abs(self._target_pos[:, 2] - pos[:, 2])
        rew_height_progress = (self._prev_height_diff - height_diff) * cfg.rew_height_progress
        self._prev_height_diff = height_diff

        # ── Action smoothness: penalize large action deltas / magnitudes,
        # directly targeting the "policy kept switching between turning
        # left and right" noisy-output failure mode noted in the reference
        # paper's own real-world tests. ──
        action_delta_sq = torch.sum((self._actions - self._prev_actions)**2, dim=-1)
        rew_action_rate = -action_delta_sq * cfg.rew_action_rate
        action_mag_sq = torch.sum(self._actions**2, dim=-1)
        rew_action_magnitude = -action_mag_sq * cfg.rew_action_magnitude

        # ── Terminal rewards (one-time, added on the terminating step) ──
        reached = dist < 0.75
        ground_crash = pos[:, 2] < cfg.min_height
        net_forces = self._contact_sensor.data.net_forces_w_history
        force_mag = torch.norm(net_forces, dim=-1)
        any_contact = torch.any(force_mag > 1.0, dim=(1, 2))
        crashed = ground_crash | any_contact
        timed_out = self.episode_length_buf >= self.max_episode_length
        out_of_bounds = dist > cfg.max_dist_from_start

        rew_reached = reached.float() * cfg.rew_reached
        rew_crashed = crashed.float() * cfg.rew_crashed

        # ── DQN diagnostic caches (kept for eval scripts written during the
        # over-complex reward era; harmless to keep populated). ──
        self._last_ground_crash = ground_crash.clone()
        self._last_any_contact = any_contact.clone()
        self._last_out_of_bounds = out_of_bounds.clone()
        self._last_reached_flag = reached.clone()
        self._last_rew_distance = rew_distance.clone()

        # ── DQN Stage 0 MINIMAL REWARD v3: rew_action_rate added back in,
        # everything else stays dropped (see v2 comment above for the full
        # rationale on the other terms). Root-caused via a controlled
        # open-loop test (test_altitude_forward_only.py, no policy/Q-net
        # involved): sustained single-direction turning for 400 steps held
        # altitude essentially flat (0.9966->0.9454m, no crash), but
        # alternating left/right turn EVERY STEP crashed multiple envs
        # within ~40 steps (altitude 1.0->0.34m). This exactly matches the
        # reference paper's own documented real-hardware failure mode
        # ("policy kept switching between turning left and right") that
        # rew_action_rate was originally added to fight -- removing it in
        # the v1/v2 minimal rewrite reopened this specific failure path.
        # Confirmed independently via simulation, not just inherited on
        # faith from the paper's claim. ──
        action_delta_sq = torch.sum((self._actions - self._prev_actions)**2, dim=-1)
        rew_action_rate = -action_delta_sq * cfg.rew_action_rate
        # ── DQN Stage 0 MINIMAL REWARD v4: rew_out_of_bounds added back in.
        # Root-caused via diagnose_orbit_dqn.py on the v3-trained (2500ep)
        # checkpoint: 26/30 episodes (87%) ended via out_of_bounds with
        # total_reward ~0.0 -- flying out of bounds was completely free
        # under v3's reward (rew_oob was dropped in the v1/v2 minimal
        # rewrite and never restored). With zero cost for wandering off
        # and terminating, the policy had no incentive to prefer flying
        # toward the target over any other stable direction -- explains
        # why MA50 plateaued around -27 after episode ~1000 despite
        # rew_action_rate having already fixed the crash-rate problem
        # (contact_crash dropped to 2/30 in this same diagnostic run). ──
        # rew_oob removed: dense per-step distance cost above already makes
        # wandering far away costly every step it continues, superseding
        # the need for a separate terminal-only out-of-bounds penalty.
        #
        # MINIMAL-REWARD PATCH (post orbit/crash diagnosis, 30-ep eval:
        # 23/30 contact_crash, orbiting with no clear approach behavior):
        # two terms were being computed above but silently dropped from
        # this return statement.
        #   - rew_obstacle: computed at line ~1057 but never summed in.
        #     The policy had ZERO shaped incentive to avoid obstacles --
        #     only the one-time rew_crashed=-50 on actual contact. Adding
        #     it back gives a continuous "too close" signal before impact.
        #   - rew_progress (new, see above): distance-only shaping gives
        #     no direct incentive to face/close on the target, which is
        #     the most likely cause of the observed orbiting -- there was
        #     no per-step reward specifically for moving toward the goal,
        #     only a flat cost proportional to how far away it currently is.
        # Deliberately NOT restoring rew_ang_vel / rew_lateral_vel /
        # rew_flat / rew_height_progress / rew_unnecessary_vz /
        # rew_action_magnitude / rew_yaw_aligned here -- none of those
        # have a direct causal link to the two failure modes above, and
        # bundling unrelated changes together would make it impossible to
        # tell which change fixed (or didn't fix) what in the next eval.
        # ── REVERTED to Stage 0's proven minimal reward. Root cause of
        # "orbit / flies straight away" was diagnosed as CATASTROPHIC
        # FORGETTING (a well-documented phenomenon in continual/curriculum
        # DQN training -- e.g. Ribeiro, Melo & Dias 2019 "Multi-task
        # Learning and Catastrophic Forgetting in Continual RL"; Kirkpatrick
        # et al. 2017 EWC), NOT a reward design problem. Verified via
        # inspect_q_values.py: the Stage-0 checkpoint correctly ranked
        # "turn to target" highest under extreme heading misalignment
        # (-165°), but after Stage 1 (obstacle) training -- under EITHER
        # reward version tried -- this same judgment collapsed (Q-values
        # ranked "forward" highest regardless of heading, even at -165°).
        # rew_obstacle/rew_progress additions did not cause or fix this;
        # they're left computed-but-unused above (harmless dead code) so
        # this isn't confused with a NameError-causing removal like the
        # earlier made_progress incident. Real fix is training-methodology
        # (rehearsal mixing in _reset_idx, see below), not reward tuning. ──
        return rew_distance + rew_reached + rew_crashed + rew_action_rate + rew_yaw_aligned

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self._robot.data.root_pos_w - self.scene.env_origins
        ground_crash = pos[:, 2] < self.cfg.min_height
        net_forces = self._contact_sensor.data.net_forces_w_history
        force_magnitude = torch.norm(net_forces, dim=-1)
        any_contact = torch.any(force_magnitude > 1.0, dim=(1, 2))
        crashed = ground_crash | any_contact
        # ── DQN diagnostic: cache altitude at the moment of termination
        # decision, to distinguish "hovering near-ground and physically
        # touched something" (any_contact, altitude close to 0 but above
        # min_height=-0.1) from other contact causes (e.g. self-collision,
        # altitude far from 0). ──
        self._last_pos_z = pos[:, 2].clone()
        timed_out = self.episode_length_buf >= self.max_episode_length


        dist = torch.norm(self._target_pos - pos, dim=-1)
        reached = dist < 0.75  # matches _get_rewards() threshold
        out_of_bounds = dist > self.cfg.max_dist_from_start


        terminal_mask = ground_crash | any_contact | reached | timed_out | out_of_bounds
        self.extras["log"] = {
            "termination/reached_target": reached[terminal_mask].float(),
            "termination/ground_crash": ground_crash[terminal_mask].float(),
            "termination/obstacle_collision": any_contact[terminal_mask].float(),
            "termination/timed_out": timed_out[terminal_mask].float(),
            "termination/out_of_bounds": out_of_bounds[terminal_mask].float(),
        }
        return crashed | reached | out_of_bounds, timed_out

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # BUGFIX: Multirotor._initialize_impl() calls _process_thruster_cfg()
        # (constructs Thruster, initializes its internal curr_thrust from
        # default_thruster_rps at THAT moment) BEFORE _process_cfg() (which
        # actually writes our configured init_state.rps into
        # default_thruster_rps). This ordering bug means changing
        # init_state.rps has NO effect on the Thruster's actual starting
        # thrust -- confirmed via debug logging showing unchanged applied
        # thrust before/after changing rps 200->470. Fix applied here (in
        # _reset_idx, not _setup_scene, since self._robot.actuators doesn't
        # exist yet at _setup_scene time -- it's only created later during
        # Multirotor._initialize_impl(), which runs after scene setup
        # completes): directly overwrite curr_thrust for the envs being
        # reset with the true hover-equivalent thrust, so every reset starts
        # at (approximately) hover thrust rather than a near-zero value that
        # caused genuine free-fall for several physics steps regardless of
        # controller design.
        if hasattr(self._robot, "actuators") and "thrusters" in self._robot.actuators:
            _thruster = self._robot.actuators["thrusters"]
            _hover_thrust = getattr(self, "_drone_mass", 1.24) * getattr(self, "_gravity", 9.81) / 4.0
            _thruster.curr_thrust[env_ids] = _hover_thrust


        # Reset robot with small random perturbation
        root_state = self._robot.data.default_root_state[env_ids].clone()
        # Add env_origins so position is in world frame
        root_state[:, :3] += self.scene.env_origins[env_ids]
        # Only randomize X/Y; keep Z at init height (1.0m above ground)
        root_state[:, 0:2] += torch.zeros(n, 2, device=self.device).uniform_(-1.0, 1.0)
        root_state[:, 7:10] += torch.zeros(n, 3, device=self.device).uniform_(-0.2, 0.2)
        self._robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        # ── Target/obstacle placement (REDESIGNED) ────────────────────────────
        # BUGFIX #4: previously target_pos was sampled FIRST and independently
        # (uniform in +/-5m box), and obstacles were then placed at FIXED
        # ABSOLUTE distances along the resulting start->target direction
        # (2.5m, 2.5+spacing, ...). Since target distance in a +/-5m box is
        # frequently LESS than 2.5m, every obstacle often ended up farther
        # from the start than the target itself -- meaning the drone could
        # reach the target without ever flying near any obstacle. This let
        # the policy achieve a high reached_target rate (~60%) with only
        # "some" avoidance behavior, without being forced to actually
        # navigate through the obstacle field on most episodes.
        #
        # FIX: reverse the generation order. Pick a random flight DIRECTION
        # first, place obstacles along it at fixed spacing (unchanged from
        # before), then force the target to spawn PAST the last active
        # obstacle (with a safety margin), so every episode requires flying
        # through the obstacle corridor to reach the target.
        obstacle_names = ["ObstacleCyl0", "ObstacleCyl1", "ObstacleCyl2", "ObstacleBox0", "ObstacleBox1"]
        z_heights = [2.0, 2.0, 2.5, 1.5, 2.0]
        n_obs = len(obstacle_names)
        n_envs = n

        start_xy = root_state[:, 0:2] - self.scene.env_origins[env_ids, 0:2]

        # 1. Random flight direction (unit vector), independent of target.
        heading_angle = torch.zeros(n_envs, device=self.device).uniform_(0.0, 2 * math.pi)
        path_dir = torch.stack([torch.cos(heading_angle), torch.sin(heading_angle)], dim=-1)  # (n, 2)
        perp_dir = torch.stack([-path_dir[:, 1], path_dir[:, 0]], dim=-1)

        # 2. Place obstacles along this direction (same spacing logic as before).
        along_dist_base = 2.5 + torch.arange(n_obs, device=self.device).float() * self.cfg.obstacle_min_spacing
        # Widened from +/-0.3m: at this small scale, obstacles' along-path
        # order always matched their index order (0 nearest, 4 farthest),
        # producing a visually "straight-line queue" look with only lateral
        # variation -- not a genuinely scattered field. A jitter closer to
        # half the spacing lets obstacles occasionally overtake each other
        # in along-path order (e.g. obstacle 3 ends up nearer than obstacle
        # 2), so the field looks properly scattered rather than a lineup.
        # NOTE: this breaks the assumption that index g and g+1 are
        # spatially adjacent -- the gap-pairing logic below now explicitly
        # sorts by actual along_dist before pairing neighbors.
        jitter_range = self.cfg.obstacle_min_spacing * 0.4
        jitter = torch.zeros(n_envs, n_obs, device=self.device).uniform_(-jitter_range, jitter_range)
        along_dist = along_dist_base.unsqueeze(0) + jitter   # (n, n_obs)
        # BUGFIX: a fixed alternating zigzag (side_sign +1/-1, magnitude
        # 0.6-1.0m) was tried first, but this still has a fixed, predictable
        # maximum lateral extent (~1.0m) -- meaning a single straight
        # trajectory offset ~1.5-2m from centerline can clear every
        # obstacle's safety radius at once, regardless of alternation. Any
        # FIXED bound on lateral offset admits this "one wide sweep skips
        # everything" shortcut, since the required detour distance is
        # always knowable in advance.
        #
        # FIX: fully independent random offset per obstacle, both magnitude
        # AND side (no forced alternation). With an unbounded-feeling wider
        # range and no fixed pattern, some obstacles will occasionally land
        # very close to the path centerline (near-zero offset) purely by
        # chance each episode, and there's no reliable "safe swing distance"
        # that clears all 5 simultaneously across episodes -- forcing the
        # policy to perceive and react to each obstacle's actual position
        # rather than learning one fixed detour trajectory.
        # Widened from [-1.5, 1.5]: the obstacle field's along-path span
        # (~12m with 5 obstacles at ~3m spacing) was far larger than its
        # lateral span (3m total), so the field always looked like a long,
        # narrow corridor no matter how the obstacles were shuffled within
        # it. Widening lateral spread to match brings the field closer to a
        # roughly square footprint, and as a side benefit increases the
        # detour distance required to "swing wide around everything",
        # making that shortcut less attractive.
        lateral = torch.zeros(n_envs, n_obs, device=self.device).uniform_(-3.0, 3.0)
        along_pos = start_xy.unsqueeze(1) + along_dist.unsqueeze(-1) * path_dir.unsqueeze(1)
        offset_pos = lateral.unsqueeze(-1) * perp_dir.unsqueeze(1)
        obs_xy = along_pos + offset_pos                                     # (n, n_obs, 2)
        obs_x = obs_xy[..., 0]
        obs_y = obs_xy[..., 1]

        # 3. BUGFIX #5: previously the target was placed anywhere past the
        # last active obstacle (any distance/lateral in a +/-1m band). This
        # let the policy learn a single "swing wide around one side" path
        # that reaches ANY such target without ever weaving between
        # obstacles -- confirmed via play.py even after obstacles were made
        # fully randomly distributed (no fixed pattern to exploit). The
        # fundamental issue: without a boundary, a wide-enough detour always
        # exists that clears every obstacle's safety radius at once.
        #
        # FIX: force the target to sit IN A GAP between two adjacent active
        # obstacles (by along-path order), at the lateral midpoint between
        # them. This makes the target's lateral position depend on that
        # episode's specific (unpredictable) obstacle layout, so "always
        # swing wide to one side" no longer reaches the target -- the drone
        # must actually thread through the corridor at the right lateral
        # offset. Falls back to the old "past the last obstacle" placement
        # only on the rare episode where no adjacent pair has enough lateral
        # clearance for a safe passage.
        n_active = min(self.cfg.num_active_obstacles, n_obs)

        if n_active >= 2:
            # Sort ACTIVE obstacles by their true along-path distance before
            # pairing neighbors. With the widened jitter above, index order
            # (0,1,2,...) no longer guarantees along-path order -- obstacle
            # 3 might now be physically nearer than obstacle 2. Pairing by
            # raw index would then compute a "gap" between two obstacles
            # that aren't actually spatially adjacent, which is geometrically
            # wrong (the true nearest neighbor might be a different index).
            active_along = along_dist[:, :n_active]      # (n, n_active)
            active_lateral = lateral[:, :n_active]        # (n, n_active)
            active_radius = self._obstacle_radius[:n_active].unsqueeze(0).expand(n_envs, -1)  # (n, n_active)

            sort_idx = torch.argsort(active_along, dim=-1)             # (n, n_active)
            sorted_along = torch.gather(active_along, 1, sort_idx)
            sorted_lateral = torch.gather(active_lateral, 1, sort_idx)
            sorted_radius = torch.gather(active_radius, 1, sort_idx)

            n_gaps = n_active - 1
            gap_valid = torch.zeros(n_envs, n_gaps, dtype=torch.bool, device=self.device)
            gap_along_lo = torch.zeros(n_envs, n_gaps, device=self.device)
            gap_along_hi = torch.zeros(n_envs, n_gaps, device=self.device)
            gap_lat_lo = torch.zeros(n_envs, n_gaps, device=self.device)
            gap_lat_hi = torch.zeros(n_envs, n_gaps, device=self.device)
            for g in range(n_gaps):
                width = torch.abs(sorted_lateral[:, g + 1] - sorted_lateral[:, g])
                required = sorted_radius[:, g] + sorted_radius[:, g + 1] + 0.3
                gap_valid[:, g] = width > required
                gap_along_lo[:, g] = sorted_along[:, g]
                gap_along_hi[:, g] = sorted_along[:, g + 1]
                gap_lat_lo[:, g] = sorted_lateral[:, g]
                gap_lat_hi[:, g] = sorted_lateral[:, g + 1]

            has_valid_gap = gap_valid.any(dim=-1)
            rand_score = torch.rand(n_envs, n_gaps, device=self.device)
            rand_score = torch.where(gap_valid, rand_score, torch.full_like(rand_score, -1.0))
            chosen = rand_score.argmax(dim=-1)   # (n,)
            idx = torch.arange(n_envs, device=self.device)

            target_along_gap = (gap_along_lo[idx, chosen] + gap_along_hi[idx, chosen]) / 2.0
            target_lateral_gap = (gap_lat_lo[idx, chosen] + gap_lat_hi[idx, chosen]) / 2.0
        else:
            has_valid_gap = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
            target_along_gap = torch.zeros(n_envs, device=self.device)
            target_lateral_gap = torch.zeros(n_envs, device=self.device)

        # Fallback: old "past the last obstacle" placement, used when no
        # adjacent pair has a safely navigable gap this episode.
        if n_active > 0:
            last_active_dist = along_dist[:, n_active - 1]
            margin = self._obstacle_radius[n_active - 1] + 1.0
            min_target_dist = last_active_dist + margin
        else:
            # v2: widened from 1.0m to give the drone more room to
            # demonstrate stable straight-line flight and turning over a
            # longer distance, and to leave more along-path depth for
            # obstacles to be reintroduced later without immediately
            # crowding the target.
            min_target_dist = torch.full((n_envs,), 5.0, device=self.device)
        max_target_dist = min_target_dist + 10.0  # v2: widened from +3.0 -> target range ~5-15m
        target_along_fallback = torch.zeros(n_envs, device=self.device).uniform_(0.0, 1.0) \
            * (max_target_dist - min_target_dist) + min_target_dist
        # For the single-obstacle curriculum stage, couple target's lateral
        # offset to the obstacle's lateral offset (rather than sampling it
        # fully independently) so the obstacle reliably sits near the
        # start->target path -- otherwise a large fraction of episodes have
        # the obstacle irrelevant to the direct path, diluting the avoidance
        # signal. The offset itself is never near-zero (uniform(0.4, 1.2)
        # magnitude, random sign) so the two sides are never exactly
        # symmetric -- avoids the degenerate "stop in front of obstacle"
        # solution that a single-mode Gaussian policy can collapse to when
        # left/right are equally optimal.
        if n_active == 1:
            obs_lateral_single = lateral[:, 0]
            offset_mag = torch.zeros(n_envs, device=self.device).uniform_(0.4, 1.2)
            offset_sign = torch.where(
                torch.rand(n_envs, device=self.device) < 0.5,
                torch.full((n_envs,), -1.0, device=self.device),
                torch.full((n_envs,), 1.0, device=self.device),
            )
            target_lateral_fallback = obs_lateral_single + offset_sign * offset_mag
        else:
            target_lateral_fallback = torch.zeros(n_envs, device=self.device).uniform_(-1.0, 1.0)

        target_along = torch.where(has_valid_gap, target_along_gap, target_along_fallback)
        target_lateral = torch.where(has_valid_gap, target_lateral_gap, target_lateral_fallback)
        target_xy = start_xy + target_along.unsqueeze(-1) * path_dir + target_lateral.unsqueeze(-1) * perp_dir

        self._target_pos[env_ids, 0] = target_xy[:, 0]
        self._target_pos[env_ids, 1] = target_xy[:, 1]
        self._target_pos[env_ids, 2] = torch.zeros(n, device=self.device).uniform_(*self.cfg.target_pos_range_z)

        # Initialize prev_distance to the distance right after reset, using
        # the NEW position (root_state, not yet written but authoritative)
        # and NEW target. This ensures the first progress-reward computation
        # after a reset measures real movement, not a "free" jump caused by
        # teleporting to a new spawn point / new target.
        new_pos = root_state[:, :3]
        self._prev_distance[env_ids] = torch.norm(self._target_pos[env_ids] - new_pos, dim=-1)
        self._min_dist_so_far[env_ids] = self._prev_distance[env_ids].clone()
        self._prev_height_diff[env_ids] = torch.abs(self._target_pos[env_ids, 2] - new_pos[:, 2])

        # Curriculum learning: obstacles beyond cfg.num_active_obstacles are
        # relocated far away (+1000m) so they don't appear in the flight
        # path, aren't detected by the depth camera, and never trigger the
        # collision check. This lets training difficulty be raised
        # gradually (more active obstacles / tighter spacing) across
        # separate resumed runs without changing the obstacle COUNT/asset
        # setup, which would require re-registering scene objects.
        if self.cfg.num_active_obstacles < n_obs:
            inactive_idx = torch.arange(self.cfg.num_active_obstacles, n_obs, device=self.device)
            obs_x[:, inactive_idx] = 1000.0 + inactive_idx.float().unsqueeze(0)
            obs_y[:, inactive_idx] = 1000.0
            # BUGFIX: obs_xy (used below by the BUGFIX #3 target-push-away
            # loop) was never updated to match the relocated obs_x/obs_y,
            # so that loop was reading stale near-field obstacle positions
            # for inactive obstacles instead of their relocated (1000m+)
            # positions. Sync it here.
            obs_xy[:, inactive_idx, 0] = obs_x[:, inactive_idx]
            obs_xy[:, inactive_idx, 1] = obs_y[:, inactive_idx]

        # ── Rehearsal mixing (catastrophic-forgetting fix): even for the
        # normally-"active" obstacle (index 0), randomly relocate it far
        # away for a FRACTION of envs on THIS reset, independent of the
        # global cfg.num_active_obstacles setting. This keeps the replay
        # buffer continuously stocked with a mix of pure-navigation (Stage-
        # 0-like) and obstacle-avoidance (Stage-1) transitions throughout
        # training, rather than the buffer being entirely overwritten by
        # only-new-scenario experience -- the established fix for
        # catastrophic forgetting in continual/curriculum DQN training
        # (rehearsal / experience replay mixing; see e.g. Ribeiro, Melo &
        # Dias 2019). Without this, gradient updates stop reinforcing
        # previously-learned behavior (e.g. correct heading-based turning
        # judgment) for states the current policy visits less often as it
        # improves, and that capability silently degrades even though
        # nothing about it was directly targeted by the new training. ──
        if self.cfg.num_active_obstacles >= 1:
            REHEARSAL_PROB = 0.5
            rehearsal_mask = torch.rand(n_envs, device=self.device) < REHEARSAL_PROB
            far_val = torch.full((n_envs,), 2000.0, device=self.device)
            obs_x[:, 0] = torch.where(rehearsal_mask, far_val, obs_x[:, 0])
            obs_y[:, 0] = torch.where(rehearsal_mask, far_val, obs_y[:, 0])
            obs_xy[:, 0, 0] = obs_x[:, 0]
            obs_xy[:, 0, 1] = obs_y[:, 0]

        # BUGFIX #3: target position was sampled completely independently of
        # obstacle placement, so a target could end up embedded inside (or
        # very close to) an obstacle — physically unreachable (the drone's
        # 0.5m arrival sphere would overlap the obstacle's own collision
        # geometry, so it might never register as "reached" without
        # crashing first). Push the target away from any obstacle it's too
        # close to, along the obstacle-center -> target direction, out to
        # just past that obstacle's safety radius (which already includes
        # the drone's own collision radius) plus the 0.5m arrival margin.
        # Resolved sequentially per obstacle — an approximation, not a
        # perfect multi-obstacle solver, but sufficient since obstacles are
        # spread out along the path (see earlier fix) and rarely overlap.
        arrival_radius = 0.5
        target_xy_resolved = target_xy.clone()
        for i in range(n_obs):
            obs_center = obs_xy[:, i, :]                                    # (n, 2)
            safety_r = self._obstacle_radius[i] + arrival_radius
            diff = target_xy_resolved - obs_center                          # (n, 2)
            dist_to_obs = torch.norm(diff, dim=-1)                          # (n,)
            needs_push = dist_to_obs < safety_r
            push_dir = diff / dist_to_obs.clamp(min=1e-6).unsqueeze(-1)
            pushed_target = obs_center + push_dir * safety_r
            target_xy_resolved = torch.where(
                needs_push.unsqueeze(-1), pushed_target, target_xy_resolved
            )
        # Write the corrected target back (Z is untouched — only XY can
        # conflict with obstacles, which are vertical columns/boxes).
        self._target_pos[env_ids, 0] = target_xy_resolved[:, 0]
        self._target_pos[env_ids, 1] = target_xy_resolved[:, 1]

        # ── Hard-case injection: force a fraction of resets to start
        # already close to and heading directly at obstacle 0, at high
        # speed. Motivation: offline analysis (dwell-tradeoff on real
        # play.py data) showed that "high-speed + close to obstacle" states
        # occur extremely rarely in normal rollouts (~0.16% of steps for
        # 0.2-0.5m, ~0.007% for <0.2m), even though the reward's net value
        # in exactly these states is strongly negative (-0.46 to -0.58).
        # This is a severe sample-imbalance problem, not a reward-magnitude
        # problem -- the policy rarely gets gradient signal from the states
        # it most needs to learn from. Injecting these states directly into
        # a fraction of episode starts increases their frequency without
        # touching reward weights or entropy_coef (kept as an independent
        # variable from the reward-tuning experiments). Only applied when
        # at least 1 obstacle is active; uses obstacle index 0 specifically
        # (not a random active obstacle) to keep this first version simple.
        # ABLATION (SimpleObs only): hard_case_frac set to 0.0, disabling
        # hard-case injection for this run. Rationale: hard-case injection was
        # motivated by the CNN pipeline's severe sample imbalance (high-speed
        # near-obstacle states occurring in only 0.16%/0.007% of steps,
        # diagnosed via dwell-tradeoff split-by-speed analysis). Under the
        # simplified ground-truth-vector observation, the distance-to-action
        # mapping is far simpler to learn per-sample, so it's untested whether
        # the same sample-imbalance problem applies here -- this run tests
        # whether normal rollout exposure alone (no forced hard starts) is
        # sufficient for the simplified architecture to learn deceleration.
        # Code below is left intact (not deleted) so hard-case can be
        # re-enabled by reverting this line if needed. Everything else
        # (2 obstacles, entropy_coef=0.001, reward weights, from-scratch
        # training) held identical to the hard-case-enabled SimpleObs run
        # for a clean single-variable comparison.
        hard_case_frac = 0.0
        # RECALIBRATED: original range (gap 0.15-0.7m, speed 1.5-2.5m/s)
        # produced a 100% collision rate on hard-case episodes (0/30 reached
        # in evaluation), while normal-start episodes were only mildly
        # affected (89.5% vs 91.84% baseline reached). Kinematic estimate
        # using the controller's actual params (vel_ctrl_kp=4.0,
        # max_accel=6.0) plus an assumed ~0.12s reaction delay for attitude
        # tilt gives a minimum stopping distance of ~0.82m at 2.5 m/s and
        # ~0.37m at 1.5 m/s -- the old gap range (up to 0.7m) was physically
        # unstoppable at the old speed range, meaning these episodes taught
        # the policy nothing (no achievable action existed) and likely
        # corrupted the value function (every action from these states led
        # to a bad outcome regardless of quality). New range leaves margin
        # above the worst-case stopping distance so the injected states are
        # survivable and can actually teach deceleration.
        hard_case_gap_range = (0.8, 1.5)    # surface-to-drone distance, meters
        hard_case_speed_range = (1.0, 2.0)  # m/s
        if n_active >= 1:
            hard_mask = torch.rand(n_envs, device=self.device) < hard_case_frac
            if hard_mask.any():
                gap = torch.zeros(n_envs, device=self.device).uniform_(*hard_case_gap_range)
                dist_from_center = self._obstacle_radius[0] + gap
                hc_along = along_dist[:, 0] - dist_from_center
                hc_lateral = lateral[:, 0]
                hc_xy = start_xy + hc_along.unsqueeze(-1) * path_dir + hc_lateral.unsqueeze(-1) * perp_dir
                speed = torch.zeros(n_envs, device=self.device).uniform_(*hard_case_speed_range)
                hc_vel = speed.unsqueeze(-1) * path_dir  # world-frame xy velocity, aimed at the obstacle
                half_yaw = heading_angle / 2.0
                mask3 = hard_mask.unsqueeze(-1)
                root_state[:, 0:1] = torch.where(hard_mask.unsqueeze(-1), hc_xy[:, 0:1] + self.scene.env_origins[env_ids, 0:1], root_state[:, 0:1])
                root_state[:, 1:2] = torch.where(hard_mask.unsqueeze(-1), hc_xy[:, 1:2] + self.scene.env_origins[env_ids, 1:2], root_state[:, 1:2])
                root_state[:, 3:4] = torch.where(hard_mask.unsqueeze(-1), torch.cos(half_yaw).unsqueeze(-1), root_state[:, 3:4])
                root_state[:, 4:5] = torch.where(hard_mask.unsqueeze(-1), torch.zeros_like(half_yaw).unsqueeze(-1), root_state[:, 4:5])
                root_state[:, 5:6] = torch.where(hard_mask.unsqueeze(-1), torch.zeros_like(half_yaw).unsqueeze(-1), root_state[:, 5:6])
                root_state[:, 6:7] = torch.where(hard_mask.unsqueeze(-1), torch.sin(half_yaw).unsqueeze(-1), root_state[:, 6:7])
                root_state[:, 7:8] = torch.where(hard_mask.unsqueeze(-1), hc_vel[:, 0:1], root_state[:, 7:8])
                root_state[:, 8:9] = torch.where(hard_mask.unsqueeze(-1), hc_vel[:, 1:2], root_state[:, 8:9])
                self._robot.write_root_state_to_sim(root_state, env_ids=env_ids)
                # Recompute distance-tracking fields for the repositioned
                # subset -- they were already set above using the OLD
                # (default far-away) spawn position, before this override.
                new_pos_hc = root_state[:, :3]
                recomputed_dist = torch.norm(self._target_pos[env_ids] - new_pos_hc, dim=-1)
                recomputed_height = torch.abs(self._target_pos[env_ids, 2] - new_pos_hc[:, 2])
                self._prev_distance[env_ids] = torch.where(hard_mask, recomputed_dist, self._prev_distance[env_ids])
                self._min_dist_so_far[env_ids] = self._prev_distance[env_ids].clone()
                self._prev_height_diff[env_ids] = torch.where(hard_mask, recomputed_height, self._prev_height_diff[env_ids])

        # ── Simplified-observation experiment: persist each obstacle's
        # local-frame position (same frame as obs_x/obs_y, no env_origins
        # offset) for the envs being reset. Read back in _get_observations()
        # to compute the nearest-visible-obstacle vector without needing the
        # depth camera. z_heights is a plain python list (same value for
        # every env), so it's broadcast via torch.tensor(...).expand(). ──
        z_heights_t = torch.tensor(z_heights, device=self.device).unsqueeze(0).expand(n_envs, -1)
        self._obstacle_world_pos[env_ids, :, 0] = obs_x
        self._obstacle_world_pos[env_ids, :, 1] = obs_y
        self._obstacle_world_pos[env_ids, :, 2] = z_heights_t

        for i, (obs_name, z) in enumerate(zip(obstacle_names, z_heights)):
            obs_obj = self.scene.rigid_objects[obs_name]
            root_state_obs = torch.zeros(n_envs, 13, device=self.device)
            root_state_obs[:, 0] = obs_x[:, i] + self.scene.env_origins[env_ids, 0]
            root_state_obs[:, 1] = obs_y[:, i] + self.scene.env_origins[env_ids, 1]
            root_state_obs[:, 2] = z
            root_state_obs[:, 3] = 1.0
            # BUGFIX: must pass env_ids= so only the environments actually
            # being reset have their obstacles moved — omitting this
            # defaulted to writing all environments, teleporting obstacles
            # under drones mid-flight in envs whose episodes were ongoing.
            obs_obj.write_root_state_to_sim(root_state_obs, env_ids=env_ids)

        self._obstacle_local_pos = torch.stack([
            obs_x.mean(dim=0),
            obs_y.mean(dim=0),
            torch.tensor(z_heights, device=self.device)
        ], dim=1)

        # Reset buffers
        self._prev_actions[env_ids] = 0.0
        self._prev_yaw_rate_cmd[env_ids] = 0.0  # method 1: slew-rate limit state
        self._actions[env_ids] = 0.0

        # Initialize prev_distance after target is set
        new_pos = self._robot.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self._prev_distance[env_ids] = torch.norm(self._target_pos[env_ids] - new_pos, dim=-1)
        self._d0[env_ids] = self._prev_distance[env_ids].clone()  # for (d0-d)/d0 progress reward
        self._min_dist_so_far[env_ids] = self._prev_distance[env_ids].clone()
        self._prev_height_diff[env_ids] = torch.abs(self._target_pos[env_ids, 2] - new_pos[:, 2])
