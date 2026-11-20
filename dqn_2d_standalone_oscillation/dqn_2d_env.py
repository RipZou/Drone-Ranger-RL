"""
Pure-2D, Isaac-Lab-free training environment -- isolates the DQN/reward/
network design from any Isaac Lab simulation detail (physics stepping,
attitude controller dynamics, camera rendering) to test whether the
oscillation/low-margin problem seen in q_sweep.py is caused by the DQN
side of things or by something in the Isaac Lab simulation layer.

Kinematics only: no thrust, no attitude, no gravity. Position and heading
are integrated directly from the same discrete action -> (vx, yaw_rate)
mapping used by the real environment. The control step is 0.1 s here, five
times Isaac Lab's 0.02 s (sim.dt=1/100 * decimation=2).

With distance_mode="ground_truth" (the default) two cameras turned half their
73 deg to either side cover 146 deg ahead with no gap, each split into
CAMERA_BINS equal bins:
    state = [fr_0..fr_4, fl_0..fl_4, dx, dy]
each an exact distance to the nearest obstacle surface in that bin
(obstacle_geometry.py), ordered from the right edge of the view to its left
edge, so a gap between two obstacles shows as a far bin between near ones.
Nothing more than 73 deg off the heading is seen. With distance_mode="camera"
the state keeps the real environment's layout, state = [d_front, d_left, d_right, dx, dy], with what
the Isaac env's three depth cameras would report (camera_emulation.py). With
num_obstacles=0 (Stage 0) both modes read 10.0 (max sensing range) in every
reading, what the real environment's cameras report when nothing is in view.

distance_mode="coords" is a diagnostic upper bound with no Isaac
counterpart. It replaces the three readings with k_nearest slots
(sx, sy, visible), the nearest surface point of each of the k_nearest
closest obstacles in the body frame, nearest first:
    state = [s1_x, s1_y, v1, ..., sK_x, sK_y, vK, dx, dy]
Its norm is the exact distance, with no sector and so no blind spot. An
obstacle beyond MAX_SENSE_RANGE or parked leaves its slot all zeros, so the
input size depends on k_nearest only and stays fixed across num_obstacles.

num_obstacles=1..8 (Stage 1) uses the layout of the collaborator's commit
1643e0e rather than the Isaac env's. The target is drawn as in Stage 0 and
redrawn until it lies MIN_START_TARGET_DIST from the start, then circles of
radius OBSTACLE_RADIUS are rejection-sampled in the same box, clear of the start
and the target, with obstacle 0 forced onto the start-target segment so it has
to be avoided. Touching an obstacle ends the episode as a crash.

Interface intentionally mirrors gymnasium's reset()/step() so train_dqn.py
needs minimal changes to switch environments.
"""
import math

import torch

from dqn_config import DQNConfig, SPEED_FRACTIONS
from camera_emulation import FAR_CLIP, TAN_HALF_FOV, camera_readings
from obstacle_geometry import _nearest_surface, obstacle_sector_distances, obstacle_surface_distance

DT = 0.1  # seconds per control step, 10 Hz instead of Isaac Lab's 50 Hz
MAX_SENSE_RANGE = FAR_CLIP  # both distance modes cap at the cameras' far clip
# Three times the real env's 20 s episode_length_s. At 20 s about seven in eight evaluation timeouts
# reached the target when given 60 s.
EPISODE_LENGTH_S = 60.0
TARGET_POS_RANGE = 5.0   # start/target sampled uniformly in [-5, 5] on each axis
REACHED_THRESHOLD = 0.75
OUT_OF_BOUNDS_DIST = 40.0  # matches the real env's widened boundary

# Obstacle shape and crash rule of commit 1643e0e, circles and no boxes, with room for eight.
OBSTACLE_RADIUS = 0.5
CYLINDER_RADII = (OBSTACLE_RADIUS,) * 8
BOX_HALF_EXTENTS = ()
NUM_CYLINDERS = len(CYLINDER_RADII)
MAX_OBSTACLES = NUM_CYLINDERS + len(BOX_HALF_EXTENTS)
DRONE_RADIUS = 0.2  # a crash once the drone's center is this close to a surface
OBSTACLE_CLEARANCE = 0.5  # extra spawn gap around the start and the target
START_KEEPOUT = OBSTACLE_RADIUS + DRONE_RADIUS + OBSTACLE_CLEARANCE  # min center distance from the start
TARGET_KEEPOUT = OBSTACLE_RADIUS + REACHED_THRESHOLD + OBSTACLE_CLEARANCE  # min center distance from the target
PLACEMENT_ATTEMPTS = 30  # a slot still invalid after this many draws stays parked
# With obstacles, the target is redrawn until it lies this far from the start. Below about 2.77 m no
# spot on the middle half of the segment clears both keepouts, so obstacle 0 could not block the path.
MIN_START_TARGET_DIST = 3.0
SENSOR_HALF_FOV = math.atan(TAN_HALF_FOV)  # half of the cameras' 73 deg HFOV
CAMERA_BINS = 5
# Front-right then front-left, so the bins run right to left across both cameras. An offset of exactly
# SENSOR_HALF_FOV puts their shared edge on the heading.
CAMERA_OFFSETS = (-SENSOR_HALF_FOV, SENSOR_HALF_FOV)
CAMERA_NAMES = ("fr", "fl")
# ground_truth sectors as (offset, half width), CAMERA_BINS per camera from its right edge to its left edge.
# The bins of one camera tile exactly its 73 deg, so their minimum is that camera's single-sector reading.
GROUND_TRUTH_SECTORS = tuple(
    (cam + SENSOR_HALF_FOV * (2 * b + 1 - CAMERA_BINS) / CAMERA_BINS, SENSOR_HALF_FOV / CAMERA_BINS)
    for cam in CAMERA_OFFSETS for b in range(CAMERA_BINS))
# One camera on the heading (camera_count 1): CAMERA_BINS sectors tiling its 73 deg, right to left.
SINGLE_CAMERA_SECTORS = tuple(
    (SENSOR_HALF_FOV * (2 * b + 1 - CAMERA_BINS) / CAMERA_BINS, SENSOR_HALF_FOV / CAMERA_BINS)
    for b in range(CAMERA_BINS))
SECTORS_BY_CAMERA_COUNT = {1: SINGLE_CAMERA_SECTORS, 2: GROUND_TRUTH_SECTORS}
DISTANCE_MODES = ("ground_truth", "camera", "coords")
DEFAULT_K_NEAREST = 3
PREV_ACTION_DIM = 3 + len(SPEED_FRACTIONS)
SIMILARITY_WINDOW = 20  # steps back to the pose the similarity penalty compares against
SIMILARITY_OBS_DIM = 3  # body-frame dx, dy and heading change since that pose

# Episode outcomes, highest priority first, for a step that meets several conditions.
OUTCOMES = ("reached", "contact_crash", "out_of_bounds", "timed_out")


class DroneNav2DEnv:
    def __init__(self, num_envs: int, cfg: DQNConfig, device: str, num_obstacles: int = 0,
                 distance_mode: str = "ground_truth", k_nearest: int = DEFAULT_K_NEAREST):
        if not 0 <= num_obstacles <= MAX_OBSTACLES:
            raise ValueError(f"num_obstacles must be in [0, {MAX_OBSTACLES}], got {num_obstacles}")
        if distance_mode not in DISTANCE_MODES:
            raise ValueError(f"distance_mode must be one of {DISTANCE_MODES}, got {distance_mode!r}")
        if not 1 <= k_nearest <= MAX_OBSTACLES:
            raise ValueError(f"k_nearest must be in [1, {MAX_OBSTACLES}], got {k_nearest}")
        if cfg.progress_mode not in ("velocity", "distance"):
            raise ValueError(f"progress_mode must be velocity or distance, got {cfg.progress_mode!r}")
        if cfg.proximity_shape not in ("linear", "inverse_square"):
            raise ValueError(f"proximity_shape must be linear or inverse_square, got {cfg.proximity_shape!r}")
        if cfg.proximity_clearance not in ("true", "sensor"):
            raise ValueError(f"proximity_clearance must be true or sensor, got {cfg.proximity_clearance!r}")
        if cfg.proximity_clearance == "sensor" and distance_mode == "coords":
            raise ValueError("proximity_clearance sensor needs distance readings, which coords mode does not give")
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = device
        self.num_obstacles = num_obstacles
        self.distance_mode = distance_mode
        self.k_nearest = k_nearest
        # Networks are sized from this, not from cfg.state_dim, because coords mode changes it.
        if cfg.camera_count not in SECTORS_BY_CAMERA_COUNT:
            raise ValueError(f"camera_count must be 1 or 2, got {cfg.camera_count}")
        if cfg.camera_count != 2 and distance_mode != "ground_truth":
            raise ValueError("camera_count 1 is only implemented for distance_mode ground_truth")
        self.sectors = SECTORS_BY_CAMERA_COUNT[cfg.camera_count]
        self.num_readings = {"coords": 3 * k_nearest, "ground_truth": len(self.sectors), "camera": 3}[distance_mode]
        self.state_dim = self.num_readings + PREV_ACTION_DIM + SIMILARITY_OBS_DIM + 2
        # One-hot of the last turn (none, left, right) followed by a one-hot of the last speed,
        # all zeros on an episode's first step.
        self.prev_action = torch.zeros(num_envs, PREV_ACTION_DIM, device=device)
        self.max_steps = int(EPISODE_LENGTH_S / DT)

        self.pos = torch.zeros(num_envs, 2, device=device)      # (N, 2) x,y
        self.yaw = torch.zeros(num_envs, device=device)          # (N,) radians
        self.target = torch.zeros(num_envs, 2, device=device)    # (N, 2)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Ring of the last SIMILARITY_WINDOW poses, slot step_count % SIMILARITY_WINDOW holding the pose SIMILARITY_WINDOW steps back.
        self.pose_hist_pos = torch.zeros(num_envs, SIMILARITY_WINDOW, 2, device=device)
        self.pose_hist_yaw = torch.zeros(num_envs, SIMILARITY_WINDOW, device=device)
        self.pose_similarity = torch.zeros(num_envs, device=device)
        # What the state shows of the pose SIMILARITY_WINDOW steps back, or of the start pose before then,
        # so the network can see the similarity penalty coming.
        self.pose_delta = torch.zeros(num_envs, SIMILARITY_OBS_DIM, device=device)

        self._circle_r = torch.tensor(CYLINDER_RADII, device=device)
        self._box_half = torch.tensor(BOX_HALF_EXTENTS, device=device).reshape(-1, 2)
        # Inactive obstacles are parked where the Isaac env parks them, far
        # beyond sensing range, so they never register in a reading.
        self._parked_xy = torch.stack([
            1000.0 + torch.arange(MAX_OBSTACLES, device=device, dtype=torch.float32),
            torch.full((MAX_OBSTACLES,), 1000.0, device=device),
        ], dim=-1)
        self.obstacle_xy = self._parked_xy.expand(num_envs, -1, -1).clone()  # (N, MAX_OBSTACLES, 2)
        # False for obstacles parked this episode, beyond num_obstacles or unplaceable.
        self.obstacle_active = torch.zeros(num_envs, MAX_OBSTACLES, dtype=torch.bool, device=device)

    def _sample_positions(self, env_ids):
        n = len(env_ids)
        # Matches the real environment exactly: the drone starts near the
        # origin with only a small perturbation (uniform +/-1.0m), while
        # the TARGET is sampled across the full +/-5m box. Getting this
        # wrong (e.g. randomizing both drone and target across the same
        # +/-5m range) would give a different start-to-target distance
        # distribution than Isaac Lab's, making any comparison between the
        # two environments' q_sweep/margin results invalid.
        self.pos[env_ids] = torch.zeros(n, 2, device=self.device).uniform_(-1.0, 1.0)
        # Obstacles are drawn last, so a seeded Stage 0 run reproduces older runs.
        self.target[env_ids] = (torch.rand(n, 2, device=self.device) * 2 - 1) * TARGET_POS_RANGE
        self.yaw[env_ids] = torch.rand(n, device=self.device) * 2 * math.pi - math.pi
        self.step_count[env_ids] = 0
        self.prev_action[env_ids] = 0.0
        self.pose_hist_pos[env_ids] = self.pos[env_ids].unsqueeze(1)
        self.pose_hist_yaw[env_ids] = self.yaw[env_ids].unsqueeze(1)
        self.pose_similarity[env_ids] = 0.0
        self.pose_delta[env_ids] = 0.0
        if self.num_obstacles > 0:
            near = (self.target[env_ids] - self.pos[env_ids]).norm(dim=-1) < MIN_START_TARGET_DIST
            while near.any():
                ids = env_ids[near]
                self.target[ids] = (torch.rand(len(ids), 2, device=self.device) * 2 - 1) * TARGET_POS_RANGE
                near[near.clone()] = (self.target[ids] - self.pos[ids]).norm(dim=-1) < MIN_START_TARGET_DIST
            self._sample_obstacles(env_ids)

    def _sample_obstacles(self, env_ids):
        """Rejection-sample the active obstacle centers as commit 1643e0e does.
        A center must lie START_KEEPOUT from the start and TARGET_KEEPOUT from the
        target. Obstacle 0 is drawn on the middle half of the start-target segment,
        within OBSTACLE_RADIUS of it sideways, and only where it clears both keepouts,
        so with MIN_START_TARGET_DIST it is always placed. Another slot that fails every
        attempt stays parked, so an episode can have fewer than num_obstacles obstacles."""
        n, k, dev = len(env_ids), self.num_obstacles, self.device
        start, tgt = self.pos[env_ids], self.target[env_ids]
        seg = tgt - start
        length = seg.norm(dim=-1).clamp(min=1e-6)
        perp = torch.stack([-seg[:, 1], seg[:, 0]], dim=-1) / length[:, None]
        out = self._parked_xy[:k].expand(n, -1, -1).clone()
        todo = torch.ones(n, k, dtype=torch.bool, device=dev)
        for _ in range(PLACEMENT_ATTEMPTS):
            cand = (torch.rand(n, k, 2, device=dev) * 2 - 1) * TARGET_POS_RANGE
            lateral = (torch.rand(n, device=dev) * 2 - 1) * OBSTACLE_RADIUS
            # The stretch of the segment, at this sideways offset, that clears both keepouts.
            # The small margin keeps a point at the very edge from failing the >= test by rounding.
            lo = ((START_KEEPOUT * 1.0001) ** 2 - lateral ** 2).sqrt() / length
            hi = 1 - ((TARGET_KEEPOUT * 1.0001) ** 2 - lateral ** 2).sqrt() / length
            lo, hi = lo.clamp(min=0.25), hi.clamp(max=0.75)
            along = lo + torch.rand(n, device=dev) * (hi - lo).clamp(min=0)
            cand[:, 0] = start + seg * along[:, None] + perp * lateral[:, None]
            ok = (((cand - start[:, None]).norm(dim=-1) >= START_KEEPOUT)
                  & ((cand - tgt[:, None]).norm(dim=-1) >= TARGET_KEEPOUT))
            out = torch.where((todo & ok)[..., None], cand, out)
            todo &= ~ok
            if not todo.any():
                break
        self.obstacle_xy[env_ids] = self._parked_xy
        self.obstacle_xy[env_ids, :k] = out
        active = torch.zeros(n, MAX_OBSTACLES, dtype=torch.bool, device=dev)
        active[:, :k] = ~todo
        self.obstacle_active[env_ids] = active

    def reset(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._sample_positions(env_ids)
        obs_dict, _dist = self._get_obs()  # discard dist here; step() re-derives it
        return obs_dict, {}

    def _get_obs(self):
        rel = self.target - self.pos  # (N, 2) world-frame offset to target
        dist = torch.norm(rel, dim=-1)
        # Rotate the world-frame offset into the body frame to get dx, dy
        cos_yaw = torch.cos(self.yaw)
        sin_yaw = torch.sin(self.yaw)
        dx = rel[:, 0] * cos_yaw + rel[:, 1] * sin_yaw
        dy = -rel[:, 0] * sin_yaw + rel[:, 1] * cos_yaw

        obstacles = (self.obstacle_xy[:, :NUM_CYLINDERS], self._circle_r,
                     self.obstacle_xy[:, NUM_CYLINDERS:], self._box_half)
        if self.distance_mode == "coords":
            if self.num_obstacles == 0:
                d = torch.zeros(self.num_envs, 3 * self.k_nearest, device=self.device)
            else:
                d = self._nearest_obstacle_points(obstacles)
        elif self.num_obstacles == 0:
            # Every obstacle is parked, so either mode would return this, without
            # its cost on every Stage 0 step.
            d = torch.full((self.num_envs, self.num_readings), MAX_SENSE_RANGE, device=self.device)
        elif self.distance_mode == "ground_truth":
            d = obstacle_sector_distances(self.pos, self.yaw, *obstacles, SENSOR_HALF_FOV, MAX_SENSE_RANGE,
                                          sectors=self.sectors)
        else:
            d = camera_readings(self.pos, self.yaw, *obstacles)
        if self.cfg.sensor_min_range > 0 and self.distance_mode != "coords":
            d = d.clamp(min=self.cfg.sensor_min_range)

        state = torch.cat([d, self.prev_action, self.pose_delta, dx.unsqueeze(-1), dy.unsqueeze(-1)], dim=-1)
        return {"dqn_state": state}, dist

    def _nearest_obstacle_points(self, obstacles):
        """(N, 3 * k_nearest) coords-mode slots (sx, sy, visible), nearest first."""
        dist, bearing = _nearest_surface(self.pos, *obstacles)
        seen = self.obstacle_active & (dist <= MAX_SENSE_RANGE)
        dist = torch.where(seen, dist, torch.full_like(dist, float("inf")))
        near, idx = torch.topk(dist, self.k_nearest, dim=1, largest=False)
        visible = torch.isfinite(near)
        near = torch.where(visible, near, torch.zeros_like(near))
        rel_bearing = bearing.gather(1, idx) - self.yaw.unsqueeze(1)
        slots = torch.stack([near * torch.cos(rel_bearing), near * torch.sin(rel_bearing), visible.float()], dim=-1)
        return slots.reshape(self.num_envs, -1)

    def step(self, actions_continuous: torch.Tensor):
        """actions_continuous: (N, 3) raw [vx_raw, vz_raw, yaw_rate_raw],
        same format discrete_action_to_continuous() produces -- squashed
        through the SAME tanh formulas the real environment uses, so the
        physical vx/yaw_rate values match exactly."""
        vx_cmd = ((torch.tanh(actions_continuous[:, 0]) + 1.0) / 2.0) * self.cfg.max_xy_vel
        yaw_rate_cmd = torch.tanh(actions_continuous[:, 2]) * self.cfg.max_yaw_rate
        dist_before = torch.norm(self.target - self.pos, dim=-1)

        # Pure kinematic integration -- no thrust, no attitude dynamics.
        self.yaw = self.yaw + yaw_rate_cmd * DT
        self.yaw = torch.atan2(torch.sin(self.yaw), torch.cos(self.yaw))  # wrap to [-pi, pi]
        vx_world = vx_cmd * torch.cos(self.yaw)
        vy_world = vx_cmd * torch.sin(self.yaw)
        self.pos = self.pos + torch.stack([vx_world, vy_world], dim=-1) * DT

        self.step_count += 1
        self._update_pose_similarity()
        turn = torch.sign(yaw_rate_cmd)
        # Nearest speed, since the tanh squash returns each speed only to within float error.
        speeds = torch.tensor(SPEED_FRACTIONS, device=self.device) * self.cfg.forward_speed
        speed = (vx_cmd.unsqueeze(-1) - speeds).abs().argmin(dim=-1)
        self.prev_action = torch.cat([torch.stack([turn == 0, turn > 0, turn < 0], dim=-1).float(),
                                      torch.nn.functional.one_hot(speed, len(SPEED_FRACTIONS)).float()], dim=-1)

        obs_dict, dist = self._get_obs()
        reached = dist < REACHED_THRESHOLD
        out_of_bounds = dist > OUT_OF_BOUNDS_DIST
        timed_out = self.step_count >= self.max_steps
        # The only crash is obstacle contact (no ground, no attitude to lose), so
        # Stage 0 cannot crash. Contact is tested after the move, which cannot
        # tunnel through an obstacle because a step is at most max_xy_vel * DT
        # (0.25 m by default), thinner than any obstacle.
        if self.num_obstacles == 0:
            surface_dist = torch.full_like(dist, math.inf)
        else:
            surface_dist = obstacle_surface_distance(
                self.pos,
                self.obstacle_xy[:, :NUM_CYLINDERS], self._circle_r,
                self.obstacle_xy[:, NUM_CYLINDERS:], self._box_half,
            )
        crashed = surface_dist < DRONE_RADIUS

        terminated = reached | crashed
        truncated = out_of_bounds | timed_out

        # Named like the Isaac env's flags. last_outcomes() and trace_2d.py read them.
        self._last_reached_flag = reached
        self._last_ground_crash = torch.zeros_like(reached)
        self._last_any_contact = crashed
        self._last_out_of_bounds = out_of_bounds

        sensor_dist = obs_dict["dqn_state"][:, :self.num_readings].min(dim=-1).values
        reward, self.last_reward_breakdown = self._compute_reward(
            dist, actions_continuous, reached, crashed, vx_signed=vx_cmd, surface_dist=surface_dist,
            sensor_dist=sensor_dist, dist_before=dist_before, return_breakdown=True)

        # The episode's true last state, which the resample below overwrites
        # for every env that just finished.
        self.pre_reset_state = {
            "pos": self.pos.clone(), "yaw": self.yaw.clone(), "target": self.target.clone(),
            "obstacle_xy": self.obstacle_xy.clone(), "obstacle_active": self.obstacle_active.clone(),
        }
        done = terminated | truncated
        reset_obs = obs_dict
        if done.any():
            done_idx = done.nonzero(as_tuple=True)[0]
            self._sample_positions(done_idx)
            reset_obs, _ = self._get_obs()

        # For a finished env obs_dict still shows the episode that just ended;
        # choose the next action from info["reset_obs"] (post-resample).
        return obs_dict, reward, terminated, truncated, {"reset_obs": reset_obs}

    def _update_pose_similarity(self):
        """Compare the pose just reached with the one SIMILARITY_WINDOW steps back, 1 for the same
        position and heading, then store the current pose in its slot."""
        rows = torch.arange(self.num_envs, device=self.device)
        slot = self.step_count % SIMILARITY_WINDOW
        offset = self.pos - self.pose_hist_pos[rows, slot]
        moved = offset.norm(dim=-1)
        turned = self.yaw - self.pose_hist_yaw[rows, slot]
        cos_yaw, sin_yaw = torch.cos(self.yaw), torch.sin(self.yaw)
        self.pose_delta = torch.stack([offset[:, 0] * cos_yaw + offset[:, 1] * sin_yaw,
                                        -offset[:, 0] * sin_yaw + offset[:, 1] * cos_yaw,
                                        torch.atan2(torch.sin(turned), torch.cos(turned))], dim=-1)
        similarity = torch.exp(-moved / self.SIMILARITY_POS_SCALE - self.pose_delta[:, 2].abs() / self.SIMILARITY_YAW_SCALE)
        # Steps before SIMILARITY_WINDOW have no pose that far back, their slots still hold the start pose.
        self.pose_similarity = torch.where(self.step_count >= SIMILARITY_WINDOW, similarity, torch.zeros_like(similarity))
        self.pose_hist_pos[rows, slot] = self.pos
        self.pose_hist_yaw[rows, slot] = self.yaw

    def last_outcomes(self):
        """One OUTCOMES name per env for the latest step(). Only meaningful
        for envs whose episode ended on that step."""
        flags = torch.stack([
            self._last_reached_flag, self._last_any_contact, self._last_out_of_bounds,
            torch.ones_like(self._last_reached_flag),  # timed_out, the remaining way to end
        ], dim=-1)
        # argmax returns the first True, so OUTCOMES order is the priority order.
        return [OUTCOMES[i] for i in flags.int().argmax(dim=-1).tolist()]

    # Standing still forever costs about -15 discounted at gamma 0.98, still above the -50 crash.
    SIMILARITY_PENALTY = -0.3
    # Metres of travel over SIMILARITY_WINDOW steps that cut the similarity to 1/e. The slowest speed covers
    # 0.15 m in that time, so crawling forward keeps about 5% of the penalty.
    SIMILARITY_POS_SCALE = 0.05
    # Radians of heading change over SIMILARITY_WINDOW steps that cut the similarity to 1/e. Steepest at no
    # change, so one 0.08 rad turn step already drops it to 0.67, while (1 + cos) / 2 was flat there and a
    # stopped drone swinging left and right saw almost no gain from turning one way.
    SIMILARITY_YAW_SCALE = 0.2

    def _compute_reward(self, dist, actions_continuous, reached, crashed,
                         vx_signed=None, surface_dist=None, sensor_dist=None, return_breakdown=False, dist_before=None):
        """Redesigned reward, replacing the additive rew_distance +
        rew_yaw_aligned + rew_action_rate combination (which had two
        separate, independently-farmable exploits diagnosed via
        trace_2d.py: reward for facing the target while not moving, and
        reward for being close while not making progress).

        Following an established pattern from mobile-robot navigation
        literature (e.g. Sathyamoorthy et al.-style designs; see also
        RUMOR, arXiv:2404.16672), the per-step shaping reward is now a
        SINGLE multiplicative term: signed forward velocity times the
        cosine of the heading error. This is structurally exploit-proof
        for the "loiter while facing the target" failure mode: turning in
        place has v=0, which zeroes the ENTIRE term regardless of how
        well-aligned the heading is -- there is no separate "heading
        credit" to bank independently of actually moving. rew_action_rate
        is also removed: with turning contributing nothing on its own,
        there's no longer an incentive to flip-flop between actions, so
        the smoothness penalty that existed specifically to suppress that
        is no longer needed.

        This describes progress_mode "velocity". "distance" pays the drop in target distance instead, and
        turning in place still pays 0 there.
        """
        if vx_signed is None:
            # vx_signed: the commanded forward speed for this step
            # (0 for turn actions, forward_speed for the forward action).
            # Callers that already computed vx_cmd in step() should pass
            # it in directly rather than re-deriving it here.
            vx_signed = ((torch.tanh(actions_continuous[:, 0]) + 1.0) / 2.0) * self.cfg.max_xy_vel

        rel = self.target - self.pos
        cos_yaw = torch.cos(self.yaw)
        sin_yaw = torch.sin(self.yaw)
        dx = rel[:, 0] * cos_yaw + rel[:, 1] * sin_yaw
        horiz_dist = torch.norm(rel, dim=-1).clamp(min=1e-6)
        yaw_cos = dx / horiz_dist

        # REVERTED from yaw_cos**3 (v4): that change reduced reached-rate
        # from 29.0% to 8.0% and made trajectories markedly worse overall,
        # not just at the near-target orbiting case it targeted -- likely
        # because cubing suppresses reward too aggressively across the
        # WHOLE range (e.g. yaw_cos=0.5 -> 0.125, a 4x reduction), not just
        # near yaw_cos=1, which hurt early-training learning of basic
        # "which way to turn" before alignment is even close. Back to
        # linear pending a properly isolated test of the actual mechanism
        # behind the near-target orbiting (hypothesis: turn action's fixed
        # angular step size, not reward shape, prevents settling exactly
        # at zero heading error -- see test_turn_granularity.py).
        if self.cfg.progress_mode == "distance":
            rew_progress = (dist_before - dist) * self.cfg.progress_weight / DT
        else:
            rew_progress = vx_signed * yaw_cos * self.cfg.progress_weight
        rew_reached = reached.float() * 100.0
        rew_crashed = crashed.float() * -50.0
        # surface_dist and the ground_truth readings are measured from the drone's center, so contact is at DRONE_RADIUS, not 0.
        # Camera readings start at the camera mounts, so sensor clearance is only approximate in that mode.
        center_dist = sensor_dist if self.cfg.proximity_clearance == "sensor" else surface_dist
        if center_dist is None or self.cfg.proximity_penalty == 0:
            rew_proximity = torch.zeros_like(rew_progress)
        else:
            clearance = center_dist - DRONE_RADIUS
            margin = self.cfg.proximity_margin
            if self.cfg.proximity_shape == "linear":
                closeness = (1 - clearance / margin).clamp(0, 1)
            else:
                c = clearance.clamp(min=self.cfg.proximity_min_clearance)
                closeness = (1 / c ** 2 - 1 / margin ** 2).clamp(min=0)
            rew_proximity = -self.cfg.proximity_penalty * closeness
            if self.cfg.proximity_travel_ref > 0:
                rew_proximity = rew_proximity * vx_signed * DT / self.cfg.proximity_travel_ref

        rew_similarity = self.pose_similarity * self.SIMILARITY_PENALTY

        total = rew_progress + rew_reached + rew_crashed + rew_similarity + rew_proximity
        if return_breakdown:
            return total, {
                "rew_progress": rew_progress, "rew_reached": rew_reached,
                "rew_crashed": rew_crashed, "rew_similarity": rew_similarity, "rew_proximity": rew_proximity,
                "yaw_cos": yaw_cos, "vx_signed": vx_signed,
            }
        return total

    @property
    def unwrapped(self):
        return self


def add_env_args(parser):
    """The obstacle and distance-input options every 2D script takes."""
    parser.add_argument("--num_obstacles", type=int, default=0,
                        help=f"active obstacles per episode, 0 (Stage 0, pure navigation) to {MAX_OBSTACLES}")
    parser.add_argument("--distance_mode", choices=DISTANCE_MODES, default="ground_truth",
                        help="obstacle inputs, exact sector distances, emulated depth cameras, "
                             "or nearest-surface-point coordinates")
    parser.add_argument("--k_nearest", type=int, default=DEFAULT_K_NEAREST,
                        help="obstacle slots in the state, coords mode only")


def inertia_enabled(cfg):
    return (cfg.inertia_tau_max > 0 or cfg.inertia_accel_max > 0 or cfg.inertia_delay_max > 0
            or bool(cfg.obs_velocity))


def env_from_args(args, num_envs, cfg, device):
    cls = InertialDroneNav2DEnv if inertia_enabled(cfg) else DroneNav2DEnv
    return cls(num_envs=num_envs, cfg=cfg, device=device, num_obstacles=args.num_obstacles,
               distance_mode=args.distance_mode, k_nearest=args.k_nearest)


class InertialDroneNav2DEnv(DroneNav2DEnv):
    """DroneNav2DEnv whose drone cannot follow its commands at once.

    Per substep of DT / inertia_substeps: the yaw rate is the applied command; the world-frame velocity
    vector moves toward speed * heading by a fraction dt / tau, its change capped at accel * dt, so the drone
    keeps its momentum and slides outward in a turn; the position integrates that velocity; contact is
    checked, and a drone that touches an obstacle stops there. The command applied is the one chosen delay
    decisions earlier, hover (zeros) before then. Each episode starts at rest and draws tau, accel and delay
    from the config ranges. With every range at 0 and inertia_substeps 1 the motion is exactly the parent's.

    Everything after the move mirrors DroneNav2DEnv.step() of commit 6ea51cf. The state still shows the
    command the policy chose, and, with obs_velocity, the actual body-frame velocity as two more inputs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        c, n, dev = self.cfg, self.num_envs, self.device
        if not (0 <= c.inertia_tau_min <= c.inertia_tau_max and 0 <= c.inertia_accel_min <= c.inertia_accel_max
                and 0 <= c.inertia_delay_min <= c.inertia_delay_max and c.inertia_substeps >= 1):
            raise ValueError("inertia ranges need 0 <= min <= max and inertia_substeps >= 1")
        if c.inertia_accel_max > 0 and c.inertia_accel_min == 0:
            raise ValueError("inertia_accel_min must be > 0 when an acceleration cap is set")
        self.obs_velocity = bool(c.obs_velocity)
        if self.obs_velocity:
            self.state_dim += 2
        self.substeps = c.inertia_substeps
        self.max_delay = c.inertia_delay_max
        self._gen = torch.Generator()
        self._gen.manual_seed(c.inertia_seed)
        self.vel = torch.zeros(n, 2, device=dev)
        self.cmd_buf = torch.zeros(n, self.max_delay, 2, device=dev)  # (speed, yaw rate) not yet applied, oldest first
        self.tau = torch.zeros(n, device=dev)
        self.accel = torch.full((n,), math.inf, device=dev)
        self.delay = torch.zeros(n, dtype=torch.long, device=dev)
        self._draw_dynamics(torch.arange(n, device=dev))

    def _draw_dynamics(self, env_ids):
        c, k = self.cfg, len(env_ids)
        u = lambda lo, hi: (lo + (hi - lo) * torch.rand(k, generator=self._gen)).to(self.device)
        self.tau[env_ids] = u(c.inertia_tau_min, c.inertia_tau_max)
        accel = u(c.inertia_accel_min, c.inertia_accel_max)
        self.accel[env_ids] = accel if c.inertia_accel_max > 0 else torch.full_like(accel, math.inf)
        self.delay[env_ids] = torch.randint(c.inertia_delay_min, c.inertia_delay_max + 1, (k,),
                                            generator=self._gen).to(self.device)

    def _sample_positions(self, env_ids):
        super()._sample_positions(env_ids)
        if hasattr(self, "vel"):  # not yet while the parent's __init__ runs
            self.vel[env_ids] = 0.0
            self.cmd_buf[env_ids] = 0.0
            self._draw_dynamics(env_ids)

    def _get_obs(self):
        obs, dist = super()._get_obs()
        if self.obs_velocity:
            c, s = torch.cos(self.yaw), torch.sin(self.yaw)
            body = torch.stack([self.vel[:, 0] * c + self.vel[:, 1] * s, -self.vel[:, 0] * s + self.vel[:, 1] * c], dim=-1)
            obs = {"dqn_state": torch.cat([obs["dqn_state"], body], dim=-1)}
        return obs, dist

    def _surface_dist(self):
        if self.num_obstacles == 0:
            return torch.full((self.num_envs,), math.inf, device=self.device)
        return obstacle_surface_distance(self.pos, self.obstacle_xy[:, :NUM_CYLINDERS], self._circle_r,
                                         self.obstacle_xy[:, NUM_CYLINDERS:], self._box_half)

    def step(self, actions_continuous: torch.Tensor):
        vx_cmd = ((torch.tanh(actions_continuous[:, 0]) + 1.0) / 2.0) * self.cfg.max_xy_vel
        yaw_rate_cmd = torch.tanh(actions_continuous[:, 2]) * self.cfg.max_yaw_rate
        pos_before = self.pos.clone()
        dist_before = torch.norm(self.target - self.pos, dim=-1)

        rows = torch.arange(self.num_envs, device=self.device)
        full = torch.cat([self.cmd_buf, torch.stack([vx_cmd, yaw_rate_cmd], dim=-1)[:, None]], dim=1)
        applied = full[rows, self.max_delay - self.delay]
        self.cmd_buf = full[:, 1:]
        v_app, w_app = applied[:, 0], applied[:, 1]

        dts = DT / self.substeps
        k_v = torch.where(self.tau > 0, (dts / self.tau.clamp(min=1e-9)).clamp(max=1.0), torch.ones_like(self.tau))
        cap = (self.accel * dts)[:, None]
        contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        hit_speed = torch.zeros(self.num_envs, device=self.device)
        for _ in range(self.substeps):
            live = ~contact
            new_yaw = self.yaw + w_app * dts
            self.yaw = torch.where(live, torch.atan2(torch.sin(new_yaw), torch.cos(new_yaw)), self.yaw)
            v_target = torch.stack([v_app * torch.cos(self.yaw), v_app * torch.sin(self.yaw)], dim=-1)
            dv = (v_target - self.vel) * k_v[:, None]
            dv = dv * (cap / dv.norm(dim=-1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0)
            self.vel = torch.where(live[:, None], self.vel + dv, self.vel)
            self.pos = torch.where(live[:, None], self.pos + self.vel * dts, self.pos)
            new_contact = live & (self._surface_dist() < DRONE_RADIUS)
            hit_speed = torch.where(new_contact, self.vel.norm(dim=-1), hit_speed)
            contact |= new_contact

        # ---- as in DroneNav2DEnv.step() ----
        self.step_count += 1
        self._update_pose_similarity()
        turn = torch.sign(yaw_rate_cmd)
        speeds = torch.tensor(SPEED_FRACTIONS, device=self.device) * self.cfg.forward_speed
        speed = (vx_cmd.unsqueeze(-1) - speeds).abs().argmin(dim=-1)
        self.prev_action = torch.cat([torch.stack([turn == 0, turn > 0, turn < 0], dim=-1).float(),
                                      torch.nn.functional.one_hot(speed, len(SPEED_FRACTIONS)).float()], dim=-1)

        obs_dict, dist = self._get_obs()
        reached = dist < REACHED_THRESHOLD
        out_of_bounds = dist > OUT_OF_BOUNDS_DIST
        timed_out = self.step_count >= self.max_steps
        surface_dist = self._surface_dist()
        crashed = contact | (surface_dist < DRONE_RADIUS)
        terminated = reached | crashed
        truncated = out_of_bounds | timed_out
        self._last_reached_flag = reached
        self._last_ground_crash = torch.zeros_like(reached)
        self._last_any_contact = crashed
        self._last_out_of_bounds = out_of_bounds

        sensor_dist = obs_dict["dqn_state"][:, :self.num_readings].min(dim=-1).values
        # The distance actually travelled, not the command, scales the proximity penalty (proximity_travel_ref).
        travel_speed = (self.pos - pos_before).norm(dim=-1) / DT
        reward, self.last_reward_breakdown = self._compute_reward(
            dist, actions_continuous, reached, crashed, vx_signed=travel_speed, surface_dist=surface_dist,
            sensor_dist=sensor_dist, dist_before=dist_before, return_breakdown=True)

        self.pre_reset_state = {
            "pos": self.pos.clone(), "yaw": self.yaw.clone(), "target": self.target.clone(),
            "obstacle_xy": self.obstacle_xy.clone(), "obstacle_active": self.obstacle_active.clone(),
            "speed": torch.where(contact, hit_speed, self.vel.norm(dim=-1)),
        }
        done = terminated | truncated
        reset_obs = obs_dict
        if done.any():
            self._sample_positions(done.nonzero(as_tuple=True)[0])
            reset_obs, _ = self._get_obs()
        return obs_dict, reward, terminated, truncated, {"reset_obs": reset_obs}
