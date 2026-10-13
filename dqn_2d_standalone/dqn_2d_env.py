"""
Pure-2D, Isaac-Lab-free training environment (kinematics only).

Position and heading are integrated directly from the discrete action ->
(vx, yaw_rate) mapping, at the same control frequency as the Isaac Lab
env (dt = 0.02 s).

State (ground-truth coordinates, NO simulated camera), body frame, divided
by cfg.state_scale:
    [dx, dy,  ox_1, oy_1, r_1,  ...,  ox_K, oy_K, r_K]
dx, dy      : target position relative to the drone
ox_i, oy_i  : i-th nearest VISIBLE obstacle centre, relative to the drone
r_i         : its radius
Obstacles are sorted by surface distance (nearest first). An obstacle is
visible if its surface is within cfg.sense_range. Unused slots are padded
with (sense_range, 0, 0) -- radius 0 means "no obstacle here".
K = min(cfg.k_nearest, cfg.num_obstacles), so with num_obstacles = 0 the
state is just [dx, dy].

Interface mirrors gymnasium's reset()/step().
"""
import math

import torch

from dqn_config import DQNConfig

DT = 0.02  # seconds per control step, matches Isaac Lab's 50Hz control rate
EPISODE_LENGTH_S = 20.0  # matches the real env's episode_length_s
TARGET_POS_RANGE = 5.0   # target (and obstacles) sampled uniformly in [-5, 5] per axis
REACHED_THRESHOLD = 0.75
OUT_OF_BOUNDS_DIST = 40.0  # matches the real env's widened boundary
FAR_AWAY = 1.0e4           # parking spot for obstacles that could not be placed


class DroneNav2DEnv:
    def __init__(self, num_envs: int, cfg: DQNConfig, device: str, num_obstacles=None):
        if num_obstacles is not None and num_obstacles != cfg.num_obstacles:
            raise ValueError(
                f"num_obstacles={num_obstacles} conflicts with cfg.num_obstacles="
                f"{cfg.num_obstacles}; set it via cfg / --num_obstacles only "
                f"(cfg.state_dim depends on it).")
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = device
        self.num_obstacles = cfg.num_obstacles
        self.k = min(cfg.k_nearest, cfg.num_obstacles)
        self.max_steps = int(EPISODE_LENGTH_S / DT)

        self.pos = torch.zeros(num_envs, 2, device=device)      # (N, 2) x,y
        self.yaw = torch.zeros(num_envs, device=device)          # (N,) radians
        self.target = torch.zeros(num_envs, 2, device=device)    # (N, 2)
        self.obs_pos = torch.full((num_envs, max(self.num_obstacles, 0), 2), FAR_AWAY,
                                  device=device)                 # (N, M, 2) world frame
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    def _sample_positions(self, env_ids):
        n = len(env_ids)
        # Drone starts near the origin (+/-1 m); the TARGET is sampled across
        # the full +/-5 m box (same distribution as the Isaac Lab env).
        self.pos[env_ids] = torch.zeros(n, 2, device=self.device).uniform_(-1.0, 1.0)
        self.target[env_ids] = (torch.rand(n, 2, device=self.device) * 2 - 1) * TARGET_POS_RANGE
        self.yaw[env_ids] = torch.rand(n, device=self.device) * 2 * math.pi - math.pi
        self.step_count[env_ids] = 0
        self._sample_obstacles(env_ids)

    def _sample_obstacles(self, env_ids):
        """Rejection-sample obstacle centres. An obstacle is valid if it keeps
        a gap from the start and from the target (so the episode is never
        unwinnable at t=0). Obstacle #0 can be forced onto the start->target
        segment so that avoidance is actually required. Obstacles that fail
        30 attempts are parked at FAR_AWAY (invisible, harmless)."""
        M = self.num_obstacles
        if M == 0:
            return
        n = len(env_ids)
        c = self.cfg
        pos0 = self.pos[env_ids]
        tgt = self.target[env_ids]
        keep_start = c.obstacle_radius + c.drone_radius + c.obstacle_clearance
        keep_target = c.obstacle_radius + REACHED_THRESHOLD + c.obstacle_clearance

        out = torch.full((n, M, 2), FAR_AWAY, device=self.device)
        todo = torch.ones(n, M, dtype=torch.bool, device=self.device)
        for _ in range(30):
            cand = (torch.rand(n, M, 2, device=self.device) * 2 - 1) * TARGET_POS_RANGE
            if c.first_obstacle_on_path:
                seg = tgt - pos0
                seg_len = torch.norm(seg, dim=-1).clamp(min=1e-6)
                perp = torch.stack([-seg[:, 1], seg[:, 0]], dim=-1) / seg_len[:, None]
                t = torch.rand(n, device=self.device) * 0.5 + 0.25          # 25%..75% along
                lateral = (torch.rand(n, device=self.device) * 2 - 1) * c.obstacle_radius
                cand[:, 0, :] = pos0 + seg * t[:, None] + perp * lateral[:, None]
            ok = ((torch.norm(cand - pos0[:, None, :], dim=-1) >= keep_start)
                  & (torch.norm(cand - tgt[:, None, :], dim=-1) >= keep_target))
            take = todo & ok
            out = torch.where(take[..., None], cand, out)
            todo = todo & ~ok
            if not todo.any():
                break
        self.obs_pos[env_ids] = out

    def reset(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._sample_positions(env_ids)
        obs_dict, _dist = self._get_obs()  # discard dist here; step() re-derives it
        return obs_dict, {}

    def _body_frame(self, rel_world):
        """Rotate world-frame offsets (..., 2) into the drone body frame.
        rel_world's leading dims must broadcast against (N,)."""
        cos_yaw = torch.cos(self.yaw)
        sin_yaw = torch.sin(self.yaw)
        while cos_yaw.dim() < rel_world.dim() - 1:
            cos_yaw = cos_yaw.unsqueeze(-1)
            sin_yaw = sin_yaw.unsqueeze(-1)
        bx = rel_world[..., 0] * cos_yaw + rel_world[..., 1] * sin_yaw
        by = -rel_world[..., 0] * sin_yaw + rel_world[..., 1] * cos_yaw
        return bx, by

    def _get_obs(self):
        c = self.cfg
        rel = self.target - self.pos
        dist = torch.norm(rel, dim=-1)
        dx, dy = self._body_frame(rel)
        feats = [torch.stack([dx, dy], dim=-1)]

        if self.k > 0:
            orel = self.obs_pos - self.pos[:, None, :]            # (N, M, 2)
            ox, oy = self._body_frame(orel)                       # (N, M)
            surf = torch.norm(orel, dim=-1) - c.obstacle_radius   # distance to surface
            surf = torch.where(surf <= c.sense_range, surf, torch.full_like(surf, float("inf")))
            vals, idx = torch.topk(surf, self.k, dim=1, largest=False)   # nearest first
            visible = torch.isfinite(vals)
            gx = torch.where(visible, torch.gather(ox, 1, idx), torch.full_like(vals, c.sense_range))
            gy = torch.where(visible, torch.gather(oy, 1, idx), torch.zeros_like(vals))
            gr = torch.where(visible, torch.full_like(vals, c.obstacle_radius), torch.zeros_like(vals))
            feats.append(torch.stack([gx, gy, gr], dim=-1).reshape(self.num_envs, 3 * self.k))

        state = torch.cat(feats, dim=-1) / c.state_scale
        return {"dqn_state": state}, dist

    def _collided(self):
        """True where the drone overlaps any obstacle (analytic circle test)."""
        if self.num_obstacles == 0:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        d = torch.norm(self.obs_pos - self.pos[:, None, :], dim=-1)
        return (d < self.cfg.obstacle_radius + self.cfg.drone_radius).any(dim=1)

    def step(self, actions_continuous: torch.Tensor):
        """actions_continuous: (N, 3) raw [vx_raw, vz_raw, yaw_rate_raw],
        same format discrete_action_to_continuous() produces -- squashed
        through the SAME tanh formulas the real environment uses, so the
        physical vx/yaw_rate values match exactly."""
        vx_cmd = ((torch.tanh(actions_continuous[:, 0]) + 1.0) / 2.0) * self.cfg.max_xy_vel
        yaw_rate_cmd = torch.tanh(actions_continuous[:, 2]) * self.cfg.max_yaw_rate

        # Pure kinematic integration -- no thrust, no attitude dynamics.
        self.yaw = self.yaw + yaw_rate_cmd * DT
        self.yaw = torch.atan2(torch.sin(self.yaw), torch.cos(self.yaw))  # wrap to [-pi, pi]
        vx_world = vx_cmd * torch.cos(self.yaw)
        vy_world = vx_cmd * torch.sin(self.yaw)
        self.pos = self.pos + torch.stack([vx_world, vy_world], dim=-1) * DT

        self.step_count += 1

        obs_dict, dist = self._get_obs()
        reached = dist < REACHED_THRESHOLD
        out_of_bounds = dist > OUT_OF_BOUNDS_DIST
        timed_out = self.step_count >= self.max_steps
        # Crash = overlap with any circular obstacle (always False when
        # num_obstacles == 0). Reaching the target wins ties.
        crashed = self._collided() & ~reached

        terminated = reached | crashed
        truncated = out_of_bounds | timed_out

        # Cache termination flags in the same attribute names train_dqn.py
        # and dqn_diagnose.py already look for, so both scripts work
        # against this environment without modification.
        self._last_reached_flag = reached
        self._last_ground_crash = torch.zeros_like(reached)
        self._last_any_contact = crashed
        self._last_out_of_bounds = out_of_bounds

        reward = self._compute_reward(dist, actions_continuous, reached, crashed, vx_signed=vx_cmd)

        done = terminated | truncated
        # [obstacles-patch] Snapshot BEFORE finished envs are resampled below, so
        # trace/plot scripts can read the true final state of an episode:
        # (pos, yaw, target, obs_pos).
        self._pre_reset = (self.pos.clone(), self.yaw.clone(), self.target.clone(), self.obs_pos.clone())
        if done.any():
            done_idx = done.nonzero(as_tuple=True)[0]
            self._sample_positions(done_idx)

        return obs_dict, reward, terminated, truncated, {}

    # Weight on the multiplicative progress term. v*cos(theta) ranges
    # [-forward_speed, +forward_speed] (1.5 at default settings), so this
    # keeps per-step reward in roughly [-0.075, +0.075] -- a similar scale
    # to the old rew_distance term, for continuity with earlier runs.
    K_PROGRESS = 0.05

    def _compute_reward(self, dist, actions_continuous, reached, crashed,
                         vx_signed=None, return_breakdown=False):
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
        """
        if vx_signed is None:
            # vx_signed: the actual physical forward speed for this step
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
        rew_progress = vx_signed * yaw_cos * self.K_PROGRESS
        rew_reached = reached.float() * 100.0
        rew_crashed = crashed.float() * -50.0

        total = rew_progress + rew_reached + rew_crashed
        if return_breakdown:
            return total, {
                "rew_progress": rew_progress, "rew_reached": rew_reached,
                "rew_crashed": rew_crashed, "yaw_cos": yaw_cos, "vx_signed": vx_signed,
            }
        return total

    @property
    def unwrapped(self):
        return self
