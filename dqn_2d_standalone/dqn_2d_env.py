"""
Pure-2D, Isaac-Lab-free training environment -- isolates the DQN/reward/
network design from any Isaac Lab simulation detail (physics stepping,
attitude controller dynamics, camera rendering) to test whether the
oscillation/low-margin problem seen in q_sweep.py is caused by the DQN
side of things or by something in the Isaac Lab simulation layer.

Kinematics only: no thrust, no attitude, no gravity. Position and heading
are integrated directly from the same discrete action -> (vx, yaw_rate)
mapping used by the real environment, at the same control frequency
(dt=0.02s, matching Isaac Lab's sim.dt=1/100 * decimation=2).

State layout is identical to the real environment:
    state = [d_front, d_left, d_right, dx, dy]
For Stage 0 (no obstacles), d_front/d_left/d_right are always pinned to
10.0 (max sensing range) -- this matches what the real environment's
cameras report when nothing is in view, so this substitution is exact for
Stage 0, not an approximation.

Interface intentionally mirrors gymnasium's reset()/step() so train_dqn.py
needs minimal changes to switch environments.
"""
import math

import torch

from dqn_config import DQNConfig

DT = 0.02  # seconds per control step, matches Isaac Lab's 50Hz control rate
MAX_SENSE_RANGE = 10.0
EPISODE_LENGTH_S = 20.0  # matches the real env's episode_length_s
TARGET_POS_RANGE = 5.0   # start/target sampled uniformly in [-5, 5] on each axis
REACHED_THRESHOLD = 0.75
OUT_OF_BOUNDS_DIST = 40.0  # matches the real env's widened boundary


class DroneNav2DEnv:
    def __init__(self, num_envs: int, cfg: DQNConfig, device: str, num_obstacles: int = 0):
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = device
        self.num_obstacles = num_obstacles  # reserved for Stage 1; unused while 0
        self.max_steps = int(EPISODE_LENGTH_S / DT)

        self.pos = torch.zeros(num_envs, 2, device=device)      # (N, 2) x,y
        self.yaw = torch.zeros(num_envs, device=device)          # (N,) radians
        self.target = torch.zeros(num_envs, 2, device=device)    # (N, 2)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Precompute the world-frame ray direction offsets (front/left/right)
        # relative to current yaw -- recomputed each step since yaw changes.
        self._ray_offsets_rad = torch.tensor([0.0, math.pi / 2, -math.pi / 2], device=device)  # front, left, right

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
        self.target[env_ids] = (torch.rand(n, 2, device=self.device) * 2 - 1) * TARGET_POS_RANGE
        self.yaw[env_ids] = torch.rand(n, device=self.device) * 2 * math.pi - math.pi
        self.step_count[env_ids] = 0

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

        # Stage 0: no obstacles, so all three directions are always at max
        # sensing range -- exactly matches what the real cameras report
        # when nothing is visible (not an approximation for this stage).
        d_front = torch.full((self.num_envs,), MAX_SENSE_RANGE, device=self.device)
        d_left = torch.full((self.num_envs,), MAX_SENSE_RANGE, device=self.device)
        d_right = torch.full((self.num_envs,), MAX_SENSE_RANGE, device=self.device)

        state = torch.stack([d_front, d_left, d_right, dx, dy], dim=-1)
        return {"dqn_state": state}, dist

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
        # Stage 0 has no crash condition -- no ground, no obstacles, no
        # attitude to lose control of. This is intentional: isolating the
        # DQN/reward/network design from Isaac Lab's physics layer means
        # there is nothing left to "crash" into in this minimal setup.
        crashed = torch.zeros_like(reached)

        terminated = reached | crashed
        truncated = out_of_bounds | timed_out

        # Cache termination flags in the same attribute names train_dqn.py
        # and dqn_diagnose.py already look for, so both scripts work
        # against this environment without modification.
        self._last_reached_flag = reached
        self._last_ground_crash = torch.zeros_like(reached)
        self._last_any_contact = torch.zeros_like(reached)
        self._last_out_of_bounds = out_of_bounds

        reward = self._compute_reward(dist, actions_continuous, reached, crashed, vx_signed=vx_cmd)

        done = terminated | truncated
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
