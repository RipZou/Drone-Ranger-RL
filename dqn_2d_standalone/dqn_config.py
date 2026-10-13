"""
Unified configuration for the DQN experiments -- network architecture,
action-space physical parameters, and training hyperparameters.

Supports command-line overrides (argparse) so A/B comparisons (e.g.
hidden_dim=64 vs 128) don't require manually editing code.

Reward weights (yaw_aligned, etc.) are NOT managed here -- those stay in
the environment file drone_nav_v2_env_dqn.py, following Isaac Lab's own
@configclass convention. This keeps DQN-algorithm config and
environment/task config as separate concerns.
"""
import argparse
from dataclasses import dataclass, fields


@dataclass
class DQNConfig:
    # ── Network architecture ──
    # state_dim is NOT a field any more: it is derived (see property at the
    # bottom of this class) from num_obstacles / k_nearest.
    num_actions: int = 3
    hidden_dim: int = 128  # raised from 64 -- testing whether network
                            # capacity was the bottleneck (see experiment log)

    # ── Action-space physical parameters (forward speed / turn rate) ──
    forward_speed: float = 1.5
    turn_rate: float = 0.8
    max_xy_vel: float = 2.5
    max_yaw_rate: float = 1.57

    # ── Training hyperparameters (initial values from the LT-DQN paper's
    # Table; replay_buffer_size was later enlarged from the paper's 5000) ──
    replay_buffer_size: int = 50000
    batch_size: int = 64
    learning_rate: float = 1e-3
    gamma: float = 0.98
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    target_update_every: int = 10
    resume_epsilon: float = 0.3  # epsilon restart value on --resume_from
                                  # (neither 1.0 nor epsilon_min)

    # ── Obstacles / state (2D standalone env; ground-truth coordinates) ──
    num_obstacles: int = 0        # 0 = Stage 0 (state is just [dx, dy])
    k_nearest: int = 3            # obstacles exposed to the network (nearest first)
    obstacle_radius: float = 0.5  # m, circular obstacles
    drone_radius: float = 0.2     # m, crash if centre distance < obstacle_radius + drone_radius
    obstacle_clearance: float = 0.5   # m, extra spawn gap around start and target
    first_obstacle_on_path: int = 1   # 1 = obstacle #0 is spawned on the start->target segment
    sense_range: float = 10.0     # m, obstacles whose surface is farther than this are invisible
    state_scale: float = 10.0     # state is divided by this (1.0 = old unnormalised scale)

    @property
    def state_dim(self) -> int:
        """[dx, dy] + 3 numbers (ox, oy, radius) per exposed obstacle."""
        return 2 + 3 * min(self.k_nearest, self.num_obstacles)


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add one command-line argument per DQNConfig field to an existing
    argparse parser (e.g. --hidden_dim), using DQNConfig's dataclass
    defaults as the argparse defaults. Callers build their own parser,
    call this to add these fields, then pass the parsed args to
    config_from_args() to get a DQNConfig with any CLI overrides applied."""
    defaults = DQNConfig()
    for f in fields(DQNConfig):
        parser.add_argument(f"--{f.name}", type=type(f.default), default=getattr(defaults, f.name))


def config_from_args(args: argparse.Namespace) -> DQNConfig:
    """Build a DQNConfig from already-parsed args (must include every
    field added by add_config_args)."""
    field_names = {f.name for f in fields(DQNConfig)}
    kwargs = {k: v for k, v in vars(args).items() if k in field_names}
    return DQNConfig(**kwargs)
