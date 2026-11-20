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

# The forward speeds the actions choose from, as fractions of forward_speed.
# At 0.2 the slowest turn radius was 0.375 m, too wide to steer off an obstacle already close,
# and 0.05 brings it to about 0.09 m. 0 adds a stop, so a turn in place drifts not at all.
SPEED_FRACTIONS = (0.0, 0.05, 0.2, 1.0)


@dataclass
class DQNConfig:
    # ── Network architecture ──
    state_dim: int = 5
    num_actions: int = 12  # len(SPEED_FRACTIONS) speeds times 3 turns
    hidden_dim: int = 128  # raised from 64 -- testing whether network
                            # capacity was the bottleneck (see experiment log)

    # ── Action-space physical parameters (forward speed / turn rate) ──
    forward_speed: float = 1.5
    turn_rate: float = 0.8
    max_xy_vel: float = 2.5
    max_yaw_rate: float = 1.57

    # ── Training hyperparameters (initial values from the LT-DQN paper's
    # Table, some since tuned) ──
    replay_buffer_size: int = 500000  # 50000 held only the last ~780 vector steps of 64 envs
    batch_size: int = 128
    learning_rate: float = 1e-3
    # Each run (one curriculum stage) holds learning_rate until episode lr_decay_start * (num_episodes - 1), then
    # cosine-anneals it to learning_rate * lr_final_fraction by its last episode.
    # WARNING: at lr_final_fraction 1.0 the optimizer's lr is left untouched (a resumed optimizer's lr survives);
    # any other value overwrites it from learning_rate every episode, so a resumed run restarts the schedule.
    # --lr_final_fraction 1.0 turns the taper off.
    lr_final_fraction: float = 0.1
    lr_decay_start: float = 0.6667
    gamma: float = 0.99
    # Rewards summed per replay transition before bootstrapping, discounted by gamma ** (number summed).
    # --n_step 1 restores one-step targets.
    n_step: int = 3
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    target_update_every: int = 20  # counted in episodes, not gradient steps; unused while target_tau > 0
    # Polyak step toward the online net after every gradient step, 0 for the periodic hard copy.
    # 0.0005 gives a time constant of about 2000 gradient steps, matching the hard copy every 20 episodes of 100 steps.
    target_tau: float = 0.0005
    resume_epsilon: float = 0.3  # epsilon restart value on --resume_from
                                  # (neither 1.0 nor epsilon_min)
    # Exploratory actions are held for a zeta(mu) number of steps, at most explore_max_hold.
    # explore_max_hold 1 gives plain per-step epsilon-greedy.
    explore_hold_mu: float = 2.0
    explore_max_hold: int = 100
    # Transitions past step replay_full_steps of their episode are stored with probability replay_late_fraction,
    # except the reaching and crashing steps (and stops, see replay_keep_stopped), so long stuck stretches do not
    # crowd the buffer.
    replay_full_steps: int = 200
    replay_late_fraction: float = 0.1
    # 1 also keeps every late transition whose first action is a stop, turning or not, so the stalls that thinning
    # would mostly drop stay in the buffer. An int, since add_config_args parses a bool field's "false" as True.
    replay_keep_stopped: int = 1
    # Proportional prioritized replay (Schaul et al. 2016). per_alpha 0 keeps uniform sampling.
    per_alpha: float = 0.0
    per_beta_start: float = 0.4  # importance-sampling exponent, raised linearly to 1 by the run's last episode
    per_eps: float = 1e-3  # added to |TD error| so a transition fitted exactly can still be drawn
    # Weight on the progress reward, paid per progress_mode below as v * cos(heading error) or as its
    # distance-drop equivalent. The discrete actions cap v at forward_speed (1.5 m/s), so at 0.05 the reward
    # stays within about [-0.075, +0.075] in either mode. A raw continuous action can reach max_xy_vel
    # (2.5 m/s), which scales that range by 2.5 / 1.5.
    progress_weight: float = 0.05
    # "velocity" pays progress_weight * v * cos(heading error), using the commanded speed. "distance" pays
    # progress_weight / DT times the drop in distance to the target over the step, the same rate per metre,
    # from the positions alone, so it stays unbiased where the real speed falls short of the command.
    progress_mode: str = "distance"
    # Per-step penalty for passing near an obstacle, 0 from proximity_margin metres of clearance outward and
    # growing as the clearance shrinks, to proximity_penalty at contact under the linear shape. 0 leaves the reward unchanged.
    proximity_penalty: float = 0.03
    proximity_margin: float = 0.3
    # "inverse_square" charges proximity_penalty * (1/c^2 - 1/proximity_margin^2) instead, 0 at the margin,
    # with the clearance c held at proximity_min_clearance or more so contact stays finite. Near contact
    # this is far larger than the linear shape, so proximity_penalty is not comparable between the two.
    proximity_shape: str = "inverse_square"
    proximity_min_clearance: float = 0.03
    # "sensor" takes the clearance from the smallest reading of the pose just reached, so an obstacle out
    # of view costs nothing. "true" uses the nearest surface, seen or not. The env rejects "sensor" in
    # coords mode, which has no distance readings, so a coords run passes "true".
    proximity_clearance: str = "sensor"
    # Above 0 the penalty is scaled by the distance moved this step over this many metres, so a stopped
    # drone pays none of it and a slow pass no more in total than a fast one.
    proximity_travel_ref: float = 0.15
    # Decoupled weight decay through AdamW. 0 keeps plain Adam.
    weight_decay: float = 1e-3
    # ── Inertia (InertialDroneNav2DEnv in dqn_2d_env.py). All off by default, which keeps the instant
    # kinematics of baseline_stop_v7.8. Each episode draws its own values uniformly from [min, max], from a
    # generator of its own seeded by inertia_seed, so the layouts drawn from the global seed do not change.
    # The world-frame velocity vector follows speed * heading with time constant tau (s), its change capped
    # at accel (m/s^2, 0 = no cap), and every command takes effect delay decision steps (0.1 s each) late.
    # Isaac Lab's velocity controller (vel_ctrl_kp 4, max_accel 6) is about tau 0.25 and accel 6.
    inertia_tau_min: float = 0.0
    inertia_tau_max: float = 0.0
    inertia_accel_min: float = 0.0
    inertia_accel_max: float = 0.0
    inertia_delay_min: int = 0
    inertia_delay_max: int = 0
    inertia_substeps: int = 5  # motion and contact are integrated in this many substeps of each 0.1 s step
    inertia_seed: int = 12345
    # 1 appends the body-frame velocity (forward, left) in m/s to the state, 2 more inputs. Changes state_dim,
    # so a checkpoint trained without it does not load.
    obs_velocity: int = 0
    # Depth cameras in ground_truth mode: 2 = two 73 deg cameras yawed +/-36.5 deg (146 deg, baseline_stop_v7.8),
    # 1 = one 73 deg camera on the heading (a single OAK-D Lite). Each gives CAMERA_BINS readings, so 1 camera
    # shrinks state_dim by 5 and a checkpoint trained with the other count does not load.
    camera_count: int = 2
    # Readings closer than this (m) report this value, as a stereo camera cannot resolve depth that close
    # (OAK-D Lite: about 0.2 m at 400P with extended disparity, 0.4 m otherwise). 0 = no limit.
    sensor_min_range: float = 0.0


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
