# DQN 2D Standalone — Isaac-Lab-Free Training

A standalone, dependency-light version of this project's DQN training loop,
running against a pure-2D kinematic environment instead of the full Isaac
Lab simulation. No Isaac Sim install, no GPU, and no macOS/Linux-specific
setup required -- just Python and a few pip packages.

## Why this exists

The main Drone-Ranger-RL project trains the same DQN algorithm inside
Isaac Lab, which requires a full Isaac Sim installation and is Linux/GPU-only.
This 2D version exists to isolate the DQN algorithm, network, action
space, and reward design from any simulation-layer detail (physics
stepping, attitude control dynamics, rendering) -- useful both for
teammates without an Isaac Lab setup (e.g. on a Mac) and for quickly
iterating on reward/algorithm changes (a 1500-episode run takes under 10
minutes on CPU here, versus roughly an hour in Isaac Lab).

The environment (`dqn_2d_env.py`) uses simple kinematic integration
(position += velocity * dt, heading += yaw_rate * dt) with the exact same
control frequency, physical speed/turn-rate parameters, and reward
structure as the full Isaac Lab environment (Stage 0: pure navigation, no
obstacles) -- see the main README's "Known Issues & Findings" section for
the diagnosed problems this comparison uncovered.

## Installation

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

That's it -- no Isaac Sim, no CUDA required (training runs on CPU
automatically if no GPU is available).

## Usage

### Training

```bash
python3 train_dqn.py --num_envs=64 --num_episodes=1500 --run_name=my_run
```

Checkpoints are saved to `checkpoints/<run_name>_ep<N>.pt` and
`checkpoints/<run_name>_final.pt` (relative to this script's own
directory, not your current working directory -- so it works the same
regardless of where you run it from). TensorBoard logs go to
`tensorboard/<run_name>/`.

Any `DQNConfig` field can be overridden on the command line, e.g.
`--hidden_dim=64 --gamma=0.95`.

- `--resume_from=<path>`: resume from a saved checkpoint's weights.
- `--resume_optimizer_from=<path>`: also restore Adam's optimizer state.
- `--output_dir=<path>`: change where `checkpoints/` and `tensorboard/`
  are created (defaults to this script's own directory).

### Batch evaluation

```bash
python3 dqn_diagnose_2d.py --checkpoint=checkpoints/my_run_final.pt \
  --num_envs=64 --num_episodes=200 --csv_out=results.csv
```

Runs the policy greedily for N episodes and reports the outcome
distribution (reached / out_of_bounds / timed_out).

### Step-by-step trace

```bash
python3 trace_2d.py --checkpoint=checkpoints/my_run_final.pt --max_steps=300 --seed=42
```

Prints every step's chosen action, position, distance, heading alignment,
and reward breakdown for one episode. `--seed` lets you reproduce a
specific start/target pair (useful for revisiting a case you've already
found interesting).

### Trajectory plot

```bash
python3 plot_2d_trajectory.py --checkpoint=checkpoints/my_run_final.pt \
  --num_episodes=8 --max_steps=1000 --out=trajectory.png
```

Runs several episodes and plots the drone's top-down path with matplotlib
(circle = start, star = target).

### Q-value sweep

```bash
python3 q_sweep.py --checkpoint=checkpoints/my_run_final.pt \
  --distances 2.0 6.0 10.0 --angle_step_deg 10.0 --device cpu
```

Constructs synthetic states across a grid of heading angles and reports
the network's action choice and decision-margin at each point -- no
environment rollout needed. Useful for checking whether the policy's
left/right turn judgment is symmetric and confident.

## Files

| File | Purpose |
|---|---|
| `dqn_config.py` | `DQNConfig` dataclass: all network/action/hyperparameter settings, CLI-overridable |
| `dqn_action_space.py` | Shared `QNetwork` and discrete-to-continuous action mapping |
| `dqn_2d_env.py` | The pure-2D kinematic environment |
| `train_dqn.py` | Training loop (this standalone entry point) |
| `dqn_diagnose_2d.py` | Batch evaluation |
| `trace_2d.py` | Step-by-step episode trace |
| `plot_2d_trajectory.py` | matplotlib trajectory visualization |
| `q_sweep.py` | Offline decision-boundary sweep |

## Relationship to the main project

This is a subset of the `dqn/` directory in the main Drone-Ranger-RL
repository, with `train_dqn.py` modified to drop the Isaac Lab code path
and use paths relative to the script's own location instead of a
hardcoded absolute path. Everything else (network, action space, reward,
config system) is identical. See the main repository's README for the
full project context, the Isaac Lab environment, and a detailed account
of diagnosed issues and findings.
