# Drone-Ranger-RL — DQN Branch

A DQN-based obstacle-avoidance navigation policy for a simulated quadrotor in
NVIDIA Isaac Lab, built as an experimental branch alongside this project's
main PPO-based pipeline (CNN depth-image and "SimpleObs" ground-truth-vector
variants, documented separately).

This branch's motivation comes from studying **LT-DQN** (Ahmed & Tomán,
*Results in Engineering* 30(2026)110150), which proposes a three-tier hybrid
UAV navigation framework: a learned DQN controller handles nominal
open-space flight, a LiDAR+A* planner takes over in a "warning zone" near
obstacles, and a simple reactive rule-based layer handles emergency
proximity. The plan here is split into two parts:

- **Part 1 (this repository):** train a complete, standalone DQN policy that
  handles the whole navigation task on its own — mirroring how the
  reference paper's own DQN layer was trained (end-to-end, exposed to the
  full task rather than a restricted subset of it).
- **Part 2 (not started):** wrap the trained DQN with an A* planner and a
  reactive fallback layer, gated by distance to the nearest obstacle. This
  part has not been implemented yet.

**Current status:** Part 1 is still under active development. Pure
navigation (no obstacles, "Stage 0" of the intended curriculum) has been
worked on extensively; obstacle avoidance ("Stage 1") has been attempted but
not yet solved -- see [Known Issues & Findings](#known-issues--findings)
below for an honest account of where things stand.

---

## Table of Contents

- [Architecture](#architecture)
  - [Perception (state space)](#perception-state-space)
  - [Action space](#action-space)
  - [Reward](#reward)
  - [DQN algorithm](#dqn-algorithm)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Diagnostics](#diagnostics)
  - [Visual inspection](#visual-inspection)
  - [Q-value sweep](#q-value-sweep)
- [Configuration](#configuration)
- [Known Issues & Findings](#known-issues--findings)
- [Current Status / Next Steps](#current-status--next-steps)

---

## Architecture

### Perception (state space)

The task is deliberately restricted to **two dimensions**: the target's
height is pinned to exactly match the drone's fixed initial altitude, and no
action controls vertical motion. The observation is a 5-dimensional vector:

```
state = [d_front, d_left, d_right, dx, dy]
```

- `d_front`, `d_left`, `d_right`: minimum detected distance (meters, capped
  at 10.0 when nothing is in range) along three fixed directions relative to
  the drone's current heading -- straight ahead, 90° left, and 90° right.
  These come from **three independent, real camera sensors** (Isaac Lab
  `TiledCamera` objects) rather than raw ray-cast queries, specifically to
  keep the design sim-to-real viable (a ray-cast has no physical hardware
  analog; a depth camera does). All three use the same 73° HFOV/VFOV camera
  model, matching a real OAK-D Lite's spec. The lateral (left/right) cameras
  use a low 8x8 render resolution since only a single global-minimum
  distance is extracted per camera -- no spatial detail is needed, and this
  keeps render cost down when running three cameras simultaneously.

  Each camera implements its own ground-filtering pass: for every pixel, the
  world-frame height its reported distance would place a hit point at is
  computed, and any hit within a small tolerance of true ground level is
  discarded (treated as "nothing detected") rather than counted as an
  obstacle. This has to be computed per-camera using each camera's own ray
  geometry and mounting orientation -- the front camera's version cannot be
  reused directly for the lateral ones since their ray directions in world
  space differ.

- `dx`, `dy`: the target's position relative to the drone, in the drone's
  own body frame (x = "how far ahead", y = "how far to the left"), z
  discarded per the 2D-only design.

This mirrors the reference paper's own DQN state (3 directional distance
readings from hardware ultrasonic/ToF "Three-Beam" sensors, plus a 2D target
offset) -- same information content, sourced from cameras and computed
inside simulation instead of dedicated point-distance sensors on real
hardware.

### Action space

The policy chooses among **3 discrete actions** each control step:

| Action | vx | yaw_rate | Description |
|---|---|---|---|
| 0: forward | `forward_speed` | 0 | move straight ahead, no turning |
| 1: turn_left | 0 | `+turn_rate` | rotate in place toward the left, no forward motion |
| 2: turn_right | 0 | `-turn_rate` | rotate in place toward the right, no forward motion |

Turning and forward motion are **fully decoupled** into separate actions,
following the convention used in embodied-AI point-goal navigation (e.g.
Habitat's/AI2-THOR's `MoveForward` + `TurnLeft` + `TurnRight`). This is a
deliberate design choice made after diagnosing a serious bug in an earlier
4-action version -- see [Known Issues & Findings](#known-issues--findings).

The chosen discrete action is converted to the environment's expected
3-dimensional continuous control tensor `[vx_raw, vz_raw, yaw_rate_raw]` via
`discrete_action_to_continuous()` in `dqn/dqn_action_space.py`. Because the
environment squashes `vx` through `((tanh(x)+1)/2) * max_xy_vel` (to force
it non-negative -- backward flight would point the forward-facing camera
away from the direction of travel), the raw values fed in are computed via
`arctanh` so the *physical* result after squashing matches the intended
speed exactly.

### Reward

```python
reward = rew_distance + rew_reached + rew_crashed + rew_action_rate + rew_yaw_aligned
```

| Term | Formula | Trigger |
|---|---|---|
| `rew_distance` | `-k * dist` (dense, k=0.01) | every step |
| `rew_reached` | `+100` | terminal, distance < 0.75m |
| `rew_crashed` | `-50` | terminal, ground contact or collision |
| `rew_action_rate` | `-0.001 * ‖Δaction‖²` | every step, discourages rapid action switching |
| `rew_yaw_aligned` | `cos(angle to target) * 0.06`, gated to 0 once well-aligned | every step |

The reward has gone through several iterations, each driven by a diagnosed
failure mode rather than blind tuning:

- **Dense, not potential-based, distance reward.** An earlier version used
  potential-based shaping (reward only for beating the closest-ever
  distance this episode). This is provably exploit-resistant (Ng, Harada &
  Russell 1999) but too sparse for a freshly-initialized network to
  bootstrap learning efficiently. Switched to a dense per-step `-k*dist`
  term, matching OpenAI Gym Reacher's standard baseline
  (`reward_dist = -‖fingertip - target‖`), which also makes "wandering far
  from target" costly every step without needing a separate
  out-of-bounds penalty.
- **`rew_action_rate`** was added after a controlled experiment
  (`test_altitude_forward_only.py`, no learned policy involved) showed that
  *sustained* single-direction turning left altitude essentially flat over
  400 steps, but *alternating* left/right turns every single step collapsed
  altitude from 1.0m to 0.34m within ~40 steps and crashed multiple
  environments -- exactly matching a real-hardware failure mode noted in
  the reference paper's own code ("policy kept switching between turning
  left and right").
- **`rew_yaw_aligned`** was added after decoupling turn and forward into
  separate actions revealed that pure-turn actions produce zero change in
  `dist`, so `rew_distance` alone gives no signal about whether a turn was
  in the correct direction. Went through two failed weight settings before
  landing on the current gated form -- see
  [Known Issues & Findings](#known-issues--findings).

### DQN algorithm

- **Network:** a small fully-connected MLP, `state_dim -> hidden -> hidden
  -> num_actions`, ReLU activations. `hidden_dim` defaults to 128 (raised
  from an initial 64 while investigating whether network capacity was
  limiting decision-boundary quality -- not yet conclusively confirmed).
- **Vanilla DQN** (no Double DQN correction implemented yet) with a
  standard target network (synced every N episodes) and epsilon-greedy
  exploration decaying from 1.0 to 0.05.
- **Experience replay** with a configurable buffer (default 50,000
  transitions; started at the reference paper's 5,000 and was enlarged
  during a catastrophic-forgetting investigation, see below).
- Hyperparameters are centralized in `dqn/dqn_config.py` (see
  [Configuration](#configuration)).

---

## Repository Layout

```
dqn/
├── dqn_config.py         # DQNConfig dataclass: network/action/hyperparameter
│                         # settings, with CLI-override support
├── dqn_action_space.py   # Shared QNetwork + discrete-to-continuous action
│                         # mapping (single source of truth, imported by
│                         # every other script below)
├── train_dqn.py          # Training loop: replay buffer, epsilon-greedy,
│                         # TD updates, checkpoint + TensorBoard logging
├── dqn_diagnose.py        # Evaluation: --mode=batch (many episodes, outcome
│                         # stats + CSV) or --mode=trace (single episode,
│                         # step-by-step Q-values)
├── watch_dqn.py           # GUI visualization with a live Q-value printout
└── q_sweep.py             # Offline (no simulator) sweep of Q-values across
                          # synthetic heading angles, for checking decision-
                          # boundary symmetry and margin size

isaaclab_env/
└── drone_nav_v2_env_dqn.py  # The Isaac Lab DirectRLEnv this project trains
                             # against (see Installation for where this
                             # needs to be placed)
```

`checkpoints/` and `tensorboard/` subdirectories are created under `dqn/` at
training time and are **not** tracked in this repository (see
`.gitignore`) -- they contain per-run model weights and logs, not source
code.

---

## Installation

This project requires a working [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
installation (tested against Isaac Sim 5.1.0). It is not a standalone
package -- the environment file needs to be placed inside your Isaac Lab
checkout for the `gym.make()` task registration to work.

1. Clone this repository.
2. Copy `isaaclab_env/drone_nav_v2_env_dqn.py` into your Isaac Lab
   checkout at:
   ```
   <IsaacLab>/source/isaaclab_tasks/isaaclab_tasks/direct/drone_arl/drone_nav_v2_env_dqn.py
   ```
3. Register the gym task in that directory's `__init__.py`:
   ```python
   from isaaclab_tasks.direct.drone_arl.drone_nav_v2_env_dqn import DroneNavV2SimpleObsEnv as DroneNavV2DQNEnv
   gym.register(
       id="Isaac-DroneNavV2-DQN-ARL-v0",
       entry_point="isaaclab_tasks.direct.drone_arl.drone_nav_v2_env_dqn:DroneNavV2SimpleObsEnv",
       disable_env_checker=True,
       kwargs={
           "env_cfg_entry_point": "isaaclab_tasks.direct.drone_arl.drone_nav_v2_env_dqn:DroneNavV2SimpleObsEnvCfg",
       },
   )
   ```
   (Note: the class names inside `drone_nav_v2_env_dqn.py` still read
   `DroneNavV2SimpleObsEnv`/`...Cfg` -- this file started life as a copy of
   an earlier "SimpleObs" PPO environment and was never renamed. It's
   harmless but can be confusing when reading the code.)
4. Copy the `dqn/` directory anywhere convenient (it doesn't need to live
   inside the Isaac Lab checkout) and run scripts from inside it, using
   Isaac Lab's own Python interpreter, e.g.:
   ```bash
   cd dqn
   /path/to/isaac-sim/python.sh train_dqn.py --num_envs=64 --num_episodes=1500
   ```

---

## Usage

### Training

```bash
cd dqn
/path/to/isaac-sim/python.sh train_dqn.py \
  --num_envs=64 --num_episodes=1500 --max_steps_per_episode=100 \
  --run_name=my_experiment
```

- `--resume_from=<path>`: resume from a saved checkpoint's weights.
- `--resume_optimizer_from=<path>`: additionally restore Adam's optimizer
  state (momentum/variance) rather than starting it fresh. Without this,
  every resume restarts Adam's internal statistics from zero, which -- due
  to Adam's early-iteration bias-correction behavior -- can produce
  larger-than-usual gradient steps right after resuming.
- Any `DQNConfig` field can be overridden on the command line, e.g.
  `--hidden_dim=64 --gamma=0.95` (see [Configuration](#configuration)).

Checkpoints are saved to `checkpoints/<run_name>_ep<N>.pt` (every 100
episodes) and `checkpoints/<run_name>_final.pt`, with the run name as a
prefix specifically so that different runs never silently overwrite each
other's checkpoints.

TensorBoard logs go to `tensorboard/<run_name>/`, with scalar groups
`Loss/*` (TD error, Q-value statistics), `Train/*` (return, epsilon, buffer
size), and `termination/*` (rolling-window outcome distribution: reached /
ground_crash / contact_crash / out_of_bounds / timed_out).

### Diagnostics

```bash
# Batch evaluation: N episodes, greedy policy, outcome distribution + CSV
/path/to/isaac-sim/python.sh dqn_diagnose.py --mode=batch \
  --checkpoint=checkpoints/my_experiment_final.pt \
  --num_envs=16 --num_episodes=30 --max_steps_per_episode=1000 \
  --csv_out=results.csv \
  --tb_log_dir=tensorboard/my_experiment --tb_step=1500

# Trace mode: single episode, step-by-step Q-values and state
/path/to/isaac-sim/python.sh dqn_diagnose.py --mode=trace \
  --checkpoint=checkpoints/my_experiment_final.pt --max_steps_per_episode=300
```

Passing `--tb_log_dir` (pointed at an existing training run's TensorBoard
directory) appends `Eval/*` and `termination/*` scalars at the given
`--tb_step`, so periodic evaluation results sit on the same timeline as the
training curves.

### Visual inspection

```bash
/path/to/isaac-sim/python.sh watch_dqn.py --checkpoint=checkpoints/my_experiment_final.pt
```

Opens a GUI window and runs the policy live, greedy, with a per-step
Q-value printout in the terminal alongside the visualization.

### Q-value sweep

```bash
/path/to/isaac-sim/python.sh q_sweep.py \
  --checkpoint=checkpoints/my_experiment_final.pt \
  --distances 2.0 6.0 10.0 --angle_step_deg 10.0 --device cpu
```

Does **not** require the simulator -- constructs synthetic state vectors
across a grid of heading angles (-180° to +180°) at fixed distances,
directly queries the network, and reports the argmax action and the margin
(Q-value gap between the top-2 actions) at every point, plus a left/right
symmetry check. This has been the single most useful diagnostic tool for
catching decision-boundary problems that don't show up clearly in
individual episode traces (see below).

---

## Configuration

All network, action-space, and training hyperparameters live in
`dqn/dqn_config.py` as a single `DQNConfig` dataclass. Every field is
automatically exposed as a CLI flag on all four scripts via
`add_config_args()` / `config_from_args()`, e.g.:

```bash
python3 train_dqn.py --hidden_dim=64 --learning_rate=5e-4 --gamma=0.995
```

Reward weights (distance scale, yaw-alignment weight/threshold, etc.) are
**not** managed here -- they live in the environment's own `@configclass`
in `drone_nav_v2_env_dqn.py`, following Isaac Lab's convention and keeping
DQN-algorithm config separate from environment/task config.

---

## Known Issues & Findings

This section records diagnosed problems and their fixes, since several of
them are non-obvious and easy to reintroduce by accident.

**1. Action-space bug (v1, now fixed): a "pure turn" action wasn't actually
pure.** An earlier 4-action design included a "turn toward target" action
intended to have zero forward velocity. Its raw `vx` input was left at its
default of `0.0` -- but the environment squashes `vx` through
`((tanh(x)+1)/2) * max_xy_vel`, and an input of exactly `0.0` maps to the
*midpoint* of the output range (1.25 m/s), not the minimum. This made that
one action the only one of the four that both moved forward **and**
adapted its turn direction to the target; the other three had turn
directions fixed regardless of where the target actually was. The learned
policy ended up depending almost entirely on that single accidental
"smart" action -- a fragile dependency that collapsed under training
disturbances like resuming from a checkpoint. **Fix:** the current 3-action
design (see [Action space](#action-space)) makes every action's `vx`
explicit and correctly computed, and gives turning its own dedicated
actions instead of overloading a forward-motion action with adaptive
steering.

**2. Catastrophic forgetting when introducing obstacles.** Resuming
training from a working pure-navigation checkpoint to introduce a single
obstacle (curriculum "Stage 1") repeatedly degraded navigation ability --
and this happened even when the obstacle was forced back to zero (i.e.,
identical to Stage 0 except for having been resumed rather than trained
from scratch), which ruled out anything obstacle-specific as the cause.
This is catastrophic forgetting, a well-documented phenomenon in
continual/curriculum DQN training (Ribeiro, Melo & Dias 2019; Kirkpatrick
et al. 2017). Root cause: `train_dqn.py` creates a fresh, empty replay
buffer on every run; `--resume_from` only loads network weights, not any
retained experience, so as training progresses the buffer fills almost
entirely with new-scenario transitions and gradient updates stop
reinforcing previously-learned judgments for states visited less often
under the improving policy. A partial mitigation ("rehearsal mixing" --
randomly relocating the obstacle far away for ~50% of resets, so the
replay buffer continuously retains a mix of pure-nav and obstacle-avoidance
transitions) is implemented in the environment's `_reset_idx()` but has not
been fully re-validated since the action-space redesign.

**3. Reward weight tuning for yaw alignment went through a "goldilocks"
problem.** With `rew_yaw_aligned` weight at 0.02, pure-turn actions lost
out to `rew_distance`'s steady per-step cost, causing the policy to
prematurely abandon turning before heading actually converged (visible as
"arcing" flight -- short turn, short forward hop, repeat). Raising the
weight to 0.06 fixed that but created the opposite exploit: once roughly
facing the target, oscillating left/right near that heading farmed
`rew_yaw_aligned` indefinitely without ever committing to move forward (0%
of episodes reached the target). **Fix:** gate the reward to zero once
heading is already well-aligned (`yaw_cos > 0.9`), removing the "hover near
correct heading" steady-state exploit.

**4. A genuine, currently-unexplained periodic oscillation.** Even with the
above fix, `q_sweep.py` and GUI observation both show the policy getting
stuck oscillating between turn_left and turn_right in certain heading-angle
ranges (Q-value margins as small as 0.001-0.02) rather than committing to
one direction. A controlled step-response test confirmed the low-level yaw
controller itself is *not* the cause (a sustained fixed yaw-rate command
converges cleanly within ~20 control steps, no oscillation) -- the network
itself is choosing to alternate. Extending training from 1,500 to 4,500
episodes did not shrink these margins or improve left/right symmetry,
suggesting the network had already converged to a stable but incorrect
decision boundary rather than still being under-trained. The current
hypothesis under test is that a 64-64 hidden layer is insufficient network
capacity for this decision boundary; a 128-128 version has been trained but
not yet conclusively evaluated (see [Current Status](#current-status--next-steps)).
If enlarging the network doesn't resolve it, Double DQN (to address
possible Q-value overestimation noise, given how small these margins are)
and reward-shaping via hard-case injection (to ensure mid-range heading
angles are adequately sampled during training) are the next candidates, in
that order.

**5. Checkpoint filename collisions.** Every training run used to write to
a fixed filename (`dqn_checkpoint_final.pt`), meaning any subsequent run --
including quick smoke tests -- silently overwrote it, destroying
intermediate results needed for comparison. This has been fixed:
checkpoints are now named `<run_name>_final.pt` / `<run_name>_ep<N>.pt`.

---

## Current Status / Next Steps

As of this writing, the 3-action, `hidden_dim=128` pure-navigation (Stage
0, no obstacles) configuration has just finished training and is pending
evaluation with `q_sweep.py` and `dqn_diagnose.py` to check whether the
larger network resolved the oscillation/margin problems described in
[Finding 4](#known-issues--findings) above. Obstacle avoidance (Stage 1)
has not been revisited since the action-space redesign and the
catastrophic-forgetting mitigation will need re-validation once Stage 0 is
solid.

Planned next steps, in priority order:
1. Evaluate the 128-128 network; if margins/symmetry improve, this
   confirms network capacity was the bottleneck.
2. If not, implement Double DQN to rule out Q-value overestimation as a
   contributor to the small, unstable margins.
3. If still unresolved, investigate training-data distribution -- whether
   mid-range heading angles are being adequately sampled during training,
   with hard-case injection as a possible fix.
4. Once Stage 0 reliably converges, re-validate the rehearsal-mixing fix
   for catastrophic forgetting on Stage 1 (single obstacle).
5. Part 2 (A* + reactive fallback wrapping the trained DQN) has not been
   started.
