"""
Phase 0: measure how the Isaac drone follows the DQN's commands, to check the 2D inertia ranges
(tau 0.15-0.4 s, accel 4-8 m/s^2, delay 0-2 decisions) cover it.

From a hover, no obstacles, every env runs the same script of discrete actions, 0.1 s each:
  1. full speed straight (1.5 m/s) for 3 s      -> forward-speed rise: delay, time constant, peak accel
  2. stop (speed 0, no turn) for 3 s            -> stopping time and distance
  3. stop and turn left (0.8 rad/s) for 2 s     -> yaw-rate rise
Velocities are recorded every physics step (100 Hz).

    python isaac_nav/step_response.py --headless
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dqn_2d_standalone_oscillation"))
sys.path.insert(0, HERE)

from isaaclab.app import AppLauncher  # noqa: E402
from dqn_config import add_config_args, config_from_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--csv", type=str, default="step_response.csv")
add_config_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
cfg = config_from_args(args)
cfg.camera_count = 1  # the script flies without obstacles; the camera layout does not matter

app = AppLauncher(args).app

import torch  # noqa: E402

from dqn_action_space import discrete_action_to_continuous  # noqa: E402
from dqn_config import SPEED_FRACTIONS  # noqa: E402
from nav_env import make_env  # noqa: E402
from response_fit import summarize  # noqa: E402

FAST = (len(SPEED_FRACTIONS) - 1) * 3 + 0   # top speed, no turn
STOP = 0                                    # speed 0, no turn
STOP_LEFT = 1                               # speed 0, turn left
SCRIPT = [(FAST, 30), (STOP, 30), (STOP_LEFT, 20)]

env = make_env(cfg, args.num_envs, 0, sensor="geometry", seed=0)
try:
    env.reset()
    # Put the target 20 m behind each drone, so the script never reaches it or leaves the bounds.
    bk = env.env.book
    bk.target[:] = bk.pos - 20.0 * torch.stack([torch.cos(bk.yaw), torch.sin(bk.yaw)], dim=-1)
    # Let the drone settle into a hover before the script starts.
    hover = discrete_action_to_continuous(torch.full((env.num_envs,), STOP, device=env.device), env.device, cfg)
    for _ in range(20):
        env.step(hover)
    env.env.physics_trace = []
    for action, steps in SCRIPT:
        a = discrete_action_to_continuous(torch.full((env.num_envs,), action, device=env.device), env.device, cfg)
        for _ in range(steps):
            env.step(a)
    trace = torch.stack(env.env.physics_trace)          # (T, N, 6), every physics step
    dt = env.env.cfg.sim.dt
    phases = [steps * env.env.cfg.decimation for _, steps in SCRIPT]
    summarize(trace, dt, phases, cfg.forward_speed, cfg.turn_rate, args.csv)
finally:
    env.close()
    app.close()
