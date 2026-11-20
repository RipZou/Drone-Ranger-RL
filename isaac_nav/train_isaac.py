"""
Train (or fine-tune) the DQN navigation policy in Isaac Lab.

Fine-tune the 2D model chosen for transfer (one camera, 0.4 m minimum range, velocity input):
    python isaac_nav/train_isaac.py --resume_from <2D s8_final.pt> --num_obstacles 8 --num_episodes 1500 \
        --camera_count 1 --sensor_min_range 0.4 --obs_velocity 1 --output_dir isaac_runs/ft_c2 --headless

The DQN options (--camera_count, --sensor_min_range, --obs_velocity, ...) are those of the 2D scripts, and
must match the checkpoint. --sensor geometry replaces the camera with the 2D env's exact readings, to tell
sensing differences from dynamics differences.
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
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_episodes", type=int, default=1500)
parser.add_argument("--max_steps_per_episode", type=int, default=100)
parser.add_argument("--num_obstacles", type=int, default=8)
parser.add_argument("--sensor", choices=("camera", "geometry"), default="camera")
parser.add_argument("--resume_from", type=str, default=None)
parser.add_argument("--resume_optimizer_from", type=str, default=None)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--run_name", type=str, default="isaac")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save_every", type=int, default=100)
add_config_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
cfg = config_from_args(args)

app = AppLauncher(args).app

import random  # noqa: E402

import torch  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from nav_env import make_env  # noqa: E402
from dqn_trainer import train  # noqa: E402

random.seed(args.seed)
torch.manual_seed(args.seed)
os.makedirs(args.output_dir, exist_ok=True)
log_path = os.path.join(args.output_dir, "train.log")
log_file = open(log_path, "a")


def log(msg):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()


env = make_env(cfg, args.num_envs, args.num_obstacles, sensor=args.sensor, seed=args.seed)
log("=" * 70)
log(f"[ISAAC DQN] num_envs={args.num_envs} num_obstacles={args.num_obstacles} sensor={args.sensor} "
    f"state_dim={env.state_dim} episodes={args.num_episodes} resume_from={args.resume_from}")
log(f"  camera_count={cfg.camera_count} sensor_min_range={cfg.sensor_min_range} obs_velocity={cfg.obs_velocity}")
log("=" * 70)
writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard", args.run_name))
try:
    train(env, cfg, args.num_episodes, args.output_dir, args.run_name, args.max_steps_per_episode,
          args.resume_from, args.resume_optimizer_from, args.save_every, writer, log)
finally:
    writer.close()
    log_file.close()
    env.close()
    app.close()
