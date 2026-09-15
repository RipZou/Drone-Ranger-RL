"""Visual check -- loads a trained DQN, single environment, opens a GUI
window, runs with a purely greedy policy."""
import argparse

import torch

from isaaclab.app import AppLauncher

from dqn_config import DQNConfig, add_config_args, config_from_args

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-DroneNavV2-DQN-ARL-v0")
parser.add_argument("--checkpoint", type=str, required=True)
add_config_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
cfg: DQNConfig = config_from_args(args)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401

from dqn_action_space import discrete_action_to_continuous, ACTION_NAMES, QNetwork

device = "cuda:0"


env_cfg = parse_env_cfg(args.task, num_envs=1)
env = gym.make(args.task, cfg=env_cfg)
unwrapped = env.unwrapped

q_net = QNetwork(state_dim=cfg.state_dim, action_dim=cfg.num_actions, hidden_dim=cfg.hidden_dim).to(device)
q_net.load_state_dict(torch.load(args.checkpoint, map_location=device))
q_net.eval()
print(f"[WATCH] loaded: {args.checkpoint}")

obs_dict, _ = env.reset()
state = obs_dict["dqn_state"]

prev_action = None
step_count = 0

header = (f"{'step':>5} {'altitude':>9} {'d_front':>8} "
          + " ".join(f"{'Q'+str(i)+'('+ACTION_NAMES[i]+')':>14}" for i in range(cfg.num_actions))
          + f" {'chosen':>10}")
print(header)

while simulation_app.is_running():
    with torch.no_grad():
        q_values = q_net(state)
        action_idx = q_values.argmax(dim=-1)

    chosen = action_idx[0].item()
    switched = (prev_action is not None and chosen != prev_action)
    marker = " <-switched!" if switched else ""
    prev_action = chosen
    q_vals = q_values[0].tolist()

    altitude = unwrapped._robot.data.root_pos_w[0, 2].item()
    row = (f"{step_count:>5} {altitude:>9.3f} {state[0,0].item():>8.2f} "
           + " ".join(f"{q:>14.3f}" for q in q_vals)
           + f" {ACTION_NAMES[chosen]:>10}{marker}")
    print(row)
    step_count += 1

    actions_continuous = discrete_action_to_continuous(action_idx, device)
    obs_dict, reward, terminated, truncated, info = env.step(actions_continuous)
    state = obs_dict["dqn_state"]

    if (terminated | truncated).any():
        print(f"  >>> episode ended (step {step_count}) <<<")
        step_count = 0

simulation_app.close()
