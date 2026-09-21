"""
Batch evaluation for the 2D environment -- mirrors dqn_diagnose.py's
--mode=batch (same outcome-distribution table format, same CSV schema)
but runs against DroneNav2DEnv instead of Isaac Lab, so results are
directly comparable between the two.
"""
import argparse
import csv

import torch

from dqn_config import DQNConfig, add_config_args, config_from_args
from dqn_action_space import discrete_action_to_continuous, QNetwork
from dqn_2d_env import DroneNav2DEnv

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_episodes", type=int, default=200)
parser.add_argument("--csv_out", type=str, default=None)
add_config_args(parser)
args = parser.parse_args()
cfg: DQNConfig = config_from_args(args)

device = "cpu"
env = DroneNav2DEnv(num_envs=args.num_envs, cfg=cfg, device=device)
q_net = QNetwork(state_dim=cfg.state_dim, action_dim=cfg.num_actions, hidden_dim=cfg.hidden_dim).to(device)
q_net.load_state_dict(torch.load(args.checkpoint, map_location=device))
q_net.eval()
print(f"[DIAG2D] loaded checkpoint: {args.checkpoint}")

obs_dict, _ = env.reset()
state = obs_dict["dqn_state"]

n = args.num_envs
cum_reward = torch.zeros(n)
ep_step = torch.zeros(n)
episodes = []

while len(episodes) < args.num_episodes:
    with torch.no_grad():
        action_idx = q_net(state).argmax(dim=-1)
    actions_continuous = discrete_action_to_continuous(action_idx, device)
    obs_dict, reward, terminated, truncated, info = env.step(actions_continuous)
    next_state = obs_dict["dqn_state"]
    done = terminated | truncated
    cum_reward += reward
    ep_step += 1

    if done.any():
        done_idx = done.nonzero(as_tuple=True)[0].tolist()
        for i in done_idx:
            if env._last_reached_flag[i].item():
                outcome = "reached"
            elif env._last_out_of_bounds[i].item():
                outcome = "out_of_bounds"
            else:
                outcome = "timed_out"
            episodes.append({
                "episode_id": len(episodes), "outcome": outcome,
                "steps": int(ep_step[i].item()), "total_reward": round(cum_reward[i].item(), 3),
            })
        cum_reward[done_idx] = 0
        ep_step[done_idx] = 0

    state = next_state

print(f"\n[DIAG2D] collected {len(episodes)} episodes\n")
outcome_counts = {}
for e in episodes:
    outcome_counts[e["outcome"]] = outcome_counts.get(e["outcome"], 0) + 1
total = len(episodes)
print(f"[DIAG2D] Outcome distribution (n={total}):")
for k in ["reached", "out_of_bounds", "timed_out"]:
    v = outcome_counts.get(k, 0)
    print(f"  {k:<14}: {v:>4} ({v/total*100:>5.1f}%)")

mean_reward = sum(e["total_reward"] for e in episodes) / total
mean_steps = sum(e["steps"] for e in episodes) / total
print(f"\n[DIAG2D] mean_reward={mean_reward:.2f}  mean_episode_length={mean_steps:.1f}")

if args.csv_out:
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode_id", "outcome", "steps", "total_reward"])
        writer.writeheader()
        writer.writerows(episodes)
    print(f"[DIAG2D] CSV written to {args.csv_out}")
