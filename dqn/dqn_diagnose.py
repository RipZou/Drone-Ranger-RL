"""
Unified diagnostic script -- replaces the old diagnose_orbit_dqn.py and
inspect_q_values.py.

--mode=batch (default): load a checkpoint, run N episodes with a purely
  greedy policy, tally the termination-reason distribution, print a
  summary table + CSV, and optionally append to TensorBoard (Eval/ and
  termination/ groups, matching train_dqn.py's style so results sit on
  the same timeline as training curves).

--mode=trace: a single episode, logging every action's Q-value and key
  state variables step by step -- used to diagnose specific value-function
  issues (e.g. "which action should be picked once facing the target").
"""
import argparse
import csv
import time

import torch

from isaaclab.app import AppLauncher

from dqn_config import DQNConfig, add_config_args, config_from_args

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-DroneNavV2-DQN-ARL-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--mode", type=str, default="batch", choices=["batch", "trace"])
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=30)
parser.add_argument("--max_steps_per_episode", type=int, default=1000)
parser.add_argument("--csv_out", type=str, default=None,
                     help="path to write per-episode results as CSV (batch mode only)")
parser.add_argument("--tb_log_dir", type=str, default=None,
                     help="existing training run's tensorboard dir to append Eval/termination "
                          "results to; if omitted, no tensorboard write happens")
parser.add_argument("--tb_step", type=int, default=None,
                     help="x-axis step to record this evaluation at (e.g. the training episode "
                          "count this checkpoint came from); defaults to current unix timestamp")
add_config_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True
cfg: DQNConfig = config_from_args(args)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401

from dqn_action_space import discrete_action_to_continuous, ACTION_NAMES, QNetwork

device = "cuda:0"


def load_env_and_net(num_envs):
    env_cfg = parse_env_cfg(args.task, num_envs=num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    q_net = QNetwork(state_dim=cfg.state_dim, action_dim=cfg.num_actions, hidden_dim=cfg.hidden_dim).to(device)
    q_net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    q_net.eval()
    print(f"[DIAG] loaded checkpoint: {args.checkpoint}")
    return env, unwrapped, q_net


def classify_outcome(unwrapped, i, steps, max_steps):
    rf = getattr(unwrapped, "_last_reached_flag", None)
    gc = getattr(unwrapped, "_last_ground_crash", None)
    ac = getattr(unwrapped, "_last_any_contact", None)
    oob = getattr(unwrapped, "_last_out_of_bounds", None)
    if rf is not None and rf[i].item():
        return "reached"
    if gc is not None and gc[i].item():
        return "ground_crash"
    if ac is not None and ac[i].item():
        return "contact_crash"
    if oob is not None and oob[i].item():
        return "out_of_bounds"
    if steps >= max_steps - 1:
        return "timed_out"
    return "unclear"


# ══════════════════════════════════════════════════════════
# BATCH MODE
# ══════════════════════════════════════════════════════════
def run_batch():
    env, unwrapped, q_net = load_env_and_net(args.num_envs)
    obs_dict, _ = env.reset()
    state = obs_dict["dqn_state"]

    n = args.num_envs
    cum_reward = torch.zeros(n, device=device)
    ep_step = torch.zeros(n, device=device)
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
                steps_i = int(ep_step[i].item())
                outcome = classify_outcome(unwrapped, i, steps_i, args.max_steps_per_episode)
                episodes.append({
                    "episode_id": len(episodes),
                    "outcome": outcome,
                    "steps": steps_i,
                    "total_reward": round(cum_reward[i].item(), 3),
                })
            cum_reward[done_idx] = 0
            ep_step[done_idx] = 0

        state = next_state

    # ── Summary table ──
    print(f"\n[DIAG] collected {len(episodes)} episodes\n")
    header = f"{'episode_id':>10} {'outcome':<14} {'steps':>6} {'total_reward':>13}"
    print(header)
    print("-" * len(header))
    for e in episodes:
        print(f"{e['episode_id']:>10} {e['outcome']:<14} {e['steps']:>6} {e['total_reward']:>13.2f}")

    outcome_counts = {}
    for e in episodes:
        outcome_counts[e["outcome"]] = outcome_counts.get(e["outcome"], 0) + 1
    total = len(episodes)
    print(f"\n[DIAG] Outcome distribution (n={total}):")
    for k in ["reached", "ground_crash", "contact_crash", "out_of_bounds", "timed_out", "unclear"]:
        v = outcome_counts.get(k, 0)
        print(f"  {k:<14}: {v:>4} ({v/total*100:>5.1f}%)")

    mean_reward = sum(e["total_reward"] for e in episodes) / total
    mean_steps = sum(e["steps"] for e in episodes) / total
    print(f"\n[DIAG] mean_reward={mean_reward:.2f}  mean_episode_length={mean_steps:.1f}")

    # ── CSV export ──
    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode_id", "outcome", "steps", "total_reward"])
            writer.writeheader()
            writer.writerows(episodes)
        print(f"[DIAG] CSV written to {args.csv_out}")

    # ── TensorBoard append ──
    if args.tb_log_dir:
        from torch.utils.tensorboard import SummaryWriter
        step = args.tb_step if args.tb_step is not None else int(time.time())
        tb_writer = SummaryWriter(log_dir=args.tb_log_dir)
        tb_writer.add_scalar("Eval/mean_reward", mean_reward, step)
        tb_writer.add_scalar("Eval/mean_episode_length", mean_steps, step)
        tb_writer.add_scalar("Eval/success_rate", outcome_counts.get("reached", 0) / total * 100, step)
        for k in ["reached", "ground_crash", "contact_crash", "out_of_bounds", "timed_out"]:
            pct = outcome_counts.get(k, 0) / total * 100
            tb_writer.add_scalar(f"termination/{k}", pct, step)
        tb_writer.close()
        print(f"[DIAG] Appended to tensorboard: {args.tb_log_dir} (step={step})")


# ══════════════════════════════════════════════════════════
# TRACE MODE
# ══════════════════════════════════════════════════════════
def run_trace():
    env, unwrapped, q_net = load_env_and_net(1)
    obs_dict, _ = env.reset()
    state = obs_dict["dqn_state"]

    header = (f"{'step':>5} {'d_front':>8} {'dx':>7} {'dy':>7} {'yaw_rate':>9} "
              + " ".join(f"{'Q'+str(i)+'('+ACTION_NAMES[i]+')':>14}" for i in range(cfg.num_actions))
              + f" {'chosen':>10} {'switched':>9}")
    print(header)
    print("-" * len(header))

    prev_action = None
    for step in range(args.max_steps_per_episode):
        with torch.no_grad():
            q_values = q_net(state)
            action_idx = q_values.argmax(dim=-1)
        chosen = action_idx[0].item()
        switched = prev_action is not None and chosen != prev_action
        prev_action = chosen
        q_vals = q_values[0].tolist()

        # Real physical yaw angular velocity (body frame) -- used to check
        # whether periodic action switching tracks the drone's actual
        # attitude oscillation (rather than reward-logic confusion).
        real_yaw_rate = unwrapped._robot.data.root_ang_vel_b[0, 2].item()

        row = (f"{step:>5} {state[0,0].item():>8.2f} {state[0,3].item():>7.2f} {state[0,4].item():>7.2f} "
               f"{real_yaw_rate:>9.3f} "
               + " ".join(f"{q:>14.3f}" for q in q_vals)
               + f" {ACTION_NAMES[chosen]:>10} {'YES' if switched else '':>9}")
        print(row)

        actions_continuous = discrete_action_to_continuous(action_idx, device)
        obs_dict, reward, terminated, truncated, info = env.step(actions_continuous)
        state = obs_dict["dqn_state"]

        if (terminated | truncated).any():
            steps_i = step + 1
            outcome = classify_outcome(unwrapped, 0, steps_i, args.max_steps_per_episode)
            print(f"\n[DIAG] episode ended at step {steps_i}, outcome={outcome}")
            break


if __name__ == "__main__":
    if args.mode == "batch":
        run_batch()
    else:
        run_trace()
    simulation_app.close()
