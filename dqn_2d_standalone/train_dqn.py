"""
DQN training script -- pure-2D kinematic environment only.

This is a standalone, Isaac-Lab-free version of the DQN training loop,
extracted from the main Drone-Ranger-RL project's Isaac Lab-based
training script. It requires only torch (CPU is fine, no GPU needed) and
runs the same DQN algorithm, network, action space, and reward against
dqn_2d_env.py -- a pure-kinematic 2D navigation environment -- instead of
the full Isaac Lab simulation. This lets anyone (including on a Mac, no
Isaac Sim install required) reproduce and experiment with the DQN
algorithm/reward design in isolation from the simulator.

See the main repository's README for full background on why this 2D
environment exists (isolating the DQN/reward/network design from Isaac
Lab simulation details) and the diagnosed issues it uncovered.
"""
import argparse
import os
import random
import time
from collections import deque

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dqn_config import DQNConfig, add_config_args, config_from_args
from dqn_action_space import discrete_action_to_continuous, QNetwork
from dqn_2d_env import DroneNav2DEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_episodes", type=int, default=1500)
parser.add_argument("--max_steps_per_episode", type=int, default=100)
parser.add_argument("--resume_from", type=str, default=None, help="path to a .pt checkpoint to resume from")
parser.add_argument("--resume_optimizer_from", type=str, default=None, help="path to a .pt optimizer state to resume from (paired with --resume_from)")
parser.add_argument("--run_name", type=str, default=None, help="name for this run's checkpoint/tensorboard files; auto-generated from timestamp if omitted")
parser.add_argument("--output_dir", type=str, default=SCRIPT_DIR,
                     help="base directory for checkpoints/ and tensorboard/ subdirectories; "
                          "defaults to this script's own directory")
add_config_args(parser)
args = parser.parse_args()
cfg: DQNConfig = config_from_args(args)

checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════
# Experience replay
# ══════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (torch.stack(s), torch.tensor(a, device=device),
                torch.tensor(r, device=device, dtype=torch.float32),
                torch.stack(s_next), torch.tensor(done, device=device, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)


# ══════════════════════════════════════════════════════════
# Set up environment, network, optimizer
# ══════════════════════════════════════════════════════════
env = DroneNav2DEnv(num_envs=args.num_envs, cfg=cfg, device=device)

q_net = QNetwork(state_dim=cfg.state_dim, action_dim=cfg.num_actions, hidden_dim=cfg.hidden_dim).to(device)
q_target = QNetwork(state_dim=cfg.state_dim, action_dim=cfg.num_actions, hidden_dim=cfg.hidden_dim).to(device)
if args.resume_from:
    q_net.load_state_dict(torch.load(args.resume_from, map_location=device))
    print(f"[DQN] resumed weights from {args.resume_from}")

q_target.load_state_dict(q_net.state_dict())
optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.learning_rate)
if args.resume_optimizer_from:
    optimizer.load_state_dict(torch.load(args.resume_optimizer_from, map_location=device))
    print(f"[DQN] resumed optimizer state from {args.resume_optimizer_from}")
elif args.resume_from:
    print("[DQN] resumed weights but NOT optimizer state (no --resume_optimizer_from given) -- Adam's per-parameter momentum starts fresh this run")
replay_buffer = ReplayBuffer(cfg.replay_buffer_size)

epsilon = cfg.resume_epsilon if args.resume_from else cfg.epsilon_start
episode_returns = []

run_name = args.run_name or time.strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(tensorboard_dir, run_name)
writer = SummaryWriter(log_dir=log_dir)
print("=" * 70)
print("[DQN] Starting training (2D kinematic environment, standalone)")
print("=" * 70)
print(f"  device          : {device}")
print(f"  num_envs        : {args.num_envs}")
print(f"  num_episodes    : {args.num_episodes}")
print(f"  max_steps/ep    : {args.max_steps_per_episode}")
print(f"  hidden_dim      : {cfg.hidden_dim}")
print(f"  state_dim       : {cfg.state_dim}  (num_obstacles={cfg.num_obstacles}, k_nearest={cfg.k_nearest}, state_scale={cfg.state_scale})")
print(f"  resume_from     : {args.resume_from or '(from scratch)'}")
print(f"  checkpoints dir : {checkpoints_dir}")
print(f"  tensorboard log : {log_dir}")
print("=" * 70)
train_start_time = time.time()

obs_dict, _ = env.reset()
state = obs_dict["dqn_state"]

episode_return = torch.zeros(args.num_envs, device=device)

WINDOW_SIZE = 200
outcome_window = deque(maxlen=WINDOW_SIZE)
global_train_step = 0

for episode in range(args.num_episodes):
    for step in range(args.max_steps_per_episode):
        with torch.no_grad():
            q_values = q_net(state)
            greedy_action = q_values.argmax(dim=-1)
            random_action = torch.randint(0, cfg.num_actions, (args.num_envs,), device=device)
            explore_mask = torch.rand(args.num_envs, device=device) < epsilon
            action_idx = torch.where(explore_mask, random_action, greedy_action)

        actions_continuous = discrete_action_to_continuous(action_idx, device)

        obs_dict, reward, terminated, truncated, info = env.step(actions_continuous)
        next_state = obs_dict["dqn_state"]
        done = terminated | truncated

        for i in range(args.num_envs):
            replay_buffer.push(
                state[i].cpu(), action_idx[i].item(), reward[i].item(),
                next_state[i].cpu(), done[i].item()
            )

        if done.any():
            unwrapped = env.unwrapped
            rf = getattr(unwrapped, "_last_reached_flag", None)
            oob = getattr(unwrapped, "_last_out_of_bounds", None)
            cr = getattr(unwrapped, "_last_any_contact", None)  # [obstacles-patch]
            done_idx = done.nonzero(as_tuple=True)[0].tolist()
            for i in done_idx:
                if rf is not None and rf[i].item():
                    outcome_window.append("reached")
                elif cr is not None and cr[i].item():
                    outcome_window.append("crashed")
                elif oob is not None and oob[i].item():
                    outcome_window.append("out_of_bounds")
                else:
                    outcome_window.append("timeout")

        episode_return += reward
        state = next_state

        if len(replay_buffer) >= cfg.batch_size:
            s, a, r, s_next, d = replay_buffer.sample(cfg.batch_size)
            s, s_next = s.to(device), s_next.to(device)

            q_pred = q_net(s).gather(1, a.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                q_next_max = q_target(s_next).max(dim=-1).values
                q_target_value = r + cfg.gamma * q_next_max * (1 - d)
            loss = F.mse_loss(q_pred, q_target_value)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_train_step += 1
            if global_train_step % 20 == 0:
                writer.add_scalar("Loss/td_error", loss.item(), global_train_step)
                writer.add_scalar("Loss/q_pred_mean", q_pred.mean().item(), global_train_step)
                writer.add_scalar("Loss/q_pred_max", q_pred.max().item(), global_train_step)
                writer.add_scalar("Loss/q_target_mean", q_target_value.mean().item(), global_train_step)

    mean_return = episode_return.mean().item()
    episode_returns.append(mean_return)
    episode_return.zero_()

    epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)

    if episode % cfg.target_update_every == 0:
        q_target.load_state_dict(q_net.state_dict())

    recent = episode_returns[-50:]
    ma50 = sum(recent) / len(recent)
    writer.add_scalar("Train/mean_reward", mean_return, episode)
    writer.add_scalar("Train/mean_reward_ma50", ma50, episode)
    writer.add_scalar("Train/epsilon", epsilon, episode)
    writer.add_scalar("Train/buffer_size", len(replay_buffer), episode)

    total_outcomes = len(outcome_window)
    if total_outcomes > 0:
        for k in ["reached", "crashed", "out_of_bounds", "timed_out"]:
            src_key = "timeout" if k == "timed_out" else k
            pct = sum(1 for o in outcome_window if o == src_key) / total_outcomes * 100
            writer.add_scalar(f"termination/{k}", pct, episode)

    if episode % 10 == 0:
        elapsed = time.time() - train_start_time
        eps_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0
        remaining = (args.num_episodes - episode - 1) / eps_per_sec if eps_per_sec > 0 else 0
        r_pct = sum(1 for o in outcome_window if o == "reached") / total_outcomes * 100 if total_outcomes else 0
        c_pct = sum(1 for o in outcome_window if o == "crashed") / total_outcomes * 100 if total_outcomes else 0
        o_pct = sum(1 for o in outcome_window if o == "out_of_bounds") / total_outcomes * 100 if total_outcomes else 0
        t_pct = sum(1 for o in outcome_window if o == "timeout") / total_outcomes * 100 if total_outcomes else 0
        print(f"[DQN] episode {episode:>4}/{args.num_episodes}  "
              f"return={mean_return:>7.2f}  MA50={ma50:>7.2f}  epsilon={epsilon:.3f}  "
              f"buffer={len(replay_buffer):>5}  elapsed={elapsed/60:>5.1f}min  ETA={remaining/60:>5.1f}min")
        print(f"       [last {total_outcomes} terminations] reached={r_pct:>5.1f}%  "
              f"crashed={c_pct:>5.1f}%  out_of_bounds={o_pct:>5.1f}%  timeout={t_pct:>5.1f}%")

    if episode % 100 == 0 and episode > 0:
        ckpt_path = os.path.join(checkpoints_dir, f"{run_name}_ep{episode}.pt")
        torch.save(q_net.state_dict(), ckpt_path)
        print("-" * 70)
        print(f"[DQN] checkpoint saved: {ckpt_path}")
        print("-" * 70)

print("=" * 70)
final_ckpt = os.path.join(checkpoints_dir, f"{run_name}_final.pt")
final_opt = os.path.join(checkpoints_dir, f"{run_name}_optimizer_final.pt")
torch.save(q_net.state_dict(), final_ckpt)
torch.save(optimizer.state_dict(), final_opt)
total_elapsed = time.time() - train_start_time
print(f"[DQN] Training complete. Final checkpoint saved: {final_ckpt}")
print(f"[DQN] Optimizer state saved: {final_opt}")
print(f"[DQN] Total time: {total_elapsed/60:.1f} minutes")
print(f"[DQN] tensorboard: tensorboard --logdir {tensorboard_dir}")
print("=" * 70)
