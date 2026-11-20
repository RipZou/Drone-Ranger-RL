"""
The baseline_stop_v7.8 DQN training loop as a function of any env with the 2D env's interface.

It mirrors dqn_2d_standalone_oscillation/train_dqn.py step for step (Double DQN, soft target update,
n-step returns, AdamW, the learning-rate taper, temporally-extended exploration, late-step replay
thinning that keeps stopped transitions), without the per-checkpoint validation subprocess: in Isaac that
would relaunch the simulator every 100 episodes, and in 2D it picked worse checkpoints than the last ones.
Checkpoints are saved every `save_every` episodes and at the end; evaluate the last few afterwards.
"""
import os
import random
import sys
import time
from collections import Counter, deque

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dqn_2d_standalone_oscillation"))

from dqn_config import DQNConfig  # noqa: E402
from dqn_action_space import NUM_TURNS, QNetwork, discrete_action_to_continuous  # noqa: E402
from dqn_2d_env import OUTCOMES  # noqa: E402
from exploration import HeldRandomExploration  # noqa: E402
from lr_schedule import scheduled_lr  # noqa: E402
from nstep import NStepAccumulator  # noqa: E402


class ReplayBuffer:
    """FIFO ring of preallocated CPU tensors, as in train_dqn.py (uniform sampling only)."""

    def __init__(self, capacity, state_dim, device):
        self.capacity, self.device = capacity, device
        self.s = torch.zeros(capacity, state_dim)
        self.a = torch.zeros(capacity, dtype=torch.long)
        self.r = torch.zeros(capacity)
        self.s_next = torch.zeros(capacity, state_dim)
        self.done = torch.zeros(capacity)
        self.discount = torch.zeros(capacity)
        self.pos = 0
        self.size = 0

    def push(self, s, a, r, s_next, done, discount=None):
        n = s.shape[0]
        first = max(n - self.capacity, 0)
        slots = (self.pos + first + torch.arange(n - first)) % self.capacity
        self.s[slots] = s[first:].cpu()
        self.a[slots] = a[first:].cpu()
        self.r[slots] = r[first:].cpu()
        self.s_next[slots] = s_next[first:].cpu()
        self.done[slots] = done[first:].cpu().float()
        if discount is not None:
            self.discount[slots] = discount[first:].cpu()
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        oldest = (self.pos - self.size) % self.capacity
        slots = (torch.tensor(random.sample(range(self.size), batch_size)) + oldest) % self.capacity
        d = self.device
        return (self.s[slots].to(d), self.a[slots].to(d), self.r[slots].to(d), self.s_next[slots].to(d),
                self.done[slots].to(d), self.discount[slots].to(d))

    def __len__(self):
        return self.size


def train(env, cfg: DQNConfig, num_episodes, output_dir, run_name, max_steps_per_episode=100,
          resume_from=None, resume_optimizer_from=None, save_every=100, writer=None, log=print):
    if cfg.per_alpha > 0:
        raise ValueError("prioritized replay is not supported here; baseline_stop_v7.8 does not use it")
    device = env.device
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    n_envs = env.num_envs

    q_net = QNetwork(state_dim=env.state_dim, action_dim=cfg.num_actions, hidden_dim=cfg.hidden_dim).to(device)
    q_target = QNetwork(state_dim=env.state_dim, action_dim=cfg.num_actions, hidden_dim=cfg.hidden_dim).to(device)
    if resume_from:
        q_net.load_state_dict(torch.load(resume_from, map_location=device))
        log(f"[DQN] resumed weights from {resume_from}")
    q_target.load_state_dict(q_net.state_dict())
    if cfg.weight_decay > 0:
        optimizer = torch.optim.AdamW(q_net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.learning_rate)
    if resume_optimizer_from:
        optimizer.load_state_dict(torch.load(resume_optimizer_from, map_location=device))
        for group in optimizer.param_groups:
            group["weight_decay"] = cfg.weight_decay
    replay = ReplayBuffer(cfg.replay_buffer_size, env.state_dim, device)
    nstep = NStepAccumulator(cfg.n_step, cfg.gamma, n_envs, env.state_dim, device) if cfg.n_step > 1 else None
    epsilon = cfg.resume_epsilon if resume_from else cfg.epsilon_start
    explorer = HeldRandomExploration(n_envs, cfg.num_actions, cfg.explore_hold_mu, cfg.explore_max_hold, device)

    obs_dict, _ = env.reset()
    state = obs_dict["dqn_state"]
    episode_return = torch.zeros(n_envs, device=device)
    outcome_window = deque(maxlen=200)
    returns = []
    train_step = 0
    t0 = time.time()

    for episode in range(num_episodes):
        if cfg.lr_final_fraction != 1.0:
            lr = scheduled_lr(cfg.learning_rate, cfg.lr_final_fraction, cfg.lr_decay_start, episode, num_episodes)
            for group in optimizer.param_groups:
                group["lr"] = lr
        for _ in range(max_steps_per_episode):
            with torch.no_grad():
                greedy = q_net(state).argmax(dim=-1)
                action_idx = explorer.select(greedy, epsilon)
            actions_continuous = discrete_action_to_continuous(action_idx, device, cfg)
            step_in_episode = env.step_count + 1
            obs_dict, reward, terminated, truncated, info = env.step(actions_continuous)
            next_state = obs_dict["dqn_state"]
            done = terminated | truncated
            terminal = terminated

            if nstep is None:
                keep = ((step_in_episode <= cfg.replay_full_steps) | terminal
                        | (torch.rand(n_envs, device=device) < cfg.replay_late_fraction))
                if cfg.replay_keep_stopped:
                    keep |= action_idx < NUM_TURNS
                replay.push(state[keep], action_idx[keep], reward[keep], next_state[keep], terminal[keep])
            else:
                env_i, s_n, a_n, r_n, s_next_n, term_n, disc_n, start_n = nstep.add(
                    state, action_idx, reward, next_state, terminal, done, step_in_episode)
                late = torch.rand(n_envs, device=device)[env_i] < cfg.replay_late_fraction
                keep = (start_n <= cfg.replay_full_steps) | term_n | late
                if cfg.replay_keep_stopped:
                    keep |= a_n < NUM_TURNS
                replay.push(s_n[keep], a_n[keep], r_n[keep], s_next_n[keep], term_n[keep], disc_n[keep])

            if done.any():
                outs = env.last_outcomes()
                for i in done.nonzero(as_tuple=True)[0].tolist():
                    outcome_window.append(outs[i])
            episode_return += reward
            state = info["reset_obs"]["dqn_state"]

            if len(replay) >= cfg.batch_size:
                s, a, r, s_next, d, disc = replay.sample(cfg.batch_size)
                q_pred = q_net(s).gather(1, a.unsqueeze(-1)).squeeze(-1)
                with torch.no_grad():
                    next_a = q_net(s_next).argmax(dim=-1, keepdim=True)
                    q_next = q_target(s_next).gather(1, next_a).squeeze(-1)
                    target = r + (cfg.gamma if nstep is None else disc) * q_next * (1 - d)
                loss = F.mse_loss(q_pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if cfg.target_tau > 0:
                    with torch.no_grad():
                        for p_t, p in zip(q_target.parameters(), q_net.parameters()):
                            p_t.lerp_(p, cfg.target_tau)
                train_step += 1
                if writer is not None and train_step % 20 == 0:
                    writer.add_scalar("Loss/td_error", loss.item(), train_step)
                    writer.add_scalar("Loss/q_pred_mean", q_pred.mean().item(), train_step)

        returns.append(episode_return.mean().item())
        episode_return.zero_()
        epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)
        if cfg.target_tau == 0 and episode % cfg.target_update_every == 0:
            q_target.load_state_dict(q_net.state_dict())

        counts = Counter(outcome_window)
        total = len(outcome_window)
        pct = {k: counts[k] / total * 100 if total else 0.0 for k in OUTCOMES}
        if writer is not None:
            writer.add_scalar("Train/mean_reward", returns[-1], episode)
            writer.add_scalar("Train/epsilon", epsilon, episode)
            writer.add_scalar("Train/learning_rate", optimizer.param_groups[0]["lr"], episode)
            for k in OUTCOMES:
                writer.add_scalar(f"termination/{k}", pct[k], episode)
        if episode % 10 == 0:
            el = time.time() - t0
            eta = (num_episodes - episode - 1) * el / (episode + 1)
            ma = sum(returns[-50:]) / len(returns[-50:])
            log(f"[DQN] episode {episode:>4}/{num_episodes}  return={returns[-1]:>7.2f}  MA50={ma:>7.2f}  "
                f"epsilon={epsilon:.3f}  buffer={len(replay):>6}  elapsed={el / 60:5.1f}min  ETA={eta / 60:5.1f}min")
            log(f"       [last {total} terminations] " + "  ".join(f"{k}={pct[k]:5.1f}%" for k in OUTCOMES))
        if save_every and episode % save_every == 0 and episode > 0:
            path = os.path.join(ckpt_dir, f"{run_name}_ep{episode}.pt")
            torch.save(q_net.state_dict(), path)
            log(f"[DQN] checkpoint saved: {path}")

    final = os.path.join(ckpt_dir, f"{run_name}_final.pt")
    torch.save(q_net.state_dict(), final)
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, f"{run_name}_optimizer_final.pt"))
    log(f"[DQN] done in {(time.time() - t0) / 60:.1f} min, final checkpoint {final}")
    return final
