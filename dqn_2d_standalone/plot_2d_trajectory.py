"""
Simplest possible visualization for the 2D environment -- runs a trained
checkpoint for one episode and plots the drone's path (top-down view) with
matplotlib. No GUI/Isaac Sim dependency; pure numpy/matplotlib.
"""
import argparse

import matplotlib.pyplot as plt
import torch

from dqn_config import DQNConfig, add_config_args, config_from_args
from dqn_action_space import discrete_action_to_continuous, QNetwork
from dqn_2d_env import DroneNav2DEnv

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--num_episodes", type=int, default=4,
                     help="how many episodes to plot (each as a separate line)")
parser.add_argument("--out", type=str, default="trajectory.png")
add_config_args(parser)
args = parser.parse_args()
cfg: DQNConfig = config_from_args(args)

device = "cpu"
env = DroneNav2DEnv(num_envs=1, cfg=cfg, device=device)
q_net = QNetwork(state_dim=cfg.state_dim, action_dim=cfg.num_actions, hidden_dim=cfg.hidden_dim).to(device)
q_net.load_state_dict(torch.load(args.checkpoint, map_location=device))
q_net.eval()
print(f"[PLOT] loaded checkpoint: {args.checkpoint}")

fig, ax = plt.subplots(figsize=(7, 7))
colors = plt.cm.tab10.colors

# [obstacles-patch]
for ep in range(args.num_episodes):
    obs_dict, _ = env.reset()
    state = obs_dict["dqn_state"]
    target = env.target[0].clone()
    obstacles = [o for o in env.obs_pos[0].tolist() if abs(o[0]) < 1e3]  # skip parked ones

    xs = [env.pos[0, 0].item()]
    ys = [env.pos[0, 1].item()]
    outcome = "timed_out"

    for step in range(args.max_steps):
        with torch.no_grad():
            action_idx = q_net(state).argmax(dim=-1)
        actions_continuous = discrete_action_to_continuous(action_idx, device)
        obs_dict, reward, terminated, truncated, info = env.step(actions_continuous)
        state = obs_dict["dqn_state"]
        # step() auto-resets finished envs, so read the true final position from
        # the pre-reset snapshot instead of env.pos.
        final_pos = env._pre_reset[0][0]
        xs.append(final_pos[0].item())
        ys.append(final_pos[1].item())

        if env._last_reached_flag[0].item():
            outcome = "reached"
        elif env._last_any_contact[0].item():
            outcome = "crashed"
        elif env._last_out_of_bounds[0].item():
            outcome = "out_of_bounds"
        if (terminated | truncated)[0].item():
            break

    color = colors[ep % len(colors)]
    ax.plot(xs, ys, color=color, alpha=0.7, linewidth=1.5,
            label=f"ep{ep} ({outcome}, {len(xs)} steps)")
    ax.scatter([xs[0]], [ys[0]], color=color, marker="o", s=60, zorder=5)  # start
    ax.scatter([target[0].item()], [target[1].item()], color=color, marker="*", s=200, zorder=5)  # target
    for (ox, oy) in obstacles:  # obstacles: filled circle in the episode's colour
        ax.add_patch(plt.Circle((ox, oy), cfg.obstacle_radius, color=color, alpha=0.25, zorder=2))
    if outcome == "crashed":
        ax.scatter([xs[-1]], [ys[-1]], color=color, marker="x", s=120, zorder=6)
    print(f"[PLOT] episode {ep}: outcome={outcome}, steps={len(xs)}, "
          f"start=({xs[0]:.2f},{ys[0]:.2f}), target=({target[0]:.2f},{target[1]:.2f}), "
          f"obstacles={len(obstacles)}")

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("2D DQN trajectories (o=start, *=target, disc=obstacle, x=crash)")
ax.legend(fontsize=8, loc="best")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"[PLOT] saved to {args.out}")
