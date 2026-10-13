"""
Step-by-step trace for the 2D environment -- prints chosen action,
position delta, distance, heading, and the full reward breakdown for
every step of one episode. Used to check whether a policy that appears
"stuck" in trajectory plots is farming rew_yaw_aligned by standing still
(or oscillating in place) rather than genuinely committing to reach the
target.
"""
import argparse

import torch

from dqn_config import DQNConfig, add_config_args, config_from_args
from dqn_action_space import discrete_action_to_continuous, QNetwork, ACTION_NAMES
from dqn_2d_env import DroneNav2DEnv

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--max_steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=None, help="torch manual seed, for reproducing a specific start/target")
add_config_args(parser)
args = parser.parse_args()
cfg: DQNConfig = config_from_args(args)

if args.seed is not None:
    torch.manual_seed(args.seed)

device = "cpu"
env = DroneNav2DEnv(num_envs=1, cfg=cfg, device=device)
q_net = QNetwork(state_dim=cfg.state_dim, action_dim=cfg.num_actions, hidden_dim=cfg.hidden_dim).to(device)
q_net.load_state_dict(torch.load(args.checkpoint, map_location=device))
q_net.eval()
print(f"[TRACE] loaded checkpoint: {args.checkpoint}")

obs_dict, _ = env.reset()
state = obs_dict["dqn_state"]
print(f"[TRACE] start={env.pos[0].tolist()}  target={env.target[0].tolist()}")
print()

# [obstacles-patch]
header = (f"{'step':>5} {'action':>10} {'pos_x':>7} {'pos_y':>7} {'d(pos)':>7} "
          f"{'dist':>7} {'d_obs':>7} {'yaw_cos':>8} {'vx':>7} {'r_progress':>10} {'r_total':>8}")
if env.num_obstacles > 0:
    print("[TRACE] obstacles (world x, y; r=%.2f):" % cfg.obstacle_radius,
          [(round(o[0], 2), round(o[1], 2)) for o in env.obs_pos[0].tolist() if abs(o[0]) < 1e3])
    print("[TRACE] d_obs = distance from the drone to the NEAREST obstacle SURFACE (crash when < %.2f)" % cfg.drone_radius)
    print()
print(header)
print("-" * len(header))

prev_pos = env.pos[0].clone()
for step in range(args.max_steps):
    with torch.no_grad():
        action_idx = q_net(state).argmax(dim=-1)
    actions_continuous = discrete_action_to_continuous(action_idx, device)

    obs_dict, reward, terminated, truncated, info = env.step(actions_continuous)
    state = obs_dict["dqn_state"]

    # step() auto-resets finished envs. Swap in the pre-reset snapshot so every
    # number below (including the LAST row of an episode) describes the real
    # post-move state, then swap the live state back.
    live = (env.pos, env.yaw, env.target, env.obs_pos)
    env.pos, env.yaw, env.target, env.obs_pos = env._pre_reset

    rel = env.target - env.pos
    dist_now = torch.norm(rel, dim=-1)
    if env.num_obstacles > 0:
        d_obs = (torch.norm(env.obs_pos - env.pos[:, None, :], dim=-1) - cfg.obstacle_radius).min().item()
    else:
        d_obs = float("nan")
    _, breakdown = env._compute_reward(dist_now, actions_continuous,
                                        env._last_reached_flag, env._last_any_contact,
                                        return_breakdown=True)  # vx_signed re-derived internally, fine for tracing

    new_pos = env.pos[0].clone()
    pos_delta = torch.norm(new_pos - prev_pos).item()
    prev_pos = new_pos

    print(f"{step:>5} {ACTION_NAMES[action_idx[0].item()]:>10} "
          f"{env.pos[0,0].item():>7.3f} {env.pos[0,1].item():>7.3f} {pos_delta:>7.4f} "
          f"{dist_now[0].item():>7.3f} {d_obs:>7.3f} {breakdown['yaw_cos'][0].item():>8.3f} "
          f"{breakdown['vx_signed'][0].item():>7.3f} {breakdown['rew_progress'][0].item():>10.4f} "
          f"{reward[0].item():>8.4f}")
    env.pos, env.yaw, env.target, env.obs_pos = live

    if (terminated | truncated)[0].item():
        if env._last_reached_flag[0].item():
            outcome = "reached"
        elif env._last_any_contact[0].item():
            outcome = "crashed"
        elif env._last_out_of_bounds[0].item():
            outcome = "out_of_bounds"
        else:
            outcome = "timed_out"
        print(f"\n[TRACE] episode ended at step {step}, outcome={outcome}")
        break
