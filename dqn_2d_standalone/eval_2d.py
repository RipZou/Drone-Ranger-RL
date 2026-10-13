"""Greedy (epsilon=0) evaluation. Each env is scored on its FIRST episode only.
Same --seed => identical start/target/obstacle layouts for runs with the same num_obstacles."""
import argparse, math
import torch
from dqn_config import DQNConfig, add_config_args, config_from_args
from dqn_action_space import discrete_action_to_continuous, QNetwork
from dqn_2d_env import DroneNav2DEnv

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=2000)
parser.add_argument("--seed", type=int, default=0)
add_config_args(parser)
args = parser.parse_args()
cfg: DQNConfig = config_from_args(args)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)
N = args.num_episodes
env = DroneNav2DEnv(num_envs=N, cfg=cfg, device=device)
q = QNetwork(state_dim=cfg.state_dim, action_dim=cfg.num_actions, hidden_dim=cfg.hidden_dim).to(device)
q.load_state_dict(torch.load(args.checkpoint, map_location=device)); q.eval()

obs, _ = env.reset(); state = obs["dqn_state"]
init_dist = torch.norm(env.target - env.pos, dim=-1).clone()
outcome = torch.full((N,), -1, dtype=torch.long, device=device)   # 0 reached 1 crashed 2 oob 3 timeout
steps = torch.zeros(N, dtype=torch.long, device=device)
for t in range(env.max_steps + 5):
    with torch.no_grad():
        a = q(state).argmax(-1)
    obs, r, term, trunc, _ = env.step(discrete_action_to_continuous(a, device))
    state = obs["dqn_state"]
    done = (term | trunc) & (outcome < 0)
    o = torch.full((N,), 3, dtype=torch.long, device=device)
    o[env._last_out_of_bounds] = 2
    o[env._last_any_contact] = 1
    o[env._last_reached_flag] = 0
    outcome[done] = o[done]; steps[done] = t + 1
    if (outcome >= 0).all():
        break

names = ["reached", "crashed", "out_of_bounds", "timed_out"]
print(f"[EVAL] {args.checkpoint}  episodes={N}  num_obstacles={cfg.num_obstacles}  seed={args.seed}")
for i, n in enumerate(names):
    p = (outcome == i).float().mean().item()
    se = math.sqrt(p * (1 - p) / N)
    print(f"  {n:<14} {p*100:6.1f}%  (±{1.96*se*100:.1f})")
rm = outcome == 0
if rm.any():
    print(f"  mean steps to reach: {steps[rm].float().mean().item():.0f}")
for lo, hi in [(0, 3), (3, 6), (6, 99)]:
    m = (init_dist >= lo) & (init_dist < hi)
    if m.any():
        print(f"  reached | start-target dist {lo}-{hi if hi<99 else 'inf'} m: {(outcome[m]==0).float().mean().item()*100:5.1f}%  (n={int(m.sum())})")
