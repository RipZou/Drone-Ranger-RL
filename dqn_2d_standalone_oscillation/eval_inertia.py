"""
Evaluate one checkpoint under a grid of fixed inertia settings (InertialDroneNav2DEnv in dqn_2d_env.py).

Each configuration fixes tau (velocity time constant, s), accel (acceleration cap, m/s^2) and the action
delay (decision steps of 0.1 s) for every episode, and is scored on the same layouts as eval_2d.py with the
same --seed, --num_obstacles and --num_episodes, first episode of each env only. The first row is always
the instant kinematics of the training env (no inertia, one substep) and matches eval_2d.py.

A checkpoint trained with --obs_velocity 1 needs --obs_velocity 1 here too; the other inertia_* options
are overridden per configuration.

Gap group (approximate re-implementation of the definition in EXPERIMENT_SUMMARY.md): an episode counts
when at some step two active obstacles lie more than 90 degrees apart in bearing from the drone and its
distances to their two surfaces sum to under 1.0 m. "gap 0.4-0.6" uses the narrowest such sum met.
"""
import argparse
import csv
import dataclasses
import itertools
import math
import time

import torch

from dqn_config import DQNConfig, add_config_args, config_from_args
from dqn_action_space import discrete_action_to_continuous, QNetwork
from dqn_2d_env import NUM_CYLINDERS, OUTCOMES, add_env_args, env_from_args
from obstacle_geometry import _nearest_surface

DIST_BUCKETS = ((0.0, 3.0), (3.0, 6.0), (6.0, math.inf))
GAP_WIDTH = 1.0
GAP_ANGLE = math.pi / 2
NARROW = (0.4, 0.6)


def gap_width(env):
    """Narrowest gap the drone sits in at the end of the latest step, per env (inf if none)."""
    s = env.pre_reset_state
    xy = s["obstacle_xy"]
    dist, bearing = _nearest_surface(s["pos"], xy[:, :NUM_CYLINDERS], env._circle_r,
                                     xy[:, NUM_CYLINDERS:], env._box_half)
    dist = torch.where(s["obstacle_active"], dist, torch.full_like(dist, math.inf))
    pair_sum = dist[:, :, None] + dist[:, None, :]
    diff = bearing[:, :, None] - bearing[:, None, :]
    apart = torch.atan2(torch.sin(diff), torch.cos(diff)).abs() > GAP_ANGLE
    pair_sum = torch.where(apart, pair_sum, torch.full_like(pair_sum, math.inf))
    return pair_sum.flatten(1).min(dim=1).values


def evaluate(q_net, args, cfg, device):
    env = env_from_args(args, args.num_episodes, cfg, device)
    torch.manual_seed(args.seed)  # same layouts as eval_2d.py with the same --seed
    obs_dict, _ = env.reset()
    state = obs_dict["dqn_state"]
    n = args.num_episodes
    init_dist = torch.norm(env.target - env.pos, dim=-1)
    outcome = [None] * n
    finished = torch.zeros(n, dtype=torch.bool)
    steps = torch.zeros(n, dtype=torch.long)
    min_gap = torch.full((n,), math.inf)
    impact_speed = torch.zeros(n)
    for t in range(env.max_steps):
        with torch.no_grad():
            action_idx = q_net(state).argmax(dim=-1)
        obs_dict, _, terminated, truncated, _ = env.step(discrete_action_to_continuous(action_idx, device, cfg))
        state = obs_dict["dqn_state"]
        if args.num_obstacles > 0:
            min_gap = torch.where(finished, min_gap, torch.minimum(min_gap, gap_width(env).cpu()))
        done = (terminated | truncated).cpu()
        if done.any():
            outs = env.last_outcomes()
            speed_now = env.pre_reset_state.get("speed", torch.full((n,), float("nan"))).cpu()
            for i in (done & ~finished).nonzero(as_tuple=True)[0].tolist():
                outcome[i] = outs[i]
                steps[i] = t + 1
                impact_speed[i] = speed_now[i]
            finished |= done
        if finished.all():
            break

    reached = torch.tensor([o == "reached" for o in outcome])
    crash = torch.tensor([o == "contact_crash" for o in outcome])
    row = {k: sum(o == k for o in outcome) for k in OUTCOMES}
    row["reached_pct"] = 100.0 * row["reached"] / n
    row["mean_steps"] = steps[reached].float().mean().item() if reached.any() else float("nan")
    row["crash_speed"] = impact_speed[crash].mean().item() if crash.any() else float("nan")
    in_gap = min_gap < GAP_WIDTH
    narrow = (min_gap >= NARROW[0]) & (min_gap < NARROW[1])
    row["gap_n"], row["gap_reached"] = int(in_gap.sum()), int((reached & in_gap).sum())
    row["narrow_n"], row["narrow_reached"] = int(narrow.sum()), int((reached & narrow).sum())
    b = init_dist >= DIST_BUCKETS[-1][0]
    row["far_n"], row["far_reached"] = int(b.sum()), int((reached & b).sum())
    return row


def pct(a, b):
    return f"{100.0 * a / b:6.2f}" if b else "    - "


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--taus", type=str, default="0,0.1,0.25,0.4", help="velocity time constants, s")
    parser.add_argument("--accels", type=str, default="6.0", help="acceleration caps, m/s^2; 0 = none")
    parser.add_argument("--delays", type=str, default="0,1,2", help="action delays, in 0.1 s decision steps")
    parser.add_argument("--substeps", type=int, default=5, help="integration substeps per 0.1 s decision")
    parser.add_argument("--csv", type=str, default=None, help="write one row per configuration here")
    add_env_args(parser)
    add_config_args(parser)
    args = parser.parse_args()
    base_cfg: DQNConfig = config_from_args(args)
    device = "cpu"

    def fixed(tau, accel, delay, substeps):
        return dataclasses.replace(base_cfg, inertia_tau_min=tau, inertia_tau_max=tau,
                                   inertia_accel_min=accel, inertia_accel_max=accel,
                                   inertia_delay_min=delay, inertia_delay_max=delay, inertia_substeps=substeps)

    ref_cfg = fixed(0.0, 0.0, 0, 1)
    probe = env_from_args(args, 1, ref_cfg, device)
    q_net = QNetwork(state_dim=probe.state_dim, action_dim=base_cfg.num_actions, hidden_dim=base_cfg.hidden_dim)
    q_net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    q_net.eval()

    floats = lambda s: [float(x) for x in s.split(",") if x.strip()]
    configs = [(0.0, 0.0, 0, 1)]
    configs += [(t, a, int(d), args.substeps)
                for t, a, d in itertools.product(floats(args.taus), floats(args.accels), floats(args.delays))]

    print(f"[EVAL_INERTIA] {args.checkpoint}  episodes={args.num_episodes}  num_obstacles={args.num_obstacles}  "
          f"seed={args.seed}  substeps={args.substeps}  obs_velocity={base_cfg.obs_velocity}")
    head = (f"{'tau':>5} {'accel':>5} {'delay':>5} | {'reached%':>8} {'crash':>5} {'timeout':>7} {'oob':>4} | "
            f"{'gap<1m%':>8} {'(n)':>6} | {'gap.4-.6%':>9} {'(n)':>5} | {'6m+%':>6} | {'steps':>5} {'v_hit':>5}")
    print(head)
    print("-" * len(head))
    rows = []
    for i, (tau, accel, delay, sub) in enumerate(configs):
        t0 = time.time()
        r = evaluate(q_net, args, fixed(tau, accel, delay, sub), device)
        r.update(tau=tau, accel=accel, delay=delay, substeps=sub, seconds=round(time.time() - t0, 1))
        rows.append(r)
        print(f"{tau:>5g} {accel if accel > 0 else math.inf:>5g} {delay:>5d} | {r['reached_pct']:8.2f} "
              f"{r['contact_crash']:>5d} {r['timed_out']:>7d} {r['out_of_bounds']:>4d} | "
              f"{pct(r['gap_reached'], r['gap_n']):>8} {r['gap_n']:>6d} | {pct(r['narrow_reached'], r['narrow_n']):>9} "
              f"{r['narrow_n']:>5d} | {pct(r['far_reached'], r['far_n']):>6} | {r['mean_steps']:5.0f} "
              f"{r['crash_speed']:5.2f} {'ref' if i == 0 else ''}  ({r['seconds']} s)", flush=True)
    print("crash / timeout / oob are episode counts out of", args.num_episodes,
          "| v_hit = mean speed at contact, m/s | first row = training-env kinematics, matches eval_2d.py")

    if args.csv:
        keys = ["tau", "accel", "delay", "substeps", "reached_pct", *OUTCOMES, "gap_n", "gap_reached",
                "narrow_n", "narrow_reached", "far_n", "far_reached", "mean_steps", "crash_speed", "seconds"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
