"""
Greedy evaluation over any env with the 2D env's interface: the first episode of every env, repeated for
`rounds` resets. Pure torch, so it runs on the 2D env too (used to test it).
"""
import math
import os
import sys
from collections import Counter

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dqn_2d_standalone_oscillation"))

from dqn_action_space import discrete_action_to_continuous  # noqa: E402
from dqn_2d_env import OUTCOMES  # noqa: E402
from eval_inertia import GAP_WIDTH, NARROW, gap_width  # noqa: E402


def evaluate(env, q_net, cfg, rounds=1, log=print):
    n = env.num_envs
    totals = Counter()
    crash_kinds = Counter()
    gap_n = gap_ok = narrow_n = narrow_ok = 0
    reach_steps = []
    hit_speed = []
    reading_err = []
    contact_surface = []
    for rnd in range(rounds):
        obs, _ = env.reset()
        state = obs["dqn_state"]
        outcome = [None] * n
        finished = torch.zeros(n, dtype=torch.bool)
        min_gap = torch.full((n,), math.inf)
        for t in range(env.max_steps):
            with torch.no_grad():
                a = q_net(state).argmax(dim=-1)
            _, _, terminated, truncated, info = env.step(discrete_action_to_continuous(a, env.device, cfg))
            state = info["reset_obs"]["dqn_state"]
            pre = env.pre_reset_state
            if getattr(env, "num_obstacles", 0) > 0:
                min_gap = torch.where(finished, min_gap, torch.minimum(min_gap, gap_width(env).cpu()))
            inner = getattr(env, "env", None)
            if inner is not None and getattr(inner, "last_camera_readings", None) is not None:
                live = (~finished).to(inner.last_camera_readings.device)
                diff = (inner.last_camera_readings - inner.last_geometry_readings).abs()[live]
                if diff.numel():
                    reading_err.append(diff.flatten().cpu())
            done = (terminated | truncated).cpu()
            if done.any():
                outs = env.last_outcomes()
                for i in (done & ~finished).nonzero(as_tuple=True)[0].tolist():
                    outcome[i] = outs[i]
                    if outs[i] == "reached":
                        reach_steps.append(t + 1)
                    if outs[i] == "contact_crash":
                        if "speed" in pre:
                            hit_speed.append(float(pre["speed"][i]))
                        kind = "obstacle (geometric)"
                        if "surface_dist" in pre and float(pre["surface_dist"][i]) >= 0.2:
                            if "height" in pre and not 0.3 <= float(pre["height"][i]) <= 2.5:
                                kind = "altitude"
                            elif "contact" in pre and bool(pre["contact"][i]):
                                kind = "physical contact only"
                                contact_surface.append(float(pre["surface_dist"][i]))
                            else:
                                kind = "tilt"
                        crash_kinds[kind] += 1
                finished |= done
            if finished.all():
                break
        for o in outcome:
            totals[o or "timed_out"] += 1
        reached = torch.tensor([o == "reached" for o in outcome])
        in_gap = min_gap < GAP_WIDTH
        narrow = (min_gap >= NARROW[0]) & (min_gap < NARROW[1])
        gap_n += int(in_gap.sum()); gap_ok += int((reached & in_gap).sum())
        narrow_n += int(narrow.sum()); narrow_ok += int((reached & narrow).sum())
        log(f"  round {rnd + 1}/{rounds}: reached {sum(o == 'reached' for o in outcome)}/{n}")

    total = sum(totals.values())
    pct = lambda a, b: f"{100.0 * a / b:6.2f}%" if b else "     -"
    log(f"episodes {total}")
    for k in OUTCOMES:
        log(f"  {k:<14} {pct(totals[k], total)}  ({totals[k]})")
    if crash_kinds:
        log("  crash kinds: " + ", ".join(f"{k} {v}" for k, v in crash_kinds.most_common()))
    if hit_speed:
        log(f"  mean speed at crash {sum(hit_speed) / len(hit_speed):.2f} m/s")
    if reach_steps:
        log(f"  mean steps to reach {sum(reach_steps) / len(reach_steps):.0f}")
    log(f"  gap < 1 m: {pct(gap_ok, gap_n)} of {gap_n} | gap 0.4-0.6 m: {pct(narrow_ok, narrow_n)} of {narrow_n}")
    if contact_surface:
        c = torch.tensor(contact_surface)
        q = torch.quantile(c, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0]))
        log("  physical-contact crashes, drone center to obstacle surface at the end of the step: "
            f"min {q[0]:.3f} p10 {q[1]:.3f} median {q[2]:.3f} p90 {q[3]:.3f} max {q[4]:.3f} m")
    if reading_err:
        e = torch.cat(reading_err)
        q = torch.quantile(e[:200000], torch.tensor([0.5, 0.9, 0.99]))
        log(f"  camera vs geometry readings |diff|: median {q[0]:.3f} m, p90 {q[1]:.3f} m, p99 {q[2]:.3f} m")
    return {"totals": dict(totals), "gap": (gap_ok, gap_n), "narrow": (narrow_ok, narrow_n)}
