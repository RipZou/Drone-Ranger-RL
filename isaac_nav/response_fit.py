"""Turn a recorded step-response trace into the numbers the 2D inertia model uses. Pure torch."""
import csv
import math

import torch

COLS = ("forward", "left", "yaw_rate", "height", "vx_cmd", "yaw_rate_cmd")


def _rise(x, target, dt):
    """Per-env (delay, time to 63%, time to 90%) in seconds for a signal rising from ~0 to target.
    x is (T, N). Times are from the start of x; nan where a level is never reached."""
    def first(level):
        hit = x >= level * target
        idx = torch.where(hit.any(dim=0), hit.float().argmax(dim=0).float(), torch.full((x.shape[1],), float("nan")))
        return idx * dt
    return first(0.05), first(0.632), first(0.9)


def summarize(trace, dt, phases, forward_speed, turn_rate, csv_path=None, out=print):
    t1, t2, t3 = phases
    fwd, yaw_rate, height = trace[..., 0], trace[..., 2], trace[..., 3]
    go, stop, turn = fwd[:t1], fwd[t1:t1 + t2], yaw_rate[t1 + t2:t1 + t2 + t3]

    d5, d63, d90 = _rise(go, forward_speed, dt)
    # Two-point fit of a delayed first-order response: t90 - t63 = tau * (ln 10 - 1), delay = t63 - tau.
    tau = (d90 - d63) / (math.log(10.0) - 1.0)
    delay = d63 - tau
    accel = ((go[1:] - go[:-1]) / dt).max(dim=0).values
    v0 = stop[0]
    still = stop < 0.05
    t_stop = torch.where(still.any(dim=0), still.float().argmax(dim=0).float() * dt, torch.full_like(v0, float("nan")))
    slide = (stop.clamp(min=0) * dt).sum(dim=0)
    y5, y63, y90 = _rise(turn, turn_rate, dt)
    h_err = (height - height[0]).abs().max(dim=0).values

    m = lambda v: float(torch.nanmean(v))
    out("Step response (mean over envs)")
    out(f"  forward 0 -> {forward_speed} m/s: delay to 5% {m(d5):.3f} s, 63% at {m(d63):.3f} s, "
        f"90% at {m(d90):.3f} s")
    out(f"    first-order fit: delay {m(delay):.3f} s, time constant tau {m(tau):.3f} s; peak accel {m(accel):.2f} m/s^2")
    out(f"  stop from {m(v0):.2f} m/s: below 0.05 m/s after {m(t_stop):.3f} s, slid {m(slide):.3f} m")
    out(f"  yaw rate 0 -> {turn_rate} rad/s: delay to 5% {m(y5):.3f} s, 63% at {m(y63):.3f} s, 90% at {m(y90):.3f} s")
    out(f"  height deviation over the script: max {m(h_err):.3f} m")
    tau_v, delay_v = m(tau), m(delay)
    ok_tau = 0.15 <= tau_v <= 0.4
    out(f"  2D training ranges: tau 0.15-0.4 s -> {'inside' if ok_tau else 'OUTSIDE'}; "
        f"delay {delay_v:.3f} s vs 0-0.2 s -> {'inside' if delay_v <= 0.2 else 'OUTSIDE'}")
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(("t",) + COLS)
            for i in range(trace.shape[0]):
                w.writerow([f"{i * dt:.3f}"] + [f"{float(v):.4f}" for v in trace[i, 0]])
        out(f"  env 0 trace written to {csv_path}")
    return {"tau": tau_v, "delay": delay_v, "accel": m(accel), "stop_time": m(t_stop), "slide": m(slide),
            "yaw_t63": m(y63)}
