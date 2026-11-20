"""
Zero-shot (or post-fine-tuning) evaluation of a checkpoint in Isaac Lab.

    python isaac_nav/eval_isaac.py --checkpoint <s8_final.pt> --num_obstacles 8 --num_envs 256 --rounds 4 \
        --camera_count 1 --sensor_min_range 0.4 --obs_velocity 1 --sensor camera --headless

Run it once with --sensor camera and once with --sensor geometry: geometry feeds the policy the 2D env's
exact readings at the drone's Isaac pose, so the gap between the two runs is what the camera costs, and the
gap between geometry and the 2D evaluation is what Isaac's dynamics cost.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dqn_2d_standalone_oscillation"))
sys.path.insert(0, HERE)

from isaaclab.app import AppLauncher  # noqa: E402
from dqn_config import add_config_args, config_from_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--rounds", type=int, default=4, help="resets of all envs; episodes = num_envs * rounds")
parser.add_argument("--num_obstacles", type=int, default=8)
parser.add_argument("--sensor", choices=("camera", "geometry"), default="camera")
parser.add_argument("--seed", type=int, default=777)
add_config_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
cfg = config_from_args(args)

app = AppLauncher(args).app

import torch  # noqa: E402

from dqn_action_space import QNetwork  # noqa: E402
from nav_env import make_env  # noqa: E402
from eval_core import evaluate  # noqa: E402

def robot_extent():
    # How far the robot's parts reach horizontally from its center (the 2D crash rule assumes 0.2 m).
    d = env.env._robot.data
    rel = d.body_pos_w[0, :, :2] - d.root_pos_w[0, None, :2]
    print(f"[ROBOT] body origins: max horizontal reach {float(rel.norm(dim=-1).max()):.3f} m "
          f"over {rel.shape[0]} bodies {list(getattr(env.env._robot, 'body_names', []))}")
    try:
        from pxr import Usd, UsdGeom
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        root = d.root_pos_w[0].cpu() - env.env.scene.env_origins[0].cpu()
        for purpose in (UsdGeom.Tokens.default_, UsdGeom.Tokens.guide):
            box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [purpose]).ComputeWorldBound(prim).ComputeAlignedRange()
            lo, hi = box.GetMin(), box.GetMax()
            o = env.env.scene.env_origins[0].cpu()
            rx = max(abs(lo[0] - float(o[0]) - float(root[0])), abs(hi[0] - float(o[0]) - float(root[0])))
            ry = max(abs(lo[1] - float(o[1]) - float(root[1])), abs(hi[1] - float(o[1]) - float(root[1])))
            print(f"[ROBOT] bounding box ({purpose}): x half-extent {rx:.3f} m, y half-extent {ry:.3f} m")
    except Exception as e:  # noqa: BLE001
        print(f"[ROBOT] bounding box unavailable: {e}")


torch.manual_seed(args.seed)
env = make_env(cfg, args.num_envs, args.num_obstacles, sensor=args.sensor, seed=args.seed)
q_net = QNetwork(state_dim=env.state_dim, action_dim=cfg.num_actions, hidden_dim=cfg.hidden_dim).to(env.device)
q_net.load_state_dict(torch.load(args.checkpoint, map_location=env.device))
q_net.eval()
print(f"[EVAL_ISAAC] {args.checkpoint}  sensor={args.sensor}  num_obstacles={args.num_obstacles}  "
      f"envs={args.num_envs} x rounds={args.rounds}  seed={args.seed}  state_dim={env.state_dim}")
try:
    try:
        robot_extent()
    except Exception as e:  # noqa: BLE001
        print(f"[ROBOT] extent unavailable: {e}")
    evaluate(env, q_net, cfg, rounds=args.rounds)
finally:
    env.close()
    app.close()
