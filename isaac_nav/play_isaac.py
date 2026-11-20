"""
Watch a checkpoint fly in the Isaac GUI, in real time.

    python isaac_nav/play_isaac.py --checkpoint runs_isaac/ft_cam/checkpoints/ft_cam_final.pt \
        --camera_count 1 --sensor_min_range 0.4 --obs_velocity 1

The view looks down on env 0's arena. The target is a red disc on the ground (the 0.75 m reach radius) and
the drone's path is drawn as a blue trail; both lie on the ground, below the camera's height band, so the
policy does not see them. Every finished episode prints its outcome.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dqn_2d_standalone_oscillation"))
sys.path.insert(0, HERE)

from isaaclab.app import AppLauncher  # noqa: E402
from dqn_config import add_config_args, config_from_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_obstacles", type=int, default=8)
parser.add_argument("--sensor", choices=("camera", "geometry"), default="camera")
parser.add_argument("--episodes", type=int, default=20, help="episodes of env 0 to show")
parser.add_argument("--speed", type=float, default=1.0, help="playback speed, 1 = real time")
parser.add_argument("--seed", type=int, default=0)
add_config_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
cfg = config_from_args(args)

app = AppLauncher(args).app

import torch  # noqa: E402

from dqn_action_space import QNetwork, discrete_action_to_continuous  # noqa: E402
from nav_env import make_env  # noqa: E402

torch.manual_seed(args.seed)
env = make_env(cfg, args.num_envs, args.num_obstacles, sensor=args.sensor, seed=args.seed)
e = env.env
q_net = QNetwork(state_dim=env.state_dim, action_dim=cfg.num_actions, hidden_dim=cfg.hidden_dim).to(env.device)
q_net.load_state_dict(torch.load(args.checkpoint, map_location=env.device))
q_net.eval()

origin = e.scene.env_origins[0].tolist()
try:
    e.sim.set_camera_view(eye=[origin[0], origin[1] - 8.0, 12.0], target=[origin[0], origin[1], 0.0])
except Exception as ex:  # noqa: BLE001
    print(f"[PLAY] could not set the view: {ex}")

markers = None
try:
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    markers = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/play", markers={
        "target": sim_utils.CylinderCfg(radius=0.75, height=0.02, visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.9, 0.15, 0.15))),
        "trail": sim_utils.SphereCfg(radius=0.04, visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.1, 0.4, 0.95))),
    }))
except Exception as ex:  # noqa: BLE001
    print(f"[PLAY] markers unavailable: {ex}")


def draw(trail):
    if markers is None:
        return
    t = e.book.target[0].tolist()
    pts = [[origin[0] + t[0], origin[1] + t[1], 0.01]] + trail
    idx = [0] + [1] * len(trail)
    markers.visualize(translations=torch.tensor(pts, device=env.device),
                      marker_indices=torch.tensor(idx, device=env.device))


obs, _ = env.reset()
state = obs["dqn_state"]
trail, shown, counts = [], 0, {}
print(f"[PLAY] {args.checkpoint}  sensor={args.sensor}  showing {args.episodes} episodes of env 0")
try:
    while app.is_running() and shown < args.episodes:
        t0 = time.time()
        with torch.no_grad():
            a = q_net(state).argmax(dim=-1)
        _, _, terminated, truncated, info = env.step(discrete_action_to_continuous(a, env.device, cfg))
        state = info["reset_obs"]["dqn_state"]
        done = bool((terminated | truncated)[0])
        if done:
            out = env.last_outcomes()[0]
            counts[out] = counts.get(out, 0) + 1
            shown += 1
            print(f"[PLAY] episode {shown}: {out}  ({len(trail) + 1} steps)  so far {counts}")
            trail = []
        else:
            p = (e._robot.data.root_pos_w[0]).tolist()
            trail.append([p[0], p[1], 0.02])
        draw(trail)
        time.sleep(max(0.0, 0.1 / args.speed - (time.time() - t0)))
finally:
    env.close()
    app.close()
