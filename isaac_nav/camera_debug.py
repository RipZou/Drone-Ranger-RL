"""
Check the camera readings against the geometry readings on a few envs, with and without the airframe filter.

    python isaac_nav/camera_debug.py --camera_count 1 --sensor_min_range 0.4 --obs_velocity 1 --headless
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
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=15)
add_config_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
cfg = config_from_args(args)

app = AppLauncher(args).app

import torch  # noqa: E402

from depth_bins import depth_to_bins  # noqa: E402
from dqn_action_space import discrete_action_to_continuous  # noqa: E402
from nav_env import CAMERA_BINS, MAX_SENSE_RANGE, SENSOR_HALF_FOV, make_env  # noqa: E402

torch.set_printoptions(precision=2, linewidth=160, sci_mode=False)
env = make_env(cfg, args.num_envs, 8, sensor="camera", seed=1)
try:
    env.reset()
    hover = discrete_action_to_continuous(torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
                                          env.device, cfg)
    for _ in range(args.steps):
        env.step(hover)
    e = env.env
    pos, yaw, _ = e._pose()
    data = e._camera.data
    depth = data.output["distance_to_camera"]
    d = depth[..., 0] if depth.dim() == 4 else depth
    print(f"depth shape {tuple(depth.shape)}  finite {torch.isfinite(d).float().mean():.3f}  "
          f"min {d[torch.isfinite(d)].min():.3f}  max {d[torch.isfinite(d)].max():.3f}")
    cam_pos, cam_quat = e._camera_pose()
    drone_w = pos + e.scene.env_origins
    print(f"sensor pose  env 0: offset {(data.pos_w - drone_w)[0].tolist()}  quat_w_ros {data.quat_w_ros[0].tolist()}")
    print(f"robot-based  env 0: offset {(cam_pos - drone_w)[0].tolist()}  quat_w_ros {cam_quat[0].tolist()}"
          f"  drone yaw {float(yaw[0]):.3f}")
    common = (e._rays, cam_pos, cam_quat, drone_w, yaw, SENSOR_HALF_FOV, CAMERA_BINS, MAX_SENSE_RANGE,
              e.cfg.height_band, e.cfg.ground_z, cfg.sensor_min_range)
    raw = depth_to_bins(depth, *common, far_clip=e.cfg.camera_far_clip, self_radius=0.0)
    filt = depth_to_bins(depth, *common, far_clip=e.cfg.camera_far_clip, self_radius=e.cfg.camera_self_radius)
    geo = e.last_geometry_readings
    for i in range(min(6, env.num_envs)):
        print(f"env {i}: geometry {geo[i]}\n        camera   {raw[i]}\n        filtered {filt[i]}")
    for name, r in (("no filter", raw), ("filtered", filt)):
        diff = (r - geo).abs()
        print(f"{name:>9}: |diff| median {diff.median():.3f} p90 {diff.flatten().quantile(0.9):.3f}  "
              f"camera at min_range {(r <= cfg.sensor_min_range + 1e-4).float().mean():.2f} "
              f"vs geometry {(geo <= cfg.sensor_min_range + 1e-4).float().mean():.2f}  "
              f"camera at max {(r >= MAX_SENSE_RANGE - 1e-4).float().mean():.2f} "
              f"vs geometry {(geo >= MAX_SENSE_RANGE - 1e-4).float().mean():.2f}")
    try:
        import numpy as np
        from PIL import Image
        img = d[0].clone().cpu()
        img[~torch.isfinite(img)] = MAX_SENSE_RANGE
        img = (255 * (1 - img.clamp(0, MAX_SENSE_RANGE) / MAX_SENSE_RANGE)).byte().numpy()
        Image.fromarray(np.kron(img, np.ones((6, 6), dtype=np.uint8))).save("camera_env0.png")
        print("env 0 depth image saved to camera_env0.png (brighter = nearer)")
    except Exception as ex:  # noqa: BLE001
        print(f"image not saved: {ex}")
finally:
    env.close()
    app.close()
