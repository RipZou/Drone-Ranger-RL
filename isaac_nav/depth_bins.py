"""
Depth image -> the 2D policy's distance readings. Pure torch, no Isaac imports, so it can be tested anywhere.

The 2D env (ground_truth mode, camera_count 1) reads, for each of CAMERA_BINS equal angular sectors of the
camera's 73 deg field of view, the horizontal distance from the drone's center to the nearest obstacle
surface inside that sector (obstacle_geometry.obstacle_sector_distances). Its obstacles are vertical
cylinders that span the flight altitude.

This computes the same quantity from a depth camera: every pixel's hit point is placed in the world, points
on the ground or outside a height band around the drone are dropped, and each remaining point counts toward
the sector its bearing from the drone's center falls in, with its horizontal distance from that center.
"""
import math

import torch


def quat_rotate(q, v):
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4) in (w, x, y, z) order, as Isaac Lab stores them."""
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * torch.cross(u.expand_as(v), v, dim=-1)
    return v + w * t + torch.cross(u.expand_as(v), t, dim=-1)


def quat_mul(a, b):
    """Hamilton product a * b of (w, x, y, z) quaternions."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def pixel_rays(intrinsics, height, width):
    """Unit ray directions (N, H, W, 3) in the camera's ROS frame (x right, y down, z forward) through
    each pixel center, from (N, 3, 3) pinhole intrinsics."""
    dev = intrinsics.device
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    cx, cy = intrinsics[:, 0, 2], intrinsics[:, 1, 2]
    u = torch.arange(width, device=dev, dtype=torch.float32) + 0.5
    v = torch.arange(height, device=dev, dtype=torch.float32) + 0.5
    x = ((u[None, :] - cx[:, None]) / fx[:, None])[:, None, :].expand(-1, height, -1)
    y = ((v[None, :] - cy[:, None]) / fy[:, None])[:, :, None].expand(-1, -1, width)
    d = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    return d / d.norm(dim=-1, keepdim=True)


def depth_to_bins(distance_to_camera, rays, cam_pos_w, cam_quat_w_ros, drone_pos_w, drone_yaw,
                  half_fov, num_bins, max_range, height_band, ground_z, min_range=0.0, far_clip=None,
                  self_radius=0.0):
    """(N, num_bins) readings ordered right to left, as GROUND_TRUTH sectors are.

    distance_to_camera (N, H, W) or (N, H, W, 1): range along each pixel's ray, as Isaac's
        "distance_to_camera" output gives it; inf, nan, 0 or >= far_clip mean no hit.
    rays (N, H, W, 3): pixel_rays() for this camera.
    cam_pos_w (N, 3), cam_quat_w_ros (N, 4): camera pose in the world, ROS convention.
    drone_pos_w (N, 3), drone_yaw (N,): the drone's center and heading.
    height_band: keep hit points within this many metres of the drone's height.
    ground_z: drop hit points at or below this height (the ground plane).
    self_radius: drop hit points closer than this (horizontally) to the drone's center: the airframe itself.
    A sector with no kept point reads max_range; every reading is at least min_range.
    """
    depth = distance_to_camera
    if depth.dim() == 4:
        depth = depth[..., 0]
    n = depth.shape[0]
    far = max_range if far_clip is None else far_clip
    hit = torch.isfinite(depth) & (depth > 0) & (depth < far - 1e-4)
    depth = torch.where(hit, depth, torch.zeros_like(depth))
    p_cam = rays * depth[..., None]
    p_w = cam_pos_w[:, None, None, :] + quat_rotate(cam_quat_w_ros[:, None, None, :], p_cam)
    rel = p_w[..., :2] - drone_pos_w[:, None, None, :2]
    dist = rel.norm(dim=-1)
    bearing = torch.atan2(rel[..., 1], rel[..., 0]) - drone_yaw[:, None, None]
    bearing = torch.atan2(torch.sin(bearing), torch.cos(bearing))
    width = 2.0 * half_fov / num_bins
    idx = torch.floor((bearing + half_fov) / width).long()
    keep = (hit & ((p_w[..., 2] - drone_pos_w[:, None, None, 2]).abs() <= height_band)
            & (p_w[..., 2] > ground_z) & (idx >= 0) & (idx < num_bins) & (dist >= self_radius))
    vals = torch.where(keep, dist, torch.full_like(dist, math.inf)).reshape(n, -1)
    out = torch.full((n, num_bins), math.inf, device=depth.device)
    out.scatter_reduce_(1, idx.clamp(0, num_bins - 1).reshape(n, -1), vals, reduce="amin", include_self=True)
    out = out.clamp(max=max_range)
    if min_range > 0:
        out = out.clamp(min=min_range)
    return out
