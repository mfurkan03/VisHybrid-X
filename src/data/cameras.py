"""
cameras.py – surround camera construction and per-frame sensor processing.
"""

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.depth_camera import DepthCamera


# ============================================================
# CAMERA BUILDER
# ============================================================
def create_surround_camera(name: str, angle_degree: float, camera_class):
    """Return a camera class rotated to the given angle around the vehicle."""

    class CustomCam(camera_class):
        def __init__(self, width, height, engine, *, cuda=False):
            super().__init__(width, height, engine, cuda=cuda)
            self._angle = angle_degree

        def perceive(self, to_float=True, new_parent_node=None, position=None, hpr=None):
            if new_parent_node is not None:
                rad    = math.radians(self._angle)
                radius = 0.5
                x = -math.sin(rad) * radius
                y =  math.cos(rad) * radius
                z = 1.5
                self.cam.reparentTo(new_parent_node)
                self.cam.setPos(x, y, z)
                self.cam.lookAt(-math.sin(rad) * 10, math.cos(rad) * 10, z)
            return super().perceive(to_float=to_float, new_parent_node=None)

    CustomCam.__name__ = name
    return CustomCam


def build_cameras(num_cameras: int):
    """Return (angles, sensors_dict, rgb_cam_names, depth_cam_names)."""
    angles          = [round(i * 360 / num_cameras) for i in range(num_cameras)]
    sensors         = {}
    rgb_cam_names   = []
    depth_cam_names = []

    for angle in angles:
        rgb_name   = f"cam_{angle}"
        depth_name = f"depth_{angle}"
        sensors[rgb_name]   = (create_surround_camera(f"Cam_{angle}",   angle, RGBCamera),   200, 200)
        sensors[depth_name] = (create_surround_camera(f"Depth_{angle}", angle, DepthCamera),  84,  84)
        rgb_cam_names.append(rgb_name)
        depth_cam_names.append(depth_name)

    return angles, sensors, rgb_cam_names, depth_cam_names


# ============================================================
# PER-FRAME PROCESSING  (GPU / CPU)
# ============================================================
def process_gpu(env, rgb_name: str, depth_name: str, combined_observations: dict):
    """Process one camera pair using CUDA tensors."""
    rgb_cupy = env.engine.get_sensor(rgb_name).perceive(
        to_float=False, new_parent_node=env.agent.origin
    )
    rgb_tensor = torch.as_tensor(rgb_cupy, device="cuda").float()
    rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)

    gray     = (0.2989 * rgb_tensor[:, 0:1] +
                0.5870 * rgb_tensor[:, 1:2] +
                0.1140 * rgb_tensor[:, 2:3])
    mask     = (gray > 180).float()
    lane_map = F.interpolate(mask, size=(84, 84), mode="area").squeeze(0)

    d_cupy   = env.engine.get_sensor(depth_name).perceive(
        to_float=True, new_parent_node=env.agent.origin
    )
    d_tensor  = torch.as_tensor(d_cupy, device="cuda").float()
    if d_tensor.dim() == 3:
        d_tensor = d_tensor[:, :, 0]
    depth_map = d_tensor.unsqueeze(0)

    combined_obs = torch.cat([depth_map, lane_map], dim=0)
    combined_observations[rgb_name].append(combined_obs)

    rgb_np_uint8 = rgb_cupy.get() if hasattr(rgb_cupy, "get") else np.array(rgb_cupy)
    combined_observations[f"{rgb_name}_rgb"].append(rgb_np_uint8.astype(np.uint8))

    return rgb_cupy, d_cupy


def process_cpu(env, rgb_name: str, depth_name: str, combined_observations: dict):
    """Process one camera pair using NumPy on CPU."""
    rgb_img = env.engine.get_sensor(rgb_name).perceive(
        to_float=False, new_parent_node=env.agent.origin
    )
    if hasattr(rgb_img, "get"):
        rgb_img = rgb_img.get()
    rgb_np = np.array(rgb_img, dtype=np.float32)

    gray     = (0.2989 * rgb_np[:, :, 0] +
                0.5870 * rgb_np[:, :, 1] +
                0.1140 * rgb_np[:, :, 2])
    mask     = (gray > 180).astype(np.float32)
    lane_map = cv2.resize(mask, (84, 84), interpolation=cv2.INTER_AREA)[np.newaxis]

    d_img = env.engine.get_sensor(depth_name).perceive(
        to_float=True, new_parent_node=env.agent.origin
    )
    if hasattr(d_img, "get"):
        d_img = d_img.get()
    d_np = np.array(d_img, dtype=np.float32)
    if d_np.ndim == 3:
        d_np = d_np[:, :, 0]
    depth_map = d_np[np.newaxis]

    combined_obs = np.concatenate([depth_map, lane_map], axis=0)
    combined_observations[rgb_name].append(combined_obs)
    combined_observations[f"{rgb_name}_rgb"].append(np.array(rgb_img, dtype=np.uint8))

    return rgb_img, d_img