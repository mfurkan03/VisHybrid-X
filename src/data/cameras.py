"""
cameras.py – surround camera construction and per-frame sensor processing.
"""

import math

import cv2
import numpy as np
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.depth_camera import DepthCamera


# ============================================================
# CAMERA BUILDER
# ============================================================
def create_surround_camera(name: str, angle_degree: float, camera_class, fov: float = 60):
    """Return a camera class rotated to the given angle around the vehicle."""

    class CustomCam(camera_class):
        def __init__(self, width, height, engine, *, cuda=False):
            super().__init__(width, height, engine, cuda=cuda)
            self._angle = angle_degree
            self.lens.setFov(fov)

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


def build_cameras(num_cameras: int, fov: float = 60):
    """Return (angles, sensors_dict, rgb_cam_names, depth_cam_names)."""
    angles          = [round(i * 360 / num_cameras) for i in range(num_cameras)]
    sensors         = {}
    rgb_cam_names   = []
    depth_cam_names = []

    for angle in angles:
        rgb_name   = f"cam_{angle}"
        depth_name = f"depth_{angle}"
        sensors[rgb_name]   = (create_surround_camera(f"Cam_{angle}",   angle, RGBCamera,   fov=fov), 196, 196)
        sensors[depth_name] = (create_surround_camera(f"Depth_{angle}", angle, DepthCamera, fov=fov), 196, 196)
        rgb_cam_names.append(rgb_name)
        depth_cam_names.append(depth_name)

    return angles, sensors, rgb_cam_names, depth_cam_names


# ============================================================
# PER-FRAME PROCESSING  (GPU / CPU)
# ============================================================
def process_gpu(env, rgb_name: str, depth_name: str, observations: dict, save: bool = True):
    """Process one camera pair using CUDA tensors."""
    rgb_cupy = env.engine.get_sensor(rgb_name).perceive(
        to_float=False, new_parent_node=env.agent.origin
    )
    # MetaDrive returns BGR, we convert it to RGB right away
    if hasattr(rgb_cupy, "copy"):
        rgb_cupy = rgb_cupy[..., ::-1].copy()
    else:
        rgb_cupy = rgb_cupy[..., ::-1]

    d_cupy   = env.engine.get_sensor(depth_name).perceive(
        to_float=True, new_parent_node=env.agent.origin
    )

    if save:
        rgb_np_uint8 = rgb_cupy.get() if hasattr(rgb_cupy, "get") else np.array(rgb_cupy)
        observations[f"{rgb_name}_rgb"].append(rgb_np_uint8.astype(np.uint8))

        d_np_float32 = d_cupy.get() if hasattr(d_cupy, "get") else np.array(d_cupy)
        observations[f"{rgb_name}_depth"].append(d_np_float32.astype(np.float32))

    return rgb_cupy, d_cupy


def process_cpu(env, rgb_name: str, depth_name: str, observations: dict, save: bool = True):
    """Process one camera pair using NumPy on CPU."""
    rgb_img = env.engine.get_sensor(rgb_name).perceive(
        to_float=False, new_parent_node=env.agent.origin
    )
    if hasattr(rgb_img, "get"):
        rgb_img = rgb_img.get()

    # MetaDrive returns BGR, we convert to RGB
    rgb_img = rgb_img[..., ::-1].copy()

    d_img = env.engine.get_sensor(depth_name).perceive(
        to_float=True, new_parent_node=env.agent.origin
    )
    if hasattr(d_img, "get"):
        d_img = d_img.get()

    if save:
        observations[f"{rgb_name}_depth"].append(np.array(d_img, dtype=np.float32))
        observations[f"{rgb_name}_rgb"].append(np.array(rgb_img, dtype=np.uint8))

    return rgb_img, d_img