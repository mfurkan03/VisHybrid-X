"""
Quick script to capture one frame at 60°, 90°, 100°, and 120° FOV for comparison.
Saves images to fov_compare/ directory.
"""
import os, sys, math
import numpy as np
import cv2

sys.path.insert(0, str(os.path.dirname(__file__)))

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera


def make_cam_class(fov: int):
    class WideCam(RGBCamera):
        def __init__(self, width, height, engine, *, cuda=False):
            super().__init__(width, height, engine, cuda=cuda)
            self.lens.setFov(fov)

        def perceive(self, to_float=True, new_parent_node=None, position=None, hpr=None):
            if new_parent_node is not None:
                self.cam.reparentTo(new_parent_node)
                self.cam.setPos(0, 0.5, 1.5)
                self.cam.lookAt(0, 10, 1.5)
            return super().perceive(to_float=to_float, new_parent_node=None)

    WideCam.__name__ = f"Cam{fov}"
    return WideCam


os.makedirs("fov_compare", exist_ok=True)
FOVS = [60, 90, 100, 120]
CAM_W, CAM_H = 400, 300

for fov in FOVS:
    cam_name = f"cam_fov{fov}"
    CamClass = make_cam_class(fov)

    env = MetaDriveEnv({
        "use_render": False,
        "show_interface": False,
        "image_observation": True,
        "sensors": {cam_name: (CamClass, CAM_W, CAM_H)},
        "vehicle_config": {
            "image_source": cam_name,
            "lidar": {"num_lasers": 0},
            "side_detector": {"num_lasers": 0},
            "lane_line_detector": {"num_lasers": 0},
        },
        "image_on_cuda": False,
        "start_seed": 42,
        "num_scenarios": 1,
        "traffic_density": 0.1,
    })

    obs, _ = env.reset()

    # Step to frame 110
    for _ in range(110):
        obs, _, done, _, _ = env.step([0.0, 0.5])
        if done:
            obs, _ = env.reset()

    img = env.engine.get_sensor(cam_name).perceive(
        to_float=False, new_parent_node=env.agent.origin
    )
    if hasattr(img, "get"):
        img = img.get()
    img = np.array(img, dtype=np.uint8)
    # MetaDrive returns BGR
    img_rgb = img[..., ::-1].copy()

    out_path = f"fov_compare/fov_{fov}.png"
    cv2.imwrite(out_path, img)   # save as BGR (OpenCV default)
    print(f"Saved {out_path}  shape={img.shape}")
    env.close()

print("\nDone! Check fov_compare/ folder.")
