"""Find a seed where frame 100 is near a junction."""
import sys, os, numpy as np, cv2
from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera

os.makedirs("fov_compare/seeds", exist_ok=True)

class Cam60(RGBCamera):
    def __init__(self, w, h, engine, *, cuda=False):
        super().__init__(w, h, engine, cuda=cuda)
        self.lens.setFov(60)
    def perceive(self, to_float=True, new_parent_node=None, position=None, hpr=None):
        if new_parent_node is not None:
            self.cam.reparentTo(new_parent_node)
            self.cam.setPos(0, 0.5, 1.5)
            self.cam.lookAt(0, 10, 1.5)
        return super().perceive(to_float=to_float, new_parent_node=None)

for seed in [0, 5, 10, 15, 20, 25, 30, 35, 40]:
    env = MetaDriveEnv({
        "use_render": False, "show_interface": False,
        "image_observation": True,
        "sensors": {"cam0": (Cam60, 400, 300)},
        "vehicle_config": {
            "image_source": "cam0",
            "lidar": {"num_lasers": 0},
            "side_detector": {"num_lasers": 0},
            "lane_line_detector": {"num_lasers": 0},
        },
        "image_on_cuda": False,
        "start_seed": seed,
        "num_scenarios": 1,
        "traffic_density": 0.1,
    })
    obs, _ = env.reset()
    for i in range(100):
        obs, _, done, _, info = env.step([0.0, 0.5])
        if done:
            obs, _ = env.reset()
    img = env.engine.get_sensor("cam0").perceive(
        to_float=False, new_parent_node=env.agent.origin
    )
    if hasattr(img, "get"):
        img = img.get()
    img = np.array(img, dtype=np.uint8)
    cv2.imwrite(f"fov_compare/seeds/seed_{seed}.png", img)
    print(f"seed {seed} saved")
    env.close()

print("Done.")
