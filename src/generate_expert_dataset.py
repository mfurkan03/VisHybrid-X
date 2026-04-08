import os
import math
import argparse
import numpy as np
import cv2
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.examples import expert

from panda3d.core import loadPrcFileData
loadPrcFileData("", "stm-max-views 20")


# ==========================================
# KAMERA OLUŞTURUCU
# ==========================================
def create_surround_camera(name, angle_degree, camera_class):
    class CustomCam(camera_class):
        def __init__(self, width, height, engine, *, cuda=False):
            super().__init__(width, height, engine, cuda=cuda)
            self._angle = angle_degree

        def perceive(self, to_float=True, new_parent_node=None, position=None, hpr=None):
            if new_parent_node is not None:
                rad = math.radians(self._angle)
                radius = 0.5
                x = -math.sin(rad) * radius
                y =  math.cos(rad) * radius
                z = 1.5
                self.cam.reparentTo(new_parent_node)
                self.cam.setPos(x, y, z)
                look_x = -math.sin(rad) * 10
                look_y =  math.cos(rad) * 10
                self.cam.lookAt(look_x, look_y, z)
                self.engine.taskMgr.step()
            return super().perceive(to_float=to_float, new_parent_node=None)

    CustomCam.__name__ = name
    return CustomCam


def build_cameras(num_cameras):
    """
    num_cameras adeti 360 dereceye esit boler.
    Ornek: num_cameras=4 -> 0, 90, 180, 270 derece
    """
    angles = [round(i * 360 / num_cameras) for i in range(num_cameras)]
    sensors = {}
    rgb_cam_names   = []
    depth_cam_names = []
    for angle in angles:
        rgb_name   = f"cam_{angle}"
        depth_name = f"depth_{angle}"
        sensors[rgb_name]   = (create_surround_camera(f"Cam_{angle}",   angle, RGBCamera),   84, 84)
        sensors[depth_name] = (create_surround_camera(f"Depth_{angle}", angle, DepthCamera), 84, 84)
        rgb_cam_names.append(rgb_name)
        depth_cam_names.append(depth_name)
    return angles, sensors, rgb_cam_names, depth_cam_names


# ==========================================
# OPENCV GORUNTULEYICI
# ==========================================
def show_cameras(env, rgb_cam_names, depth_cam_names):
    THUMB = 280
    rgb_frames   = []
    depth_frames = []

    for cam_name in rgb_cam_names:
        img = env.engine.get_sensor(cam_name).perceive(
            to_float=False, new_parent_node=env.agent.origin
        )
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_big = cv2.resize(img_bgr, (THUMB, THUMB), interpolation=cv2.INTER_NEAREST)
        angle = cam_name.split("_")[1]
        cv2.putText(img_big, f"RGB {angle}deg", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        rgb_frames.append(img_big)

    for depth_name in depth_cam_names:
        d = env.engine.get_sensor(depth_name).perceive(
            to_float=False, new_parent_node=env.agent.origin
        )
        d = np.array(d)
        if d.ndim == 3:
            d = d[:, :, 0]
        d_norm  = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
        d_big   = cv2.resize(d_color, (THUMB, THUMB), interpolation=cv2.INTER_NEAREST)
        angle = depth_name.split("_")[1]
        cv2.putText(d_big, f"DEPTH {angle}deg", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        depth_frames.append(d_big)

    rgb_row   = np.hstack(rgb_frames)
    depth_row = np.hstack(depth_frames)
    combined  = np.vstack([rgb_row, depth_row])
    cv2.imshow("MetaDrive Cameras  (Q = cikis)", combined)
    return (cv2.waitKey(1) & 0xFF) == ord('q')


# ==========================================
# VERI TOPLAMA FONKSIYONU
# ==========================================
def collect_expert_data(seed, num_episodes=10, save_dir="dataset", visualize=True, num_cameras=2):
    os.makedirs(save_dir, exist_ok=True)

    angles, sensors, rgb_cam_names, depth_cam_names = build_cameras(num_cameras)

    print(f"Veriler '{save_dir}' klasorune kaydediliyor...")
    print(f"Kamera sayisi: {num_cameras}  ->  acilar: {angles} derece")

    config = {
        "use_render": False,
        "image_observation": True,
        "show_interface": False,
        "preload_models": True,
        "decision_repeat": 5,
        "sensors": sensors,
        "vehicle_config": dict(image_source=rgb_cam_names[0]),
        "start_seed": seed,
        "num_scenarios": num_episodes,
        "random_lane_width": True,
        "random_lane_num": True,
        "traffic_density": 0.15,
        "random_traffic": True,
        "map_config": {
            BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
            BaseMap.GENERATE_CONFIG: 7,
        },
    }

    env = MetaDriveEnv(config)
    total_steps = 0
    quit_requested = False

    for ep in range(num_episodes):
        if quit_requested:
            break

        obs, info = env.reset()
        rgb_data   = {name: [] for name in rgb_cam_names}
        depth_data = {name: [] for name in depth_cam_names}
        actions = []
        done = False

        while not done:
            action = expert(env.agent, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            for cam_name in rgb_cam_names:
                img = env.engine.get_sensor(cam_name).perceive(
                    to_float=False, new_parent_node=env.agent.origin
                )
                rgb_data[cam_name].append(np.transpose(np.array(img), (2, 0, 1)))

            for depth_name in depth_cam_names:
                img = env.engine.get_sensor(depth_name).perceive(
                    to_float=False, new_parent_node=env.agent.origin
                )
                depth_data[depth_name].append(np.transpose(np.array(img), (2, 0, 1)))

            actions.append(action)
            done = terminated or truncated
            total_steps += 1

            if visualize:
                quit_requested = show_cameras(env, rgb_cam_names, depth_cam_names)
                if quit_requested:
                    done = True

        save_dict = {"action": np.array(actions)}
        for name in rgb_cam_names:
            save_dict[name] = np.array(rgb_data[name])
        for name in depth_cam_names:
            save_dict[name] = np.array(depth_data[name])

        save_path = os.path.join(save_dir, f"episode_{ep}.npz")
        np.savez_compressed(save_path, **save_dict)
        print(f"Bolum {ep+1}/{num_episodes} kaydedildi. (Adim: {len(actions)})")

    cv2.destroyAllWindows()
    env.close()
    print(f"Veri toplama tamamlandi! Toplam Adim: {total_steps}")


# ==========================================
# ANA CALISTIRMA BLOGU
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetaDrive Imitation Learning Pipeline")
    parser.add_argument("--episodes",    type=int,  default=10)
    parser.add_argument("--save_dir",    type=str,  default="dataset")
    parser.add_argument("--start_seed",  type=int,  default=42)
    parser.add_argument("--num_cameras", type=int,  default=2,
                        help="Aracin etrafina esit dagitilacak kamera sayisi (ornek: 2, 4, 6)")
    parser.add_argument("--no_vis",      action="store_true",
                        help="Goruntulemeyi kapat (daha hizli)")

    args = parser.parse_args()

    collect_expert_data(
        seed=args.start_seed,
        num_episodes=args.episodes,
        save_dir=args.save_dir,
        visualize=not args.no_vis,
        num_cameras=args.num_cameras,
    )