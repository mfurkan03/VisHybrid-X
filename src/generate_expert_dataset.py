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

                # Kamerayı araca bağla ve konumlandır
                self.cam.reparentTo(new_parent_node)
                self.cam.setPos(x, y, z)

                # lookAt ile yönü ayarla
                look_x = -math.sin(rad) * 10
                look_y =  math.cos(rad) * 10
                self.cam.lookAt(look_x, look_y, z)

                self.engine.taskMgr.step()  # render güncelle


                # super()'a new_parent_node=None geçiyoruz
                # böylece super() içindeki setPos/setHpr çağrılmaz
                return super().perceive(to_float=to_float, new_parent_node=None)
            
            return super().perceive(to_float=to_float, new_parent_node=None)

    CustomCam.__name__ = name
    return CustomCam


# 2 RGB + 2 Depth kamera
Cam_0   = create_surround_camera("Cam_0",   0,   RGBCamera)
Cam_180 = create_surround_camera("Cam_180", 180, RGBCamera)
Depth_0   = create_surround_camera("Depth_0",   0,   DepthCamera)
Depth_180 = create_surround_camera("Depth_180", 180, DepthCamera)


# ==========================================
# OPENCV GÖRÜNTÜLEYICI
# ==========================================
def show_cameras(env):
    """RGB ve Depth kameralarını yan yana gösterir. 'q' ile çıkılır."""

    # --- RGB ---
    img_0   = np.array(env.engine.get_sensor("cam_0").get_image(env.agent))
    img_180 = np.array(env.engine.get_sensor("cam_180").get_image(env.agent))

    img_0_bgr   = cv2.cvtColor(img_0,   cv2.COLOR_RGB2BGR)
    img_180_bgr = cv2.cvtColor(img_180, cv2.COLOR_RGB2BGR)

    img_0_big   = cv2.resize(img_0_bgr,   (336, 336), interpolation=cv2.INTER_NEAREST)
    img_180_big = cv2.resize(img_180_bgr, (336, 336), interpolation=cv2.INTER_NEAREST)

    cv2.putText(img_0_big,   "CAM_0  (ileri)",  (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(img_180_big, "CAM_180 (geri)",  (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    rgb_row = np.hstack([img_0_big, img_180_big])

    # --- Depth ---
    d0   = np.array(env.engine.get_sensor("depth_0").get_image(env.agent))
    d180 = np.array(env.engine.get_sensor("depth_180").get_image(env.agent))

    # Tek kanal ise sıkıştır
    if d0.ndim == 3:
        d0   = d0[:, :, 0]
        d180 = d180[:, :, 0]

    d0_norm   = cv2.normalize(d0,   None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    d180_norm = cv2.normalize(d180, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    d0_color   = cv2.applyColorMap(d0_norm,   cv2.COLORMAP_JET)
    d180_color = cv2.applyColorMap(d180_norm, cv2.COLORMAP_JET)

    d0_big   = cv2.resize(d0_color,   (336, 336), interpolation=cv2.INTER_NEAREST)
    d180_big = cv2.resize(d180_color, (336, 336), interpolation=cv2.INTER_NEAREST)

    cv2.putText(d0_big,   "DEPTH_0  (ileri)", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(d180_big, "DEPTH_180 (geri)", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    depth_row = np.hstack([d0_big, d180_big])

    # --- Birleştir ve göster ---
    combined = np.vstack([rgb_row, depth_row])
    cv2.imshow("MetaDrive Cameras  (Q = cikis)", combined)

    key = cv2.waitKey(1) & 0xFF
    return key == ord('q')


# ==========================================
# VERİ TOPLAMA FONKSİYONU
# ==========================================
def collect_expert_data(seed, num_episodes=10, save_dir="dataset", visualize=True):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Veriler '{save_dir}' klasörüne kaydediliyor...")

    angles = ["0", "180"]
    rgb_cam_names   = [f"cam_{a}"   for a in angles]
    depth_cam_names = [f"depth_{a}" for a in angles]

    config = {
        "use_render": False,
        "image_observation": True,
        "show_interface": False,
        "preload_models": True,
        #"image_on_cuda":True,
        "decision_repeat": 5,
        "sensors": {
            "cam_0":     (Cam_0,     84, 84),
            "cam_180":   (Cam_180,   84, 84),
            "depth_0":   (Depth_0,   84, 84),
            "depth_180": (Depth_180, 84, 84),
        },
        "vehicle_config": dict(image_source="cam_0"),
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

            # Kamera görüntülerini kaydet
            for cam_name in rgb_cam_names:
                sensor = env.engine.get_sensor(cam_name)
                img = sensor.perceive(
                    to_float=False,
                    new_parent_node=env.agent.origin,
                    position=None,
                    hpr=None
                )
                img_processed = np.transpose(img, (2, 0, 1))
                rgb_data[cam_name].append(img_processed)

            for depth_name in depth_cam_names:
                sensor = env.engine.get_sensor(depth_name)
                img = sensor.perceive(
                    to_float=False,
                    new_parent_node=env.agent.origin,
                    position=None,
                    hpr=None
                )
                img_processed = np.transpose(img, (2, 0, 1))
                depth_data[depth_name].append(img_processed)

            actions.append(action)
            done = terminated or truncated
            total_steps += 1

            # Gerçek zamanlı görüntüleme
            if visualize:
                quit_requested = show_cameras(env)
                if quit_requested:
                    done = True

        # Bölümü kaydet
        save_dict = {"action": np.array(actions)}
        for name in rgb_cam_names:
            save_dict[name] = np.array(rgb_data[name])
        for name in depth_cam_names:
            save_dict[name] = np.array(depth_data[name])

        save_path = os.path.join(save_dir, f"episode_{ep}.npz")
        np.savez_compressed(save_path, **save_dict)
        print(f"Bölüm {ep+1}/{num_episodes} kaydedildi. (Adım: {len(actions)})")

    cv2.destroyAllWindows()
    env.close()
    print(f"Veri toplama tamamlandı! Toplam Adım: {total_steps}")


# ==========================================
# ANA ÇALIŞTIRMA BLOĞU
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetaDrive Imitation Learning Pipeline")
    parser.add_argument("--episodes",   type=int,  default=10)
    parser.add_argument("--save_dir",   type=str,  default="dataset")
    parser.add_argument("--start_seed", type=int,  default=42)
    parser.add_argument("--no_vis",     action="store_true", help="Görüntülemeyi kapat (daha hızlı)")

    args = parser.parse_args()

    collect_expert_data(
        seed=args.start_seed,
        num_episodes=args.episodes,
        save_dir=args.save_dir,
        visualize=not args.no_vis
    )