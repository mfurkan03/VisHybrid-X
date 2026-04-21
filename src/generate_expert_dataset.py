import os
import time
import threading
import torch
import math
import argparse
import random
import numpy as np
import cv2
from collections import deque
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.examples import expert
import torch.nn.functional as F

from panda3d.core import loadPrcFileData
loadPrcFileData("", "stm-max-views 20")


# ==========================================
# FPS SAYACI
# ==========================================
class FPSCounter:
    """
    Kayan pencere ile anlık ve ortalama FPS hesaplar.
    """
    def __init__(self, window=60):
        self.window = window
        self._times = deque(maxlen=window)
        self._start = None
        self.total_steps = 0
        self.total_time  = 0.0

    def tick(self):
        now = time.perf_counter()
        if self._start is not None:
            delta = now - self._start
            self._times.append(delta)
            self.total_time  += delta
            self.total_steps += 1
        self._start = now

    @property
    def instant_fps(self):
        if len(self._times) < 2:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))

    @property
    def average_fps(self):
        if self.total_steps == 0 or self.total_time == 0:
            return 0.0
        return self.total_steps / self.total_time

    def summary(self):
        return (
            f"  Toplam Adim : {self.total_steps}\n"
            f"  Toplam Sure : {self.total_time:.2f} s\n"
            f"  Ortalama FPS: {self.average_fps:.1f}\n"
            f"  Anlık FPS   : {self.instant_fps:.1f}"
        )


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
                rad    = math.radians(self._angle)
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
    angles         = [round(i * 360 / num_cameras) for i in range(num_cameras)]
    sensors        = {}
    rgb_cam_names  = []
    depth_cam_names = []

    for angle in angles:
        rgb_name   = f"cam_{angle}"
        depth_name = f"depth_{angle}"
        sensors[rgb_name]   = (create_surround_camera(f"Cam_{angle}",   angle, RGBCamera),   200, 200)
        sensors[depth_name] = (create_surround_camera(f"Depth_{angle}", angle, DepthCamera),  84,  84)
        rgb_cam_names.append(rgb_name)
        depth_cam_names.append(depth_name)

    return angles, sensors, rgb_cam_names, depth_cam_names


# ==========================================
# OPENCV GÖRÜNTÜLEYICI  (FPS overlay dahil)
# raw_frames: {cam_name: (rgb_raw, depth_raw)} — process_fn'den gelen, zaten render edilmiş
# ==========================================
def show_cameras(raw_frames, rgb_cam_names, depth_cam_names, fps_counter: FPSCounter):
    THUMB = 280
    rgb_frames   = []
    depth_frames = []

    for cam_name, depth_name in zip(rgb_cam_names, depth_cam_names):
        rgb_raw, depth_raw = raw_frames[cam_name]

        # RGB
        img = rgb_raw
        if hasattr(img, 'get'):   # CuPy -> NumPy
            img = img.get()
        img = np.array(img, dtype=np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_big = cv2.resize(img_bgr, (THUMB, THUMB), interpolation=cv2.INTER_NEAREST)
        angle   = cam_name.split("_")[1]
        cv2.putText(img_big, f"RGB {angle}deg", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        rgb_frames.append(img_big)

        # Depth
        d = depth_raw
        if hasattr(d, 'get'):     # CuPy -> NumPy
            d = d.get()
        d = np.array(d, dtype=np.float32)
        if d.ndim == 3:
            d = d[:, :, 0]
        d_norm  = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
        d_big   = cv2.resize(d_color, (THUMB, THUMB), interpolation=cv2.INTER_NEAREST)
        angle   = depth_name.split("_")[1]
        cv2.putText(d_big, f"DEPTH {angle}deg", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        depth_frames.append(d_big)

    rgb_row   = np.hstack(rgb_frames)
    depth_row = np.hstack(depth_frames)
    combined  = np.vstack([rgb_row, depth_row])

    # ---- FPS overlay ----
    fps_text = (f"FPS: {fps_counter.instant_fps:5.1f}  |  "
                f"Ort: {fps_counter.average_fps:5.1f}  |  "
                f"Adim: {fps_counter.total_steps}")
    cv2.rectangle(combined, (0, 0), (len(fps_text) * 11 + 10, 30), (0, 0, 0), -1)
    cv2.putText(combined, fps_text, (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("MetaDrive Cameras  (Q = cikis)", combined)
    return (cv2.waitKey(1) & 0xFF) == ord('q')


# ==========================================
# GPU İŞLEME (image_on_cuda=True)
# ==========================================
def process_gpu(env, rgb_name, depth_name, combined_observations):
    rgb_cupy = env.engine.get_sensor(rgb_name).perceive(
        to_float=False, new_parent_node=env.agent.origin
    )
    rgb_tensor = torch.as_tensor(rgb_cupy, device='cuda').float()
    rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)

    gray     = (0.2989 * rgb_tensor[:, 0:1] +
                0.5870 * rgb_tensor[:, 1:2] +
                0.1140 * rgb_tensor[:, 2:3])
    mask     = (gray > 180).float()
    lane_map = F.interpolate(mask, size=(84, 84), mode='area').squeeze(0)

    d_cupy   = env.engine.get_sensor(depth_name).perceive(
        to_float=True, new_parent_node=env.agent.origin
    )
    d_tensor  = torch.as_tensor(d_cupy, device='cuda').float()
    if d_tensor.dim() == 3:
        d_tensor = d_tensor[:, :, 0]
    depth_map = d_tensor.unsqueeze(0)

    combined_obs = torch.cat([depth_map, lane_map], dim=0)
    combined_observations[rgb_name].append(combined_obs)

    # +++ Save raw RGB as uint8 CPU numpy for the training pipeline +++
    rgb_np_uint8 = rgb_cupy.get() if hasattr(rgb_cupy, 'get') else np.array(rgb_cupy)
    rgb_np_uint8 = rgb_np_uint8.astype(np.uint8)
    combined_observations[f"{rgb_name}_rgb"].append(rgb_np_uint8)

    return rgb_cupy, d_cupy


# ==========================================
# CPU İŞLEME (image_on_cuda=False)
# ==========================================
def process_cpu(env, rgb_name, depth_name, combined_observations):
    rgb_img = env.engine.get_sensor(rgb_name).perceive(
        to_float=False, new_parent_node=env.agent.origin
    )
    if hasattr(rgb_img, 'get'):
        rgb_img = rgb_img.get()
    rgb_np = np.array(rgb_img, dtype=np.float32)

    gray     = (0.2989 * rgb_np[:, :, 0] +
                0.5870 * rgb_np[:, :, 1] +
                0.1140 * rgb_np[:, :, 2])
    mask     = (gray > 180).astype(np.float32)
    lane_map = cv2.resize(mask, (84, 84), interpolation=cv2.INTER_AREA)
    lane_map = lane_map[np.newaxis, :, :]

    d_img = env.engine.get_sensor(depth_name).perceive(
        to_float=True, new_parent_node=env.agent.origin
    )
    if hasattr(d_img, 'get'):
        d_img = d_img.get()
    d_np = np.array(d_img, dtype=np.float32)
    if d_np.ndim == 3:
        d_np = d_np[:, :, 0]
    depth_map = d_np[np.newaxis, :, :]

    combined_obs = np.concatenate([depth_map, lane_map], axis=0)
    combined_observations[rgb_name].append(combined_obs)

    # +++ Save raw RGB as uint8 for the training pipeline +++
    rgb_np_uint8 = np.array(rgb_img, dtype=np.uint8)
    combined_observations[f"{rgb_name}_rgb"].append(rgb_np_uint8)

    return rgb_img, d_img


# ==========================================
# VERİ TOPLAMA FONKSIYONU
# ==========================================
def collect_expert_data(
    seed,
    num_episodes   = 10,
    save_dir       = "dataset",
    visualize      = True,
    num_cameras    = 2,
    image_on_cuda  = True,
):
    os.makedirs(save_dir, exist_ok=True)

    angles, sensors, rgb_cam_names, depth_cam_names = build_cameras(num_cameras)

    mode_str = "GPU (CUDA)" if image_on_cuda else "CPU (NumPy)"
    print(f"\n{'='*55}")
    print(f"  İşlem Modu   : {mode_str}")
    print(f"  Veriler      : '{save_dir}' klasörüne kaydediliyor")
    print(f"  Kamera sayısı: {num_cameras}  ->  açılar: {angles} derece")
    print(f"{'='*55}\n")

    config = {
        "use_render":        False,
        "image_observation": True,
        "show_interface":    False,
        "preload_models":    True,
        "decision_repeat":   5,
        "sensors":           sensors,
        "vehicle_config":    dict(image_source=rgb_cam_names[0]),
        "start_seed":        seed,
        "num_scenarios":     num_episodes,
        "random_lane_width": True,
        "random_lane_num":   True,
        "traffic_density":   0.15,
        "random_traffic":    True,
        "map_config": {
            BaseMap.GENERATE_TYPE:   MapGenerateMethod.BIG_BLOCK_NUM,
            BaseMap.GENERATE_CONFIG: 7,
        },
        "image_on_cuda": image_on_cuda,
    }

    env            = MetaDriveEnv(config)
    fps_counter    = FPSCounter(window=60)
    process_fn     = process_gpu if image_on_cuda else process_cpu
    quit_requested = False

    for ep in range(num_episodes):
        if quit_requested:
            break

        obs, info = env.reset()
        # +++ Add *_rgb lists alongside the existing *_combined lists +++
        combined_observations = {
            key: []
            for rgb_name in rgb_cam_names
            for key in (rgb_name, f"{rgb_name}_rgb")
        }
        actions = []
        done    = False

        while not done:
            fps_counter.tick()   # <-- adım başında sayaç

            # Uzman karar + gürültü
            expert_action  = expert(env.agent, deterministic=True)
            applied_action = expert_action.copy()
            if fps_counter.total_steps % 10 == 0:
                applied_action[0] += random.uniform(-0.3, 0.3)

            # Kamera işleme (GPU veya CPU) — ham kareler saklanır, çift render yok
            raw_frames = {}
            for rgb_name, depth_name in zip(rgb_cam_names, depth_cam_names):
                rgb_raw, depth_raw = process_fn(env, rgb_name, depth_name, combined_observations)
                raw_frames[rgb_name] = (rgb_raw, depth_raw)

            actions.append(expert_action.copy())
            obs, reward, terminated, truncated, info = env.step(applied_action)
            done = terminated or truncated

            # Terminal çıktısı (her 50 adımda)
            if fps_counter.total_steps % 50 == 0:
                print(f"  [Ep {ep+1}/{num_episodes}] "
                      f"Adım: {fps_counter.total_steps:5d}  |  "
                      f"Anlık FPS: {fps_counter.instant_fps:5.1f}  |  "
                      f"Ort FPS: {fps_counter.average_fps:5.1f}",
                      flush=True)

            if visualize:
                quit_requested = show_cameras(
                    raw_frames, rgb_cam_names, depth_cam_names, fps_counter
                )
                if quit_requested:
                    done = True

        # Bölüm kaydı — GPU tensörleri toplu CPU'ya indir, async kayıt
        save_dict = {"action": np.array(actions)}
        for rgb_name in rgb_cam_names:
            # --- combined (depth+lane, legacy key kept for compatibility) ---
            obs_list = combined_observations[rgb_name]
            if image_on_cuda and isinstance(obs_list[0], torch.Tensor):
                stacked = torch.stack(obs_list).cpu().numpy()
            else:
                stacked = np.array(obs_list)
            save_dict[f"{rgb_name}_combined"] = stacked

            # +++ raw RGB frames uint8 [N, H, W, 3] — consumed by MetaDriveRGBDataset +++
            rgb_list = combined_observations[f"{rgb_name}_rgb"]
            save_dict[f"{rgb_name}_rgb"] = np.array(rgb_list, dtype=np.uint8)

        save_path = os.path.join(save_dir, f"episode_{ep}.npz")
        ep_steps  = len(actions)

        def _save(p, d):
            np.savez_compressed(p, **d)

        t = threading.Thread(target=_save, args=(save_path, save_dict), daemon=True)
        t.start()
        print(f"\n  --> Bölüm {ep+1}/{num_episodes} kaydediliyor (arka planda). "
              f"(Adım: {ep_steps})")

    cv2.destroyAllWindows()
    env.close()

    # ---- Özet Rapor ----
    print(f"\n{'='*55}")
    print(f"  VERİ TOPLAMA TAMAMLANDI")
    print(f"  Mod: {mode_str}")
    print(fps_counter.summary())
    print(f"{'='*55}\n")


# ==========================================
# ANA ÇALIŞTIRMA BLOĞU
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MetaDrive Multi-Cam Imitation Learning"
    )
    parser.add_argument("--episodes",       type=int,  default=10)
    parser.add_argument("--save_dir",       type=str,  default="dataset")
    parser.add_argument("--start_seed",     type=int,  default=42)
    parser.add_argument("--num_cameras",    type=int,  default=1,
                        help="Aracın etrafına eşit dağıtılacak kamera sayısı")
    parser.add_argument("--no_vis",         action="store_true",
                        help="Görüntülemeyi kapat (daha hızlı)")
    parser.add_argument("--image_on_cuda",  action="store_true", default=False,
                        help="GPU (CUDA) işleme pipeline'ını etkinleştir. "
                             "Kapalıysa klasik CPU/NumPy kullanılır.")

    args = parser.parse_args()

    collect_expert_data(
        seed          = args.start_seed,
        num_episodes  = args.episodes,
        save_dir      = args.save_dir,
        visualize     = not args.no_vis,
        num_cameras   = args.num_cameras,
        image_on_cuda = args.image_on_cuda,
    )