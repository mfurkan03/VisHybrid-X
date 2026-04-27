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

# Import ego-state helper from models
from models import extract_ego_state, EGO_DIM


# ==========================================
# FPS COUNTER
# ==========================================
class FPSCounter:
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
            f"  Total Steps : {self.total_steps}\n"
            f"  Total Time  : {self.total_time:.2f} s\n"
            f"  Average FPS : {self.average_fps:.1f}\n"
            f"  Instant FPS : {self.instant_fps:.1f}"
        )


# ==========================================
# CAMERA BUILDER
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


# ==========================================
# OPENCV VIEWER (with FPS overlay)
# ==========================================
def show_cameras(raw_frames, rgb_cam_names, depth_cam_names, fps_counter: FPSCounter):
    THUMB = 280
    rgb_frames   = []
    depth_frames = []

    for cam_name, depth_name in zip(rgb_cam_names, depth_cam_names):
        rgb_raw, depth_raw = raw_frames[cam_name]

        img = rgb_raw
        if hasattr(img, 'get'):
            img = img.get()
        img     = np.array(img, dtype=np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_big = cv2.resize(img_bgr, (THUMB, THUMB), interpolation=cv2.INTER_NEAREST)
        angle   = cam_name.split("_")[1]
        cv2.putText(img_big, f"RGB {angle}deg", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        rgb_frames.append(img_big)

        d = depth_raw
        if hasattr(d, 'get'):
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

    fps_text = (f"FPS: {fps_counter.instant_fps:5.1f}  |  "
                f"Avg: {fps_counter.average_fps:5.1f}  |  "
                f"Steps: {fps_counter.total_steps}")
    cv2.rectangle(combined, (0, 0), (len(fps_text) * 11 + 10, 30), (0, 0, 0), -1)
    cv2.putText(combined, fps_text, (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("MetaDrive Cameras  (Q = Quit)", combined)
    return (cv2.waitKey(1) & 0xFF) == ord('q')


# ==========================================
# GPU PROCESSING
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

    rgb_np_uint8 = rgb_cupy.get() if hasattr(rgb_cupy, 'get') else np.array(rgb_cupy)
    rgb_np_uint8 = rgb_np_uint8.astype(np.uint8)
    combined_observations[f"{rgb_name}_rgb"].append(rgb_np_uint8)

    return rgb_cupy, d_cupy


# ==========================================
# CPU PROCESSING
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

    rgb_np_uint8 = np.array(rgb_img, dtype=np.uint8)
    combined_observations[f"{rgb_name}_rgb"].append(rgb_np_uint8)

    return rgb_img, d_img


# ==========================================
# DATA COLLECTION FUNCTION
# ==========================================
# ==========================================
# DATA COLLECTION FUNCTION
# ==========================================
def collect_expert_data(
    seed,
    num_episodes   = 10,
    save_dir       = "dataset",
    visualize      = True,
    num_cameras    = 2,
    image_on_cuda  = True,
    split_ratios   = (0.8, 0.1, 0.1),
):
    os.makedirs(os.path.join(save_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "val"),   exist_ok=True)
    os.makedirs(os.path.join(save_dir, "test"),  exist_ok=True)

    angles, sensors, rgb_cam_names, depth_cam_names = build_cameras(num_cameras)

    train_count = int(num_episodes * split_ratios[0])
    val_count   = int(num_episodes * split_ratios[1])

    mode_str = "GPU (CUDA)" if image_on_cuda else "CPU (NumPy)"
    print(f"\n{'='*55}")
    print(f"  Processing Mode : {mode_str}")
    print(f"  Saving to       : '{save_dir}' (Split into train/val/test)")
    print(f"  Camera Count    : {num_cameras}  ->  Angles: {angles} degrees")
    print(f"  Ego-state dim   : {EGO_DIM}  -> model sees [total_speed, last_steer]")
    print(f"  Ego logged (full): total_speed, last_steer, forward_speed, lateral_speed, heading_delta, timestamp")
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
        "num_scenarios":     num_episodes * 2, # Buffer in case many episodes are skipped
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

    ep = 0
    while ep < num_episodes:
        if quit_requested:
            break
        
        current_split = (
            "train" if ep < train_count
            else "val" if ep < train_count + val_count
            else "test"
        )

        obs, info = env.reset()

        # Storage buffers
        combined_observations = {
            key: []
            for rgb_name in rgb_cam_names
            for key in (rgb_name, f"{rgb_name}_rgb")
        }
        actions         = []
        ego_states      = []   # ← model-facing slice: (T, EGO_DIM) = [total_speed, last_steer]
        ego_states_full = []   # ← full EgoReading fields for offline analysis
        frame_timestamps = []  # ← wall-clock time of each ego/frame reading
        last_steer      = 0.0  # ← track previous steer for ego state

        done = False
        ep_steps = 0
        while not done:
            fps_counter.tick()
            ep_steps+=1
            # Expert action + noise
            expert_action  = expert(env.agent, deterministic=True)
            applied_action = expert_action.copy()
            # if fps_counter.total_steps % 10 == 0:
            #     applied_action[0] += random.uniform(-0.05, 0.05)

            # --- Ego state (before step, reflects current state) ---
            reading = extract_ego_state(env.agent, last_steer=last_steer)
            ego_states.append(reading.ego_model)           # 2-dim model input
            ego_states_full.append([                       # full snapshot for logging
                reading.total_speed,
                reading.last_steer,
                reading.forward_speed,
                reading.lateral_speed,
                reading.heading_delta,
            ])
            frame_timestamps.append(reading.timestamp)     # wall-clock seconds

            # Process cameras
            raw_frames = {}
            for rgb_name, depth_name in zip(rgb_cam_names, depth_cam_names):
                rgb_raw, depth_raw = process_fn(env, rgb_name, depth_name, combined_observations)
                raw_frames[rgb_name] = (rgb_raw, depth_raw)

            actions.append(expert_action.copy())
            last_steer = float(expert_action[0])   # update for next step

            obs, reward, terminated, truncated, info = env.step(applied_action)
            done = terminated or truncated
            if ep_steps >= 1000:
                done = True
            if fps_counter.total_steps % 50 == 0:
                print(f"  [Ep {ep+1}/{num_episodes}] "
                      f"Step: {fps_counter.total_steps:5d}  |  "
                      f"Inst FPS: {fps_counter.instant_fps:5.1f}  |  "
                      f"Avg FPS: {fps_counter.average_fps:5.1f}  |  "
                      f"Ego: spd={reading.total_speed:+.2f} fwd={reading.forward_speed:+.2f} "
                      f"lat={reading.lateral_speed:+.2f} hdg={reading.heading_delta:+.2f} "
                      f"str={reading.last_steer:+.2f}",
                      flush=True)

            if visualize:
                quit_requested = show_cameras(
                    raw_frames, rgb_cam_names, depth_cam_names, fps_counter
                )
                if quit_requested:
                    done = True

        # Build save dict
        save_dict = {
            "action":          np.array(actions),
            # Model input: (T, EGO_DIM) = [total_speed, last_steer]
            "ego_state":       np.array(ego_states,      dtype=np.float32),
            # Full snapshot: (T, 5) = [total_speed, last_steer, forward_speed, lateral_speed, heading_delta]
            # Column order matches the list above; use ego_states_full[:, 2] for forward_speed, etc.
            "ego_state_full":  np.array(ego_states_full, dtype=np.float32),
            # Wall-clock timestamp (seconds since epoch) for each step — use np.diff() to get dt
            "frame_timestamps": np.array(frame_timestamps, dtype=np.float64),
        }
        for rgb_name in rgb_cam_names:
            obs_list = combined_observations[rgb_name]
            if image_on_cuda and isinstance(obs_list[0], torch.Tensor):
                stacked = torch.stack(obs_list).cpu().numpy()
            else:
                stacked = np.array(obs_list)
            save_dict[f"{rgb_name}_combined"] = stacked

            rgb_list = combined_observations[f"{rgb_name}_rgb"]
            save_dict[f"{rgb_name}_rgb"] = np.array(rgb_list, dtype=np.uint8)

        save_path = os.path.join(save_dir, current_split, f"episode_{ep}.npz")

        def _save(p, d):
            np.savez_compressed(p, **d)

        t = threading.Thread(target=_save, args=(save_path, save_dict), daemon=True)
        t.start()
        print(f"\n  --> Saving Episode {ep+1}/{num_episodes} to '{current_split}' "
              f"(Steps: {ep_steps}, Ego-states: {len(ego_states)})")
        
        # Increment episode counter only upon a successful run <= 1000 steps
        ep += 1

    cv2.destroyAllWindows()
    env.close()

    print(f"\n{'='*55}")
    print(f"  DATA COLLECTION COMPLETED")
    print(f"  Mode: {mode_str}")
    print(fps_counter.summary())
    print(f"{'='*55}\n")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MetaDrive Multi-Cam Imitation Learning"
    )
    parser.add_argument("--episodes",       type=int,  default=10)
    parser.add_argument("--save_dir",       type=str,  default="dataset")
    parser.add_argument("--start_seed",     type=int,  default=42)
    parser.add_argument("--num_cameras",    type=int,  default=1)
    parser.add_argument("--no_vis",         action="store_true")
    parser.add_argument("--image_on_cuda",  action="store_true", default=False)

    args = parser.parse_args()

    collect_expert_data(
        seed          = args.start_seed,
        num_episodes  = args.episodes,
        save_dir      = args.save_dir,
        visualize     = not args.no_vis,
        num_cameras   = args.num_cameras,
        image_on_cuda = args.image_on_cuda,
    )