"""
generate_expert_dataset.py – collect expert demonstrations from MetaDrive.

Usage
-----
python src/generate_expert_dataset.py --episodes 10 --save_dir dataset
"""

import argparse
import os
import random
import threading

import numpy as np
from panda3d.core import loadPrcFileData

loadPrcFileData("", "stm-max-views 20")

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.examples import expert

from models import extract_ego_state, EGO_DIM
from data.cameras import build_cameras, process_gpu, process_cpu
from data.viewer import show_cameras
from utils.fps import FPSCounter


# ============================================================
# DATA COLLECTION
# ============================================================
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
    mode_str    = "GPU (CUDA)" if image_on_cuda else "CPU (NumPy)"

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
        "num_scenarios":     num_episodes * 2,
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

        combined_observations = {
            key: []
            for rgb_name in rgb_cam_names
            for key in (rgb_name, f"{rgb_name}_rgb")
        }
        actions          = []
        ego_states       = []
        ego_states_full  = []
        frame_timestamps = []
        last_steer       = 0.0

        done = False
        ep_steps = 0

        while not done:
            fps_counter.tick()
            ep_steps += 1

            expert_action  = expert(env.agent, deterministic=True)
            applied_action = expert_action.copy()
            if fps_counter.total_steps % 10 == 0:
                applied_action[0] += random.uniform(-0.3, 0.3)

            reading = extract_ego_state(env.agent, last_steer=last_steer)
            ego_states.append(reading.ego_model)
            ego_states_full.append([
                reading.total_speed,
                reading.last_steer,
                reading.forward_speed,
                reading.lateral_speed,
                reading.heading_delta,
            ])
            frame_timestamps.append(reading.timestamp)

            raw_frames = {}
            for rgb_name, depth_name in zip(rgb_cam_names, depth_cam_names):
                rgb_raw, depth_raw = process_fn(env, rgb_name, depth_name, combined_observations)
                raw_frames[rgb_name] = (rgb_raw, depth_raw)

            actions.append(expert_action.copy())
            last_steer = float(expert_action[0])

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
                quit_requested = show_cameras(raw_frames, rgb_cam_names, depth_cam_names, fps_counter)
                if quit_requested:
                    done = True

        # Build and save episode
        save_dict = {
            "action":           np.array(actions),
            "ego_state":        np.array(ego_states,      dtype=np.float32),
            "ego_state_full":   np.array(ego_states_full, dtype=np.float32),
            "frame_timestamps": np.array(frame_timestamps, dtype=np.float64),
        }
        for rgb_name in rgb_cam_names:
            obs_list = combined_observations[rgb_name]
            if image_on_cuda and hasattr(obs_list[0], "cpu"):
                import torch
                stacked = torch.stack(obs_list).cpu().numpy()
            else:
                stacked = np.array(obs_list)
            save_dict[f"{rgb_name}_combined"] = stacked
            save_dict[f"{rgb_name}_rgb"] = np.array(combined_observations[f"{rgb_name}_rgb"], dtype=np.uint8)

        save_path = os.path.join(save_dir, current_split, f"episode_{ep}.npz")
        t = threading.Thread(
            target=lambda p, d: np.savez_compressed(p, **d),
            args=(save_path, save_dict),
            daemon=True,
        )
        t.start()
        print(f"\n  --> Saving Episode {ep+1}/{num_episodes} to '{current_split}' "
              f"(Steps: {ep_steps}, Ego-states: {len(ego_states)})")

        ep += 1

    import cv2
    cv2.destroyAllWindows()
    env.close()

    print(f"\n{'='*55}")
    print(f"  DATA COLLECTION COMPLETED")
    print(f"  Mode: {mode_str}")
    print(fps_counter.summary())
    print(f"{'='*55}\n")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetaDrive Multi-Cam Imitation Learning")
    parser.add_argument("--episodes",      type=int,  default=10)
    parser.add_argument("--save_dir",      type=str,  default="dataset")
    parser.add_argument("--start_seed",    type=int,  default=42)
    parser.add_argument("--num_cameras",   type=int,  default=1)
    parser.add_argument("--no_vis",        action="store_true")
    parser.add_argument("--image_on_cuda", action="store_true", default=False)
    args = parser.parse_args()

    collect_expert_data(
        seed          = args.start_seed,
        num_episodes  = args.episodes,
        save_dir      = args.save_dir,
        visualize     = not args.no_vis,
        num_cameras   = args.num_cameras,
        image_on_cuda = args.image_on_cuda,
    )