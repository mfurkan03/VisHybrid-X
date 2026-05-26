"""
generate_expert_dataset.py – collect expert demonstrations from MetaDrive in parallel.

Usage
-----
python generate_expert_dataset.py --episodes 100 --num_workers 4 --save_dir dataset
"""

import argparse
import os
import random
import threading
import multiprocessing as mp

import numpy as np
from panda3d.core import loadPrcFileData

loadPrcFileData("", "stm-max-views 20")

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.examples import expert

from models import extract_ego_state, navigation_command_onehot, EGO_DIM
from data.cameras import build_cameras, process_gpu, process_cpu
from data.viewer import show_cameras
from utils.fps import FPSCounter


# MetaDrive's own info["navigation_command"] string → our (navi_left, navi_right)
# one-hot. Used to validate that navigation_command_onehot() matches the
# simulator's convention (catches a flipped left/right label).
_MD_CMD_TO_ONEHOT = {
    "forward": (0.0, 0.0),
    "straight": (0.0, 0.0),
    "left":    (1.0, 0.0),
    "right":   (0.0, 1.0),
}


# ============================================================
# WORKER FUNCTION
# ============================================================
def _worker_collect(
    worker_id,
    start_ep_idx,
    num_episodes,
    total_episodes,
    seed,
    save_dir,
    action_noise,
    visualize,
    num_cameras,
    image_on_cuda,
    split_ratios,
    decision_repeat=1,
    save_every_n=20,
    poster_path=None,
    traffic_density_min=0.1,
    traffic_density_max=0.7,
):
    """
    Worker process that handles a subset of the total episodes.

    Navigation is collected in the SAME loop that captures the images (the only
    reliable way — MetaDrive episodes are not reproducible across runs, so a
    separate replay pass would misalign nav with the saved frames) and saved
    as the `navi_state` key inside the main episode .npz file alongside RGB,
    depth, action, and ego_state.
    """

    angles, sensors, rgb_cam_names, depth_cam_names = build_cameras(num_cameras)

    # Calculate global splits
    train_count = int(total_episodes * split_ratios[0])
    val_count   = int(total_episodes * split_ratios[1])
    rng = np.random.default_rng(seed)
    traffic_density = (traffic_density_min if traffic_density_min == traffic_density_max
                       else float(rng.uniform(traffic_density_min, traffic_density_max)))
    config = {
        "use_render":        False,
        "image_observation": True,
        "show_interface":    False,
        "preload_models":    True,
        "decision_repeat":   decision_repeat,  # 100 Hz physics / 5 = 20 FPS
        "sensors":           sensors,
        "vehicle_config":    dict(image_source=rgb_cam_names[0]),
        "start_seed":        seed, # Unique start seed per worker to avoid duplicate maps
        "num_scenarios":     num_episodes * 2,
        "random_lane_width": True,
        "random_lane_num":   True,
        "traffic_density":   traffic_density,
        "random_traffic":    True,
        "map_config": {
            BaseMap.GENERATE_TYPE:   MapGenerateMethod.BIG_BLOCK_NUM,
            BaseMap.GENERATE_CONFIG: 3,
        },
        "image_on_cuda": image_on_cuda,
    }

    env = MetaDriveEnv(config)
    fps_counter = FPSCounter(window=60)
    process_fn = process_gpu if image_on_cuda else process_cpu
    quit_requested = False

    # Nav-label validation accumulators (compare our one-hot vs MetaDrive's
    # own info["navigation_command"] over the whole worker run).
    nav_key_present = 0   # steps where info exposed navigation_command
    nav_recognized  = 0   # of those, value mapped to a known command
    nav_mismatch    = 0   # of recognized, our one-hot disagreed

    ep = 0
    while ep < num_episodes:
        if quit_requested:
            break

        global_ep_id = start_ep_idx + ep

        # Determine which folder this specific episode belongs to
        current_split = (
            "train" if global_ep_id < train_count
            else "val" if global_ep_id < train_count + val_count
            else "test"
        )

        obs, info = env.reset()

        observations = {
            key: []
            for rgb_name in rgb_cam_names
            for key in (f"{rgb_name}_depth", f"{rgb_name}_rgb")
        }
        actions          = []
        ego_states       = []
        ego_states_full  = []
        navi_states      = []
        frame_timestamps = []
        last_steer       = 0.0

        done = False
        ep_steps = 0

        while not done:
            fps_counter.tick()
            ep_steps += 1

            expert_action  = expert(env.agent, deterministic=True)
            applied_action = expert_action.copy()
            if fps_counter.total_steps % 50 == 0 and current_split =="train": # This disables the augmentation in validaiton and test processes
                applied_action[0] += random.uniform(-action_noise, action_noise)

            reading = extract_ego_state(env.agent, last_steer=last_steer)

            save_this_frame = (ep_steps % save_every_n == 0)
            if save_this_frame:
                ego_states.append(reading.ego_motion)   # 3-dim motion only (nav kept separate)
                navi_states.append(reading.ego_nav)     # 2-dim [navi_left, navi_right]
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
                rgb_raw, depth_raw = process_fn(env, rgb_name, depth_name, observations, save=save_this_frame)
                raw_frames[rgb_name] = (rgb_raw, depth_raw)

            if save_this_frame:
                actions.append(expert_action.copy())
            last_steer = float(expert_action[0])

            obs, reward, terminated, truncated, info = env.step(applied_action)
            done = terminated or truncated
            if ep_steps >= 2500:
                done = True

            # --- Nav-label validation (read-only; does not alter saved data) ---
            # Cross-check our computed turn command against MetaDrive's own
            # post-step info["navigation_command"]. A non-zero mismatch rate on
            # turns means the label is flipped/mis-decoded and CIL conditioning
            # would train on wrong targets.
            md_cmd = info.get("navigation_command", None)
            if md_cmd is not None:
                nav_key_present += 1
                expected = _MD_CMD_TO_ONEHOT.get(str(md_cmd).lower())
                if expected is not None:
                    nav_recognized += 1
                    ours = navigation_command_onehot(env.agent)
                    if tuple(float(v) for v in ours) != expected:
                        nav_mismatch += 1

            if fps_counter.total_steps % 100 == 0:
                print(f"  [Worker {worker_id} | Ep {ep+1}/{num_episodes}] "
                      f"Step: {fps_counter.total_steps:5d}  |  "
                      f"Avg FPS: {fps_counter.average_fps:5.1f}  |  "
                      f"Ego: spd={reading.total_speed:+.2f} str={reading.last_steer:+.2f}",
                      flush=True)

            if visualize:
                ego_info = {
                    "speed":         reading.total_speed,
                    "steer":         reading.last_steer,
                    "heading_delta": reading.heading_delta,
                }
                quit_requested = show_cameras(
                    raw_frames, rgb_cam_names, depth_cam_names, fps_counter,
                    ego_info=ego_info,
                    poster_path=poster_path,
                )
                if quit_requested:
                    done = True

        save_threads = []

        # ── Main episode (RGB/depth/action/ego/nav) ──────────────────────────────
        save_dict = {
            "action":           np.array(actions),
            "ego_state":        np.array(ego_states,      dtype=np.float32),
            "ego_state_full":   np.array(ego_states_full, dtype=np.float32),
            "navi_state":       np.array(navi_states,     dtype=np.float32),  # (N, NAVI_DIM)
            "frame_timestamps": np.array(frame_timestamps, dtype=np.float64),
        }
        for rgb_name in rgb_cam_names:
            depth_list = observations[f"{rgb_name}_depth"]
            rgb_list = observations[f"{rgb_name}_rgb"]

            save_dict[f"{rgb_name}_depth"] = np.array(depth_list, dtype=np.float32)
            save_dict[f"{rgb_name}_rgb"] = np.array(rgb_list, dtype=np.uint8)

        td_str = (f"td{traffic_density_min:.2f}"
                  if traffic_density_min == traffic_density_max
                  else f"td{traffic_density_min:.2f}-{traffic_density_max:.2f}")
        ep_prefix = f"s{seed}_n{action_noise:.2f}_{td_str}_dr{decision_repeat}"
        save_path = os.path.join(save_dir, current_split, f"{ep_prefix}_ep{global_ep_id}.npz")
        main_t = threading.Thread(
            target=lambda p, d: np.savez_compressed(p, **d),
            args=(save_path, save_dict),
            daemon=True,
        )
        main_t.start()
        save_threads.append(main_t)

        print(f"\n  --> [Worker {worker_id}] Saving Global Episode {global_ep_id} to '{current_split}'")

        ep += 1
        for t in save_threads:
            t.join()  # wait for saves to finish before the next heavy render

    if visualize:
        import cv2
        cv2.destroyAllWindows()

    # ── Nav-label validation summary ──────────────────────────────────────
    if nav_key_present == 0:
        print(f"\n  [Worker {worker_id}] [NAV-CHECK] info['navigation_command'] was never "
              f"present — could not validate the turn label. The saved navi_state relies "
              f"entirely on navigation_command_onehot(); verify it manually if conditioning "
              f"fails.", flush=True)
    elif nav_recognized == 0:
        print(f"\n  [Worker {worker_id}] [NAV-CHECK] navigation_command was present but never "
              f"matched a known value — extend _MD_CMD_TO_ONEHOT to validate.", flush=True)
    else:
        rate = nav_mismatch / nav_recognized * 100.0
        verdict = "OK — labels agree with MetaDrive." if rate < 1.0 else \
                  "MISMATCH — turn labels likely flipped/mis-decoded; fix before training!"
        print(f"\n  [Worker {worker_id}] [NAV-CHECK] {nav_mismatch}/{nav_recognized} "
              f"({rate:.2f}%) of recognized commands disagreed with our one-hot. {verdict}",
              flush=True)

    env.close()
    return worker_id, fps_counter.total_steps


# ============================================================
# PARALLEL ORCHESTRATOR
# ============================================================
def collect_expert_data_parallel(
    seed,
    num_episodes         = 10,
    num_workers          = 1,
    save_dir             = "dataset",
    visualize            = True,
    num_cameras          = 2,
    action_noise         = 0.3,
    image_on_cuda        = True,
    split_ratios         = (0.8, 0.1, 0.1),
    decision_repeat      = 5,
    save_every_n         = 20,
    poster_path          = None,
    traffic_density_min  = 0.1,
    traffic_density_max  = 0.7,
):
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(save_dir, split), exist_ok=True)

    mode_str = "GPU (CUDA)" if image_on_cuda else "CPU (NumPy)"

    # Auto-disable visualization if running multiple workers
    if num_workers > 1 and visualize:
        print("\n[WARNING] Visualization is disabled because multiple workers are running.")
        visualize = False

    print(f"\n{'='*55}")
    print(f"  Parallel Processing : {num_workers} Workers")
    print(f"  Total Episodes  : {num_episodes}")
    print(f"  Processing Mode : {mode_str}")
    print(f"  Saving to       : '{save_dir}'")
    print(f"  Camera Count    : {num_cameras}")
    print(f"  Sim FPS         : {100 // decision_repeat} Hz  (decision_repeat={decision_repeat}, save_every_n={save_every_n})")
    density_str = (f"{traffic_density_min:.2f}" if traffic_density_min == traffic_density_max
                   else f"random [{traffic_density_min:.2f}, {traffic_density_max:.2f}]")
    print(f"  Traffic Density : {density_str}")
    print(f"{'='*55}\n")

    # Chunk the episodes for each worker
    episodes_per_worker = [num_episodes // num_workers] * num_workers
    for i in range(num_episodes % num_workers):
        episodes_per_worker[i] += 1

    worker_args = []
    current_idx = 0
    
    for i in range(num_workers):
        worker_eps = episodes_per_worker[i]
        if worker_eps == 0:
            continue
            
        worker_args.append((
            i,                          # worker_id
            current_idx,                # start_ep_idx
            worker_eps,                 # num_episodes
            num_episodes,               # total_episodes
            seed + (i * 1000),          # seed (offset so workers generate distinct maps)
            save_dir,                   # save_dir
            action_noise,
            visualize,                  # visualize
            num_cameras,                # num_cameras
            image_on_cuda,              # image_on_cuda
            split_ratios,               # split_ratios
            decision_repeat,            # decision_repeat
            save_every_n,               # save_every_n
            poster_path,                # poster_path
            traffic_density_min,        # traffic_density_min
            traffic_density_max,        # traffic_density_max
        ))
        current_idx += worker_eps

    # Run processes
    if num_workers == 1:
        # Run sequentially if only 1 worker is requested (helpful for debugging)
        _worker_collect(*worker_args[0])
    else:
        with mp.Pool(num_workers) as pool:
            pool.starmap(_worker_collect, worker_args)

    print(f"\n{'='*55}")
    print(f"  DATA COLLECTION COMPLETED ACROSS {num_workers} WORKERS")
    print(f"{'='*55}\n")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Required for multiprocessing with PyTorch/CUDA and heavy visual contexts
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="MetaDrive Multi-Cam Imitation Learning")
    parser.add_argument("--episodes",      type=int,  default=10)
    parser.add_argument("--num_workers",   type=int,  default=1, help="Number of parallel processes")
    parser.add_argument("--save_dir",      type=str,  default="dataset")
    parser.add_argument("--start_seed",    type=int,  default=42)
    parser.add_argument("--num_cameras",   type=int,  default=1)
    parser.add_argument("--act_noise",   type=float,  default=0.3)
    parser.add_argument("--no_vis",        action="store_true")
    parser.add_argument("--image_on_cuda",    action="store_true", default=False)
    parser.add_argument("--decision_repeat",  type=int, default=1,
                        help="Physics steps per decision (default 1 = 100Hz)")
    parser.add_argument("--save_every_n",     type=int, default=20,
                        help="Save 1 frame every N steps (default 20)")
    parser.add_argument("--poster",        type=str, default=None,
                        metavar="PATH",
                        help="Save a high-res poster PNG to PATH, then exit")
    parser.add_argument("--traffic_density", type=float, default=None,
                        help="Fixed traffic density [0, 1]. Omit to use random range.")
    parser.add_argument("--traffic_density_min", type=float, default=0.1,
                        help="Min traffic density for random range (default 0.1)")
    parser.add_argument("--traffic_density_max", type=float, default=0.7,
                        help="Max traffic density for random range (default 0.7)")
    args = parser.parse_args()

    td_min = args.traffic_density if args.traffic_density is not None else args.traffic_density_min
    td_max = args.traffic_density if args.traffic_density is not None else args.traffic_density_max

    collect_expert_data_parallel(
        seed                = args.start_seed,
        num_episodes        = args.episodes,
        num_workers         = args.num_workers,
        save_dir            = args.save_dir,
        action_noise        = args.act_noise,
        decision_repeat     = args.decision_repeat,
        save_every_n        = args.save_every_n,
        visualize           = not args.no_vis,
        num_cameras         = args.num_cameras,
        image_on_cuda       = args.image_on_cuda,
        poster_path         = args.poster,
        traffic_density_min = td_min,
        traffic_density_max = td_max,
    )