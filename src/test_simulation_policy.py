"""
test_simulation_policy.py – run a trained DrivingPolicyNet in the MetaDrive simulator.

Usage
-----
python src/test_simulation_policy.py \
    --model_path models/policy_model_best.pth \
    --dpt_path   models/dpt_finetuned.pth \
    --episodes   5
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.examples import expert

from models import DepthEstimationModel, build_policy, extract_ego_state
from policy.trainer import extract_features_frozen
from data.cameras import build_cameras
from utils.seed import seed_everything


STUCK_SPEED    = 0.05   # normalized total_speed (0-1) below which the car is considered stationary
STUCK_PATIENCE = 500    # consecutive steps below threshold before terminating


def run_simulation(
    model_path:         str,
    dpt_path:           str,
    num_episodes:       int   = 1,
    image_size:         int   = None,
    arch:               str   = "simple",
    always_lane_masked: bool  = False,
    seed:               int   = 42,
    steer_momentum:     float = 0.5,
    no_render:          bool  = False,
    max_steps:          int   = 1000,
    decision_repeat:    int   = 5,
    log_path:           str   = None,
    start_seed:         int   = 316181,
    map_config:         str   = None,
):
    print("--- Online Evaluation (Simulation) ---")
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_expert = arch == "expert"

    if not use_expert:
        policy_model = build_policy(arch, image_size).to(device)
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                # Standard IL checkpoint
                state = ckpt["model"]
            elif "policy" in ckpt:
                # RL checkpoint (ILActorCritic) — strip "il_model." prefix to
                # get back the raw IL backbone weights.
                state = {
                    k[len("il_model."):]: v
                    for k, v in ckpt["policy"].items()
                    if k.startswith("il_model.")
                }
                print(f"[INFO] RL checkpoint detected — extracted {len(state)} IL backbone tensors.")
            else:
                state = ckpt
        else:
            state = ckpt
        policy_model.load_state_dict(state)
        policy_model.eval()

    depth_estimator = None if use_expert else DepthEstimationModel(finetuned_path=dpt_path)

    angles, sensors, rgb_cam_names, _ = build_cameras(1)
    rgb_name = rgb_cam_names[0]

    config = {
        "use_render":        not no_render,
        "image_observation": True,
        "sensors":           {rgb_name: sensors[rgb_name]},
        "vehicle_config":    {"image_source": rgb_name, "show_navi_mark": False},
        "show_interface":    False,
        "image_on_cuda":     False,
        "start_seed":        start_seed,
        "num_scenarios":     num_episodes,
        "horizon":           max_steps,
        "decision_repeat":   decision_repeat,   # MUST match the value used in generate_expert_dataset.py
        "traffic_density":   0.15,
    }
    if map_config is not None:
        # Override procedural map with a fixed block-string layout.
        # Each char is a block type: S=Straight, C=Curve, T=TIntersection, X=XIntersection
        # Example: "SXTXSX" → straight, X-intersection, T-intersection repeated
        config["map"] = map_config
        # map_config overrides start_seed-based generation; num_scenarios still controls resets
        print(f"[INFO] Using fixed map layout: '{map_config}' (overrides seed-based generation)")
    print(f"[INFO] decision_repeat = {decision_repeat} (physics steps per action). "
          f"This MUST match data collection or the last_steer/heading feedback is miscalibrated.")
    env = MetaDriveEnv(config)

    success_flags, route_completions = [], []
    out_of_roads, crash_vehicles, crash_objects = [], [], []
    survival_times, average_speeds, jitter_rates, safety_scores = [], [], [], []

    log_data: dict = {
        "config": {
            "arch": arch,
            "model_path": model_path,
            "dpt_path": dpt_path,
            "seed": seed,
            "decision_repeat": decision_repeat,
            "max_steps": max_steps,
            "steer_ema_alpha": steer_momentum,
            "always_lane_masked": always_lane_masked,
        },
        "episodes": [],
    }

    for ep in range(num_episodes):
        obs, info  = env.reset(seed=start_seed + ep)
        done        = False
        step_count  = 0
        anlik_fps   = 0.0
        last_time   = time.time()
        last_steer  = 0.0
        speeds      = []
        steers      = []
        stuck_steps = 0
        stuck       = False

        step_log: list = []

        while not done:
            step_count += 1

            ego_reading = extract_ego_state(env.agent, last_steer=last_steer)

            if use_expert:
                pred_action = expert(env.agent, deterministic=True)
                pred_action[0] = steer_momentum * last_steer + (1.0 - steer_momentum) * pred_action[0]
                last_steer = float(pred_action[0])

                if not no_render:
                    hud = np.zeros((40, 400, 3), dtype=np.uint8)
                    cv2.putText(
                        hud,
                        f"[EXPERT]  spd:{ego_reading.total_speed:+.2f}  "
                        f"str:{pred_action[0]:+.2f}  throt:{pred_action[1]:+.2f}",
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1,
                    )
                    cv2.imshow("Expert Policy", hud)
                    cv2.waitKey(1)
            else:
                rgb_img = env.engine.get_sensor(rgb_name).perceive(
                    to_float=False, new_parent_node=env.agent.origin
                )
                if hasattr(rgb_img, "get"):
                    rgb_img = rgb_img.get()
                rgb_img = np.array(rgb_img, dtype=np.uint8)
                # MetaDrive RGBCamera natively returns BGR; convert to RGB
                rgb_img = rgb_img[..., ::-1].copy()

                combined_tensor, _ = extract_features_frozen(
                    rgb_img[np.newaxis], depth_estimator, image_size=image_size, device=device,
                    always_lane_masked=always_lane_masked,
                )

                ego_t = torch.tensor(ego_reading.ego_model, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    alpha, beta = policy_model(combined_tensor, ego_t)
                    mean_01     = alpha / (alpha + beta)  # Beta mean; exact for throttle (= mu), close to mode for steer
                    pred_action = (mean_01 * 2.0 - 1.0).cpu().numpy()[0]

                at_junction = bool(ego_reading.navi_left) or bool(ego_reading.navi_right)

                # Disable EMA at junctions so turns execute at full speed.
                # Smoothing a 0.8 steer command to 0.4→0.6→0.7 over 3 frames
                # means the car is already wide before it commits to the turn.
                ema = 0.0 if at_junction else steer_momentum
                pred_action[0] = ema * last_steer + (1.0 - ema) * pred_action[0]
                last_steer = float(pred_action[0])

                if not no_render:
                    depth_uint8  = (combined_tensor[0, 0].cpu().numpy() * 255).astype(np.uint8)
                    blended_rgb  = combined_tensor[0, 1:4].cpu().numpy()
                    blended_rgb  = np.transpose(blended_rgb, (1, 2, 0))
                    blended_uint8 = (blended_rgb * 255).astype(np.uint8)
                    blended_bgr  = cv2.cvtColor(blended_uint8, cv2.COLOR_RGB2BGR)

                    depth_color = cv2.resize(cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO), (400, 400))
                    rgb_color   = cv2.resize(blended_bgr, (400, 400))

                    nav_cmd = ("LEFT" if ego_reading.navi_left else
                               "RIGHT" if ego_reading.navi_right else "FWD")
                    hud = np.zeros((40, 800, 3), dtype=np.uint8)
                    cv2.putText(
                        hud,
                        f"spd:{ego_reading.total_speed:+.2f}  hdg:{ego_reading.heading_delta:+.2f}  "
                        f"str:{ego_reading.last_steer:+.2f}  nav:{nav_cmd:<5s}  "
                        f"->  steer:{pred_action[0]:+.2f}  throt:{pred_action[1]:+.2f}",
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1,
                    )
                    cv2.imshow("Depth | RGB  (with Ego HUD)",
                               np.vstack((hud, np.hstack((depth_color, rgb_color)))))
                    cv2.waitKey(1)

            obs, reward, terminated, truncated, info = env.step(pred_action)
            done = terminated or truncated

            if ego_reading.total_speed < STUCK_SPEED:
                stuck_steps += 1
            else:
                stuck_steps = 0
            if stuck_steps >= STUCK_PATIENCE:
                stuck = True
                done  = True

            steers.append(float(pred_action[0]))
            if not done:
                speeds.append(ego_reading.total_speed)

            if log_path:
                step_log.append({
                    "step":          step_count,
                    "steer":         round(float(pred_action[0]), 4),
                    "throttle":      round(float(pred_action[1]), 4),
                    "speed":         round(float(ego_reading.total_speed), 4),
                    "heading_delta": round(float(ego_reading.heading_delta), 4),
                    "last_steer":    round(float(ego_reading.last_steer), 4),
                    "navi_left":     bool(ego_reading.navi_left),
                    "navi_right":    bool(ego_reading.navi_right),
                    "reward":        round(float(reward), 4),
                })

            cur_time  = time.time()
            elapsed   = cur_time - last_time
            last_time = cur_time
            if elapsed > 0:
                anlik_fps = 0.9 * anlik_fps + 0.1 * (1.0 / elapsed)
            print(
                f"EP: {ep+1} | Steer: {pred_action[0]:.2f} | "
                f"Throttle {pred_action[1]:.2f} | "
                f"Spd: {ego_reading.total_speed:+.2f} | FPS: {anlik_fps:.1f}",
                end="\r",
            )

        success_flags.append(bool(info.get("arrive_dest", False)))
        route_completions.append(info.get("route_completion", 0.0))
        out_of_roads.append(bool(info.get("out_of_road", False)))
        crash_vehicles.append(bool(info.get("crash_vehicle", False)))
        crash_objects.append(bool(info.get("crash_object", False)))
        survival_times.append(step_count)
        average_speeds.append(float(np.mean(speeds)) if speeds else 0.0)
        jitter_rates.append(float(np.mean(np.abs(np.diff(steers)))) if len(steers) > 1 else 0.0)
        safety_scores.append(
            0.35 * (not crash_vehicles[-1])
            + 0.35 * (not out_of_roads[-1])
            + 0.30 * max(0.0, 1.0 - jitter_rates[-1] / 0.3)
        )

        reason = "success" if success_flags[-1] else (
            "stuck"        if stuck              else (
            "out_of_road"  if out_of_roads[-1]  else (
            "crash_vehicle" if crash_vehicles[-1] else (
            "crash_object"  if crash_objects[-1] else "timeout/other"
        ))))
        print(f"\nEpisode {ep+1} done. Reason: {reason} | "
              f"Route: {route_completions[-1]*100:.1f}% | "
              f"Avg Spd: {average_speeds[-1]:.2f} | "
              f"Jitter: {jitter_rates[-1]:.4f} | "
              f"Safety: {safety_scores[-1]*100:.1f}%")

        if log_path:
            log_data["episodes"].append({
                "episode":        ep + 1,
                "termination":    reason,
                "success":        bool(success_flags[-1]),
                "route_pct":      round(route_completions[-1] * 100, 2),
                "avg_speed":      round(average_speeds[-1], 4),
                "jitter":         round(jitter_rates[-1], 4),
                "safety_score":   round(safety_scores[-1], 4),
                "survival_steps": step_count,
                "out_of_road":    bool(out_of_roads[-1]),
                "crash_vehicle":  bool(crash_vehicles[-1]),
                "crash_object":   bool(crash_objects[-1]),
                "steps":          step_log,
            })

    summary = {
        "success_rate_pct":    round(float(np.mean(success_flags)) * 100, 2),
        "route_completion_pct": round(float(np.mean(route_completions)) * 100, 2),
        "out_of_road_rate_pct": round(float(np.mean(out_of_roads)) * 100, 2),
        "crash_vehicle_rate_pct": round(float(np.mean(crash_vehicles)) * 100, 2),
        "crash_object_rate_pct":  round(float(np.mean(crash_objects)) * 100, 2),
        "avg_survival_steps":  round(float(np.mean(survival_times)), 2),
        "avg_driving_speed":   round(float(np.mean(average_speeds)), 4),
        "avg_steering_jitter": round(float(np.mean(jitter_rates)), 4),
        "safe_driving_score_pct": round(float(np.mean(safety_scores)) * 100, 2),
    }
    print(
        f"\n=== ONLINE SUMMARY ===\n"
        f"Success Rate:         {summary['success_rate_pct']:.1f}%\n"
        f"Route Completion:     {summary['route_completion_pct']:.1f}%\n"
        f"Out of Road Rate:     {summary['out_of_road_rate_pct']:.1f}%\n"
        f"Crash Vehicle Rate:   {summary['crash_vehicle_rate_pct']:.1f}%\n"
        f"Crash Object Rate:    {summary['crash_object_rate_pct']:.1f}%\n"
        f"Avg Survival Time:    {summary['avg_survival_steps']:.1f} steps\n"
        f"Avg Driving Speed:    {summary['avg_driving_speed']:.2f}\n"
        f"Avg Steering Jitter:  {summary['avg_steering_jitter']:.4f}\n"
        f"Safe Driving Score:   {summary['safe_driving_score_pct']:.1f}%  "
        f"(35% collision-free + 35% road-adherence + 30% steering-smoothness)"
    )

    if log_path:
        log_data["summary"] = summary
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"\n[LOG] Saved simulation log → {log_path}")

    env.close()
    if not no_render:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,   default=None,
                        help="Path to trained policy checkpoint. Not required when --arch expert.")
    parser.add_argument("--dpt_path",   type=str,   default="models/dpt_finetuned_ep17.pth")
    parser.add_argument("--episodes",   type=int,   default=1)
    parser.add_argument("--image_size",         type=int,   default=84)
    parser.add_argument("--arch",               type=str,   default="simple",
                        choices=["simple", "impala", "impala_v2", "expert"])
    parser.add_argument("--always_lane_masked", action="store_true",
                        help="Force alpha=0 (fully lane-masked) during simulation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed for reproducibility")
    parser.add_argument("--steer_momentum", type=float, default=0.5,
                        help="Steering EMA weight on straight roads (0=off, 0.5=default). "
                             "Automatically set to 0 at junctions so turns execute crisply. "
                             "Raise toward 0.7 for more damping on straights.")
    parser.add_argument("--no_render", action="store_true",
                        help="Disable MetaDrive window and cv2 HUD (headless/server mode)")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Hard episode step limit (MetaDrive horizon). "
                             "Episodes also end early if speed < 1 km/h for 150 consecutive steps.")
    parser.add_argument("--decision_repeat", type=int, default=5,
                        help="Physics steps per action. MUST match the value used during data "
                             "collection (generate_expert_dataset.py). A mismatch miscalibrates "
                             "the last_steer/heading_delta feedback and causes a slow steering weave.")
    parser.add_argument("--log_path", type=str, default=None,
                        help="If set, save a JSON log of every step and episode summary to this path. "
                             "Example: logs/sim_run.json")
    parser.add_argument("--start_seed", type=int, default=316181,
                        help="MetaDrive scenario start seed. Episode i uses seed start_seed+i. "
                             "Use find_curvy_seeds.py to discover seeds with lots of turns.")
    parser.add_argument("--map", type=str, default=None, dest="map_config",
                        help="Fixed map block string, e.g. 'SXTXSX'. "
                             "S=Straight C=Curve T=TIntersection X=XIntersection. "
                             "Overrides seed-based procedural generation. "
                             "Recommended for 90-degree turn testing: 'STSTXTXTXT'")
    args = parser.parse_args()
    if args.arch != "expert" and args.model_path is None:
        parser.error("--model_path is required unless --arch expert")

    run_simulation(
        model_path          = args.model_path,
        dpt_path            = args.dpt_path,
        num_episodes        = args.episodes,
        image_size          = args.image_size,
        arch                = args.arch,
        always_lane_masked  = args.always_lane_masked,
        seed                = args.seed,
        steer_momentum      = args.steer_momentum,
        no_render           = args.no_render,
        max_steps           = args.max_steps,
        decision_repeat     = args.decision_repeat,
        log_path            = args.log_path,
        start_seed          = args.start_seed,
        map_config          = args.map_config,
    )
