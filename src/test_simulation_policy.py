"""
test_simulation_policy.py – Simulation testing script utilizing MetaDrive and UI visualization.
"""

import argparse
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from metadrive import MetaDriveEnv

from models import DepthEstimationModel, DrivingPolicyNet, DrivingPolicyNet2, extract_ego_state
from policy.trainer import extract_features_frozen
from data.cameras import build_cameras

def test_simulation(
    model_path:   str,
    dpt_path:     str,
    num_episodes: int,
    image_size:   int  = None,
    policy:       str  = "standard",
):
    print("--- Online Evaluation (Simulation) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cls    = DrivingPolicyNet2 if policy == "deep" else DrivingPolicyNet
    policy_model = model_cls(image_size=image_size).to(device)
    print(f"[INFO] Policy network: {model_cls.__name__}")
        
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict):
        key = "model" if "model" in ckpt else ("policy" if "policy" in ckpt else None)
        policy_model.load_state_dict(ckpt[key] if key else ckpt)
    else:
        policy_model.load_state_dict(ckpt)
    policy_model.eval()

    depth_estimator = DepthEstimationModel(finetuned_path=dpt_path)

    angles, sensors, rgb_cam_names, depth_cam_names = build_cameras(1)
    rgb_name = rgb_cam_names[0]

    config = {
        "use_render":        True,
        "image_observation": True,
        "sensors":           {rgb_name: sensors[rgb_name]},
        "vehicle_config":    {"image_source": rgb_name},
        "show_interface":    False,
        "image_on_cuda":     False,
        "start_seed":        316182,
    }
    
    env = MetaDriveEnv(config)
    success_flags, route_completions = [], []
    out_of_roads, crash_vehicles, crash_objects = [], [], []
    survival_times, average_speeds = [], []

    for ep in range(num_episodes):
        obs, info  = env.reset()
        done       = False
        step_count = 0
        anlik_fps  = 0.0
        last_time  = time.time()
        last_steer = 0.0
        speeds     = []

        while not done:
            step_count += 1
            rgb_img = env.engine.get_sensor(rgb_name).perceive(
                to_float=False, new_parent_node=env.agent.origin
            )
            if hasattr(rgb_img, "get"):
                rgb_img = rgb_img.get()
            rgb_img = np.array(rgb_img, dtype=np.uint8)
            # MetaDrive RGBCamera natively returns BGR, so we convert it to RGB
            rgb_img = rgb_img[..., ::-1].copy()

            combined_tensor, _ = extract_features_frozen(rgb_img[np.newaxis], depth_estimator, image_size=image_size, device=device)

            ego_reading = extract_ego_state(env.agent, last_steer=last_steer)
            ego_t       = torch.tensor(ego_reading.ego_model, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                pred_action = policy_model(combined_tensor, ego_t).cpu().numpy()[0]
            last_steer = float(pred_action[0])

            # HUD visualisation
            depth_uint8   = (combined_tensor[0, 0].cpu().numpy() * 255).astype(np.uint8)
            
            blended_rgb   = combined_tensor[0, 1:4].cpu().numpy() 
            blended_rgb   = np.transpose(blended_rgb, (1, 2, 0))  
            blended_uint8 = (blended_rgb * 255).astype(np.uint8)
            blended_bgr   = cv2.cvtColor(blended_uint8, cv2.COLOR_RGB2BGR)
            
            depth_color   = cv2.resize(cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO), (400, 400))
            rgb_color     = cv2.resize(blended_bgr, (400, 400))
            
            hud           = np.zeros((40, 800, 3), dtype=np.uint8)
            cv2.putText(
                hud,
                f"spd:{ego_reading.total_speed:+.2f}  fwd:{ego_reading.forward_speed:+.2f}  "
                f"lat:{ego_reading.lateral_speed:+.2f}  hdg:{ego_reading.heading_delta:+.2f}  "
                f"str:{ego_reading.last_steer:+.2f}  "
                f"->  steer:{pred_action[0]:+.2f}  throt:{pred_action[1]:+.2f}",
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1,
            )
            cv2.imshow("Depth | RGB  (with Ego HUD)", np.vstack((hud, np.hstack((depth_color, rgb_color)))))
            cv2.waitKey(1)
            
            obs, reward, terminated, truncated, info = env.step(pred_action)
            done = terminated or truncated
            
            if not done:
                speeds.append(ego_reading.total_speed)

            cur_time  = time.time()
            elapsed   = cur_time - last_time
            last_time = cur_time
            if elapsed > 0:
                anlik_fps = 0.9 * anlik_fps + 0.1 * (1.0 / elapsed)
            print(f"EP: {ep+1} | Steer: {pred_action[0]:.2f} | "
                  f"Throttle {pred_action[1]:.2f} | "
                  f"Spd: {ego_reading.total_speed:+.2f} | FPS: {anlik_fps:.1f}",
                  end="\r")

        success_flags.append(bool(info.get("arrive_dest", False)))
        route_completions.append(info.get("route_completion", 0.0))
        out_of_roads.append(bool(info.get("out_of_road", False)))
        crash_vehicles.append(bool(info.get("crash_vehicle", False)))
        crash_objects.append(bool(info.get("crash_object", False)))
        survival_times.append(step_count)
        average_speeds.append(float(np.mean(speeds)) if len(speeds) > 0 else 0.0)
        
        reason = "success" if success_flags[-1] else (
            "out_of_road" if out_of_roads[-1] else (
                "crash_vehicle" if crash_vehicles[-1] else (
                    "crash_object" if crash_objects[-1] else "timeout/other"
                )
            )
        )
        print(f"\nEpisode {ep+1} done. Reason: {reason} | Route: {route_completions[-1]*100:.1f}% | Avg Spd: {average_speeds[-1]:.2f}")

    print(f"\n=== ONLINE SUMMARY ===\n"
          f"Success Rate:         {np.mean(success_flags)*100:.1f}%\n"
          f"Route Completion:     {np.mean(route_completions)*100:.1f}%\n"
          f"Out of Road Rate:     {np.mean(out_of_roads)*100:.1f}%\n"
          f"Crash Vehicle Rate:   {np.mean(crash_vehicles)*100:.1f}%\n"
          f"Crash Object Rate:    {np.mean(crash_objects)*100:.1f}%\n"
          f"Avg Survival Time:    {np.mean(survival_times):.1f} steps\n"
          f"Avg Driving Speed:    {np.mean(average_speeds):.2f}")
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",   type=int,   default=1)
    parser.add_argument("--dpt_path",   type=str,   default="models/dpt_finetuned.pth")
    parser.add_argument("--model_path", type=str,   default="models/policy_model.pth")
    parser.add_argument("--image_size", type=int,   default=84)
    parser.add_argument("--policy",     type=str,   default="standard", choices=["standard", "deep"])
    args = parser.parse_args()

    test_simulation(args.model_path, args.dpt_path, args.episodes, image_size=args.image_size, policy=args.policy)