"""
train_test_policy.py – train, fine-tune, and test the DrivingPolicyNet.

Usage
-----
# Train from scratch (precomputed depth)
python src/train_test_policy.py --mode train \
    --pred_dir data/processed/dpt_pred \
    --model_path models/policy_model.pth

# Fine-tune a saved checkpoint
python src/train_test_policy.py --mode finetune \
    --finetune_from models/policy_model_best.pth \
    --model_path models/policy_finetuned.pth \
    --pred_dir data/processed/dpt_pred \
    --epochs 10 --lr 2e-5 \
    --freeze_backbone

# Test (offline + simulation)
python src/train_test_policy.py --mode test \
    --model_path models/policy_model_best.pth \
    --pred_dir data/processed/dpt_pred
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera

from models import DepthEstimationModel, DrivingPolicyNet, extract_ego_state, EGO_DIM
from policy.datasets import PrecomputedDepthDataset, MetaDriveRGBDataset
from policy.losses import custom_driving_loss, compute_offline_metrics
from policy.trainer import build_loaders, train_loop, extract_features_frozen
from data.cameras import build_cameras
from utils.checkpoints import (
    save_checkpoint, load_checkpoint, freeze_backbone, print_trainable_params
)


# ============================================================
# 1. TRAIN FROM SCRATCH
# ============================================================
def train_policy(
    epochs:     int   = 20,
    batch_size: int   = 64,
    model_path: str   = "models/policy_model.pth",
    dpt_path:   str   = None,
    data_dir:   str   = "data/raw",
    lr:         float = 1e-4,
    pred_dir:   str   = None,
):
    print("--- Phase 2: Training Driving Policy (from scratch) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_precomputed = pred_dir is not None and os.path.isdir(os.path.join(pred_dir, "train"))
    depth_estimator = None if use_precomputed else DepthEstimationModel(finetuned_path=dpt_path)

    print(f"[INFO] {'Using PRECOMPUTED DPT from: ' + pred_dir if use_precomputed else 'Live DPT inference.'}")

    train_loader, val_loader = build_loaders(use_precomputed, pred_dir, data_dir, batch_size, depth_estimator)

    policy_model = DrivingPolicyNet().to(device)
    optimizer    = optim.AdamW(policy_model.parameters(), lr=lr)
    scheduler    = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*10**-2)

    train_loop(
        policy_model, device, train_loader, val_loader,
        optimizer, scheduler,
        epochs=epochs, start_epoch=0, best_val_loss=float("inf"),
        model_path=model_path, use_precomputed=use_precomputed,
        depth_estimator=depth_estimator, tag="Train",
    )


# ============================================================
# 2. FINE-TUNE
# ============================================================
def finetune_policy(
    finetune_from:   str,
    epochs:          int   = 10,
    batch_size:      int   = 32,
    model_path:      str   = "models/policy_finetuned.pth",
    dpt_path:        str   = None,
    data_dir:        str   = "data/raw",
    lr:              float = 2e-5,
    pred_dir:        str   = None,
    freeze_bb:       bool  = False,
    reset_optimizer: bool  = False,
    resume:          bool  = False,
):
    """
    Fine-tune (or resume) a previously saved policy model.

    freeze_bb       : freeze CNN/backbone layers; only train the head
    reset_optimizer : ignore saved optimizer/scheduler (good for new datasets)
    resume          : restore optimizer & scheduler to continue seamlessly
    """
    print("--- Fine-tuning Driving Policy ---")
    print(f"    Source : {finetune_from}  |  Output : {model_path}")
    print(f"    Epochs : {epochs}  |  LR : {lr}  |  Freeze backbone : {freeze_bb}")
    print(f"    Resume optimizer : {resume and not reset_optimizer}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_precomputed = pred_dir is not None and os.path.isdir(os.path.join(pred_dir, "train"))
    depth_estimator = None if use_precomputed else DepthEstimationModel(finetuned_path=dpt_path)

    train_loader, val_loader = build_loaders(use_precomputed, pred_dir, data_dir, batch_size, depth_estimator)

    policy_model = DrivingPolicyNet().to(device)
    if freeze_bb:
        freeze_backbone(policy_model)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, policy_model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 1e-2)

    restore_opt              = resume and not reset_optimizer
    start_epoch, best_val   = load_checkpoint(
        finetune_from, policy_model,
        optimizer = optimizer if restore_opt else None,
        scheduler = scheduler if restore_opt else None,
        device    = device,
    )

    if not restore_opt:
        start_epoch = 0
        best_val    = float("inf")
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        print(f"[INFO] Optimizer reset. Fine-tuning from epoch 0, LR={lr}.")

    print_trainable_params(policy_model)

    train_loop(
        policy_model, device, train_loader, val_loader,
        optimizer, scheduler,
        epochs=epochs, start_epoch=start_epoch, best_val_loss=best_val,
        model_path=model_path, use_precomputed=use_precomputed,
        depth_estimator=depth_estimator, tag="Finetune",
    )


# ============================================================
# 3. TEST
# ============================================================
def test_policy(
    model_path:   str,
    dpt_path:     str,
    data_dir:     str,
    num_episodes: int,
    pred_dir:     str  = None,
    test_mode:    str  = "all",
):
    print("--- Phase 3: Testing Driving Policy ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_model = DrivingPolicyNet().to(device)
        
    ckpt         = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict):
        key = "model" if "model" in ckpt else ("policy" if "policy" in ckpt else None)
        policy_model.load_state_dict(ckpt[key] if key else ckpt)
    else:
        policy_model.load_state_dict(ckpt)
    policy_model.eval()

    depth_estimator = None

    # ── OFFLINE ─────────────────────────────────────────────────────────────
    if test_mode in ("offline", "all"):
        print("\n=> Offline Evaluation on Test Split...")
 
        use_precomputed = pred_dir is not None and os.path.isdir(os.path.join(pred_dir, "test"))
 
        if use_precomputed:
            from policy.trainer import apply_lane_mask
            test_ds = PrecomputedDepthDataset(pred_dir=pred_dir, split="test")
            def collate_fn(batch):
                depths, rgbs, actions, egos = zip(*batch)
                return (
                    torch.tensor(np.stack(depths), dtype=torch.float32),
                    np.stack(rgbs),
                    np.stack(actions),
                    np.stack(egos),
                )
        else:
            depth_estimator = DepthEstimationModel(finetuned_path=dpt_path)
            test_ds = MetaDriveRGBDataset(data_dir=data_dir, split="test")
            def collate_fn(batch):
                rgbs, actions, egos = zip(*batch)
                return np.stack(rgbs), np.stack(actions), np.stack(egos)
 
        if len(test_ds) > 0:
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
            test_loss   = 0.0
            test_pred, test_true = [], []
 
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Testing"):
                    if use_precomputed:
                        depth_t, rgb_np, actions_np, ego_np = batch
                        depth_t   = depth_t.to(device)
                        actions_t = torch.tensor(actions_np, dtype=torch.float32, device=device)
                        ego_t     = torch.tensor(ego_np,     dtype=torch.float32, device=device)
                        combined  = apply_lane_mask(depth_t, rgb_np, device)
                    else:
                        rgb_np, actions_np, ego_np = batch
                        actions_t   = torch.tensor(actions_np, dtype=torch.float32, device=device)
                        combined, _ = extract_features_frozen(rgb_np, depth_estimator, device)
                        ego_t       = torch.tensor(ego_np, dtype=torch.float32, device=device)
 
                    pred       = policy_model(combined, ego_t)
                    test_loss += custom_driving_loss(pred, actions_t).item()
                    test_pred.append(pred.cpu().numpy())
                    test_true.append(actions_np)

    # ── SIMULATION ──────────────────────────────────────────────────────────
    if test_mode in ("simulation", "all"):
        print("\n=> Online Evaluation (Simulation)...")
        if depth_estimator is None:
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
            "start_seed":        316181,
        }
        env = MetaDriveEnv(config)
        success_flags, route_completions = [], []

        for ep in range(num_episodes):
            obs, info  = env.reset()
            done       = False
            step_count = 0
            anlik_fps  = 0.0
            last_time  = time.time()
            last_steer = 0.0

            while not done:
                step_count += 1
                rgb_img = env.engine.get_sensor(rgb_name).perceive(
                    to_float=False, new_parent_node=env.agent.origin
                )
                if hasattr(rgb_img, "get"):
                    rgb_img = rgb_img.get()
                rgb_img = np.array(rgb_img, dtype=np.uint8)

                combined_tensor, _ = extract_features_frozen(rgb_img[np.newaxis], depth_estimator, device)

                ego_reading = extract_ego_state(env.agent, last_steer=last_steer)
                ego_t       = torch.tensor(ego_reading.ego_model, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    pred_action = policy_model(combined_tensor, ego_t).cpu().numpy()[0]
                last_steer = float(pred_action[0])

                # HUD visualisation
                depth_uint8   = (combined_tensor[0, 0].cpu().numpy() * 255).astype(np.uint8)
                lane_uint8    = (combined_tensor[0, 1].cpu().numpy() * 255).astype(np.uint8)
                depth_color   = cv2.resize(cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO), (400, 400))
                lane_color    = cv2.resize(cv2.cvtColor(lane_uint8, cv2.COLOR_GRAY2BGR), (400, 400))
                hud           = np.zeros((40, 800, 3), dtype=np.uint8)
                cv2.putText(
                    hud,
                    f"spd:{ego_reading.total_speed:+.2f}  fwd:{ego_reading.forward_speed:+.2f}  "
                    f"lat:{ego_reading.lateral_speed:+.2f}  hdg:{ego_reading.heading_delta:+.2f}  "
                    f"str:{ego_reading.last_steer:+.2f}  "
                    f"->  steer:{pred_action[0]:+.2f}  throt:{pred_action[1]:+.2f}",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1,
                )
                cv2.imshow("Depth | Lane  (with Ego HUD)",
                           np.vstack((hud, np.hstack((depth_color, lane_color)))))
                cv2.waitKey(1)
                obs, reward, terminated, truncated, info = env.step(pred_action)
                done = terminated or truncated

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
            print(f"\nEpisode {ep+1} done. Success: {success_flags[-1]}")

        print(f"\n=== ONLINE SUMMARY ===\n"
              f"Success: {np.mean(success_flags)*100:.1f}%  "
              f"Route: {np.mean(route_completions)*100:.1f}%")
        env.close()
        cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       type=str, required=True,
                        choices=["train", "finetune", "test", "all"])
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--episodes",   type=int,   default=1)
    parser.add_argument("--data_dir",   type=str,   default="dataset")
    parser.add_argument("--dpt_path",   type=str,   default="models/dpt_finetuned.pth")
    parser.add_argument("--model_path", type=str,   default="models/policy_model.pth")
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--pred_dir",   type=str,   default=None)
    parser.add_argument("--test_mode",  type=str,   default="all",
                        choices=["offline", "simulation", "all"])
    # fine-tune args
    parser.add_argument("--finetune_from",  type=str, default=None)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--reset_optimizer", action="store_true")
    parser.add_argument("--resume",          action="store_true")
    args = parser.parse_args()

    if args.mode in ("train", "all"):
        train_policy(args.epochs, 32, args.model_path,
                     args.dpt_path, args.data_dir, args.lr, pred_dir=args.pred_dir)

    if args.mode == "finetune":
        if args.finetune_from is None:
            parser.error("--finetune_from is required when --mode finetune")
        finetune_policy(
            finetune_from   = args.finetune_from,
            epochs          = args.epochs,
            batch_size      = 32,
            model_path      = args.model_path,
            dpt_path        = args.dpt_path,
            data_dir        = args.data_dir,
            lr              = args.lr if args.lr != 1e-4 else 1e-6,
            pred_dir        = args.pred_dir,
            freeze_bb       = args.freeze_backbone,
            reset_optimizer = args.reset_optimizer,
            resume          = args.resume,
        )

    if args.mode in ("test", "all"):
        test_policy(args.model_path, args.dpt_path, args.data_dir,
                    args.episodes, pred_dir=args.pred_dir, test_mode=args.test_mode)