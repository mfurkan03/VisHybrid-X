""""
python src/train_dpt.py --mode train --epochs 5 --data_dir dataset --model_path models/dpt_finetuned.pth

# Step 2 – precompute & cache predictions ONCE (new)
python src/train_dpt.py --mode precompute \
    --model_path models/dpt_finetuned.pth \
    --data_dir dataset \
    --out_dir data/processed/dpt_pred

# Step 3 – train policy at full speed (no DPT inference per step)
python src/train_test_policy.py --mode train \
    --pred_dir data/processed/dpt_pred \
    --model_path models/policy_model.pth
"""

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera

import cv2
import time
import sys
from pathlib import Path
from tqdm import tqdm

target_folder = Path(__file__).resolve().parent.parent / 'Depth-Anything-V2'
sys.path.append(str(target_folder))
from depth_anything_v2.dpt import DepthAnythingV2


# ==========================================
# 1. HELPERS & FROZEN DEPTH MODEL
#    (kept for online / live inference only)
# ==========================================
def get_lane_mask_visual(rgb_image, threshold_value=180):
    if rgb_image.max() <= 1.0:
        img_uint8 = (rgb_image * 255.0).astype(np.uint8)
    else:
        img_uint8 = rgb_image.astype(np.uint8)
    h, w = img_uint8.shape[:2]
    roi_img = img_uint8.copy()
    roi_img[0:int(h * 0.55), :] = 0
    gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return mask


class DepthEstimationModel:
    """Frozen DPT wrapper – used only for online (simulation) inference."""

    def __init__(self, encoder='vits', finetuned_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        self.model = DepthAnythingV2(**model_configs[encoder])

        if finetuned_path and os.path.exists(finetuned_path):
            self.model.load_state_dict(torch.load(finetuned_path, map_location='cpu'))
            print(f"[INFO] Fine-tuned DPT loaded: {finetuned_path}")
        else:
            ckpt_path = f'{target_folder}/checkpoints/depth_anything_v2_{encoder}.pth'
            if os.path.exists(ckpt_path):
                self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
                print(f"[INFO] Base DepthAnythingV2 loaded: {ckpt_path}")

        self.model = self.model.to(self.device)
        self.use_fp16 = (self.device == 'cuda')

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def predict_batch(self, rgb_images: np.ndarray) -> torch.Tensor:
        results = []
        for rgb in rgb_images:
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if self.use_fp16:
                with torch.amp.autocast('cuda'):
                    depth_raw = self.model.infer_image(rgb_bgr)
            else:
                depth_raw = self.model.infer_image(rgb_bgr)

            d_tensor = torch.from_numpy(depth_raw).to(self.device).unsqueeze(0).unsqueeze(0)
            import torch.nn.functional as F
            depth_resized = F.interpolate(d_tensor, size=(84, 84), mode='bilinear', align_corners=False)
            d_min, d_max = depth_resized.min(), depth_resized.max()
            depth_norm = (depth_resized - d_min) / (d_max - d_min + 1e-6)
            results.append(depth_norm.float())

        return torch.cat(results, dim=0)


def extract_features_frozen(rgb_batch: np.ndarray,
                             depth_estimator: DepthEstimationModel,
                             device: torch.device) -> torch.Tensor:
    """Live inference path – used only during simulation testing."""
    with torch.no_grad():
        depth_tensors = depth_estimator.predict_batch(rgb_batch)

    lane_list = []
    for rgb in rgb_batch:
        mask = get_lane_mask_visual(rgb)
        lane_norm = (cv2.resize(mask, (84, 84)) / 255.0).astype(np.float32)
        lane_list.append(lane_norm)
    lane_tensor = torch.tensor(np.stack(lane_list, axis=0), device=device).unsqueeze(1)

    return torch.cat([depth_tensors, lane_tensor], dim=1)


# ==========================================
# 2. POLICY NETWORK
# ==========================================
class DrivingPolicyNet(nn.Module):
    def __init__(self, action_dim=2):
        super().__init__()
        self.steer_conv = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.steer_fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 1),
        )
        self.accel_conv = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.accel_fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 1),
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        depth_input = x[:, 0:1, :, :]
        lane_input  = x[:, 1:2, :, :]
        s_feat = self.flatten(self.steer_conv(lane_input))
        a_feat = self.flatten(self.accel_conv(depth_input))
        return torch.cat([self.steer_fc(s_feat), self.accel_fc(a_feat)], dim=1)


# ==========================================
# 3. DATASETS
# ==========================================
class MetaDriveRGBDataset(Dataset):
    """
    Original dataset that loads raw RGB frames.
    Used when no precomputed DPT predictions are available.
    """

    def __init__(self, data_dir, split="train"):
        split_dir = os.path.join(data_dir, split)
        self.files = glob.glob(os.path.join(split_dir, "*.npz"))
        self.rgb_frames, self.actions = [], []

        for f in self.files:
            try: data = np.load(f, allow_pickle=True)
            except Exception: continue

            rgb_keys = [k for k in data.files if k.endswith('_rgb')]
            if not rgb_keys or 'action' not in data.files: continue

            self.rgb_frames.extend(data[rgb_keys[0]])
            self.actions.extend(data['action'])

        print(f"[INFO] PolicyDataset-RGB ({split}): {len(self.actions)} samples loaded from {split_dir}.")

    def __len__(self): return len(self.actions)
    def __getitem__(self, idx):
        return self.rgb_frames[idx], np.array(self.actions[idx], dtype=np.float32)


class PrecomputedDepthDataset(Dataset):
    """
    Fast dataset that loads pre-cached DPT depth predictions and lane masks
    produced by ``train_dpt.py --mode precompute``.

    Expected file layout (one .npz per original episode):
        <pred_dir>/<split>/episode_N.npz
            depth_pred : float32  (N, 1, 84, 84)   – normalised depth [0,1]
            lane_mask  : float32  (N, 1, 84, 84)   – binary lane mask [0,1]
            action     : float32  (N, 2)            – [steer, accel]

    The dataset concatenates all episodes so the DataLoader sees a flat list
    of (combined_obs, action) pairs, identical to what the original live-
    inference pipeline produced – but without any forward pass overhead.
    """

    def __init__(self, pred_dir: str, split: str = "train"):
        split_dir  = os.path.join(pred_dir, split)
        self.files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))

        self.depth_frames: list[np.ndarray] = []
        self.lane_frames:  list[np.ndarray] = []
        self.actions:      list[np.ndarray] = []

        for f in self.files:
            try:
                data = np.load(f, allow_pickle=True)
            except Exception as e:
                print(f"[WARNING] Could not load {f}: {e}")
                continue

            if 'depth_pred' not in data.files or 'lane_mask' not in data.files or 'action' not in data.files:
                print(f"[WARNING] Missing keys in {f}, skipping.")
                continue

            depths  = data['depth_pred']   # (N,1,84,84)
            lanes   = data['lane_mask']    # (N,1,84,84)
            actions = data['action']       # (N,2)

            # Validate lengths match
            n = min(len(depths), len(lanes), len(actions))
            self.depth_frames.extend(depths[:n])
            self.lane_frames.extend(lanes[:n])
            self.actions.extend(actions[:n])

        print(
            f"[INFO] PrecomputedDepthDataset ({split}): "
            f"{len(self.actions)} samples loaded from {split_dir}."
        )

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        # Stack depth + lane into a (2, 84, 84) tensor – matching DrivingPolicyNet input
        depth  = self.depth_frames[idx]   # (1,84,84) float32
        lane   = self.lane_frames[idx]    # (1,84,84) float32
        combined = np.concatenate([depth, lane], axis=0)   # (2,84,84)
        action = np.array(self.actions[idx], dtype=np.float32)
        return combined, action


# ==========================================
# 4. LOSS & METRICS
# ==========================================
def custom_driving_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (pred - target) ** 2
    brake_mask     = (target[:, 1] < 0.0).float()
    penalty_weight = 1.0 + brake_mask * 2.0
    mse_weighted   = mse.clone()
    mse_weighted[:, 1] = mse[:, 1] * penalty_weight
    return mse_weighted.mean()


def compute_offline_metrics(pred_actions, true_actions):
    steering_mse  = float(np.mean((pred_actions[:, 0] - true_actions[:, 0]) ** 2))
    accel_mse     = float(np.mean((pred_actions[:, 1] - true_actions[:, 1]) ** 2))
    corr, _       = pearsonr(pred_actions[:, 0], true_actions[:, 0])
    direction_acc = float(np.mean(np.sign(pred_actions[:, 1]) == np.sign(true_actions[:, 1])))
    return {"steering_mse": steering_mse, "accel_mse": accel_mse,
            "steering_corr": float(corr), "direction_acc": direction_acc}


# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train_policy(
    epochs:     int  = 20,
    batch_size: int  = 64,
    model_path: str  = "policy_model.pth",
    dpt_path:   str  = None,
    data_dir:   str  = "data/raw",
    lr:         float = 1e-4,
    pred_dir:   str  = None,   # path to precomputed DPT predictions
):
    """
    Train the driving policy.

    If ``pred_dir`` is provided (and contains the expected split sub-folders),
    the trainer uses ``PrecomputedDepthDataset`` – no DPT inference at all.
    Otherwise it falls back to live DPT inference via ``MetaDriveRGBDataset``.
    """
    print("--- Phase 2: Training Driving Policy ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Choose dataset mode ----
    use_precomputed = (
        pred_dir is not None
        and os.path.isdir(os.path.join(pred_dir, "train"))
    )

    if use_precomputed:
        print(f"[INFO] Using PRECOMPUTED DPT predictions from: {pred_dir}")
        train_ds = PrecomputedDepthDataset(pred_dir=pred_dir, split="train")
        val_ds   = PrecomputedDepthDataset(pred_dir=pred_dir, split="val")
        depth_estimator = None   # not needed

        def collate_fn(batch):
            combined, actions = zip(*batch)
            return (
                torch.tensor(np.stack(combined), dtype=torch.float32),
                np.stack(actions),
            )

    else:
        print("[INFO] No precomputed predictions found – using live DPT inference.")
        depth_estimator = DepthEstimationModel(finetuned_path=dpt_path)
        train_ds = MetaDriveRGBDataset(data_dir=data_dir, split="train")
        val_ds   = MetaDriveRGBDataset(data_dir=data_dir, split="val")

        def collate_fn(batch):
            rgbs, actions = zip(*batch)
            return np.stack(rgbs), np.stack(actions)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    policy_model = DrivingPolicyNet().to(device)
    optimizer    = optim.AdamW(policy_model.parameters(), lr=lr)
    scheduler    = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        policy_model.train()
        train_loss = 0.0
        all_pred, all_true = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            if use_precomputed:
                # batch = (combined_tensor, actions_np)
                combined_t, actions_np = batch
                combined   = combined_t.to(device)
                actions_t  = torch.tensor(actions_np, dtype=torch.float32, device=device)
            else:
                # batch = (rgb_np, actions_np)
                rgb_np, actions_np = batch
                actions_t = torch.tensor(actions_np, dtype=torch.float32, device=device)
                combined  = extract_features_frozen(rgb_np, depth_estimator, device)

            optimizer.zero_grad()
            pred = policy_model(combined)
            loss = custom_driving_loss(pred, actions_t)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            all_pred.append(pred.detach().cpu().numpy())
            all_true.append(actions_np)

        # ---- Validation ----
        policy_model.eval()
        val_loss = 0.0
        val_pred, val_true = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                if use_precomputed:
                    combined_t, actions_np = batch
                    combined  = combined_t.to(device)
                    actions_t = torch.tensor(actions_np, dtype=torch.float32, device=device)
                else:
                    rgb_np, actions_np = batch
                    actions_t = torch.tensor(actions_np, dtype=torch.float32, device=device)
                    combined  = extract_features_frozen(rgb_np, depth_estimator, device)

                pred = policy_model(combined)
                val_loss += custom_driving_loss(pred, actions_t).item()
                val_pred.append(pred.cpu().numpy())
                val_true.append(actions_np)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)

        tr_m  = compute_offline_metrics(np.concatenate(all_pred), np.concatenate(all_true))
        val_m = compute_offline_metrics(np.concatenate(val_pred),  np.concatenate(val_true))

        print(
            f"Epoch [{epoch+1:02d}/{epochs}] "
            f"Loss Tr/Val: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
            f"Steer MSE Tr/Val: {tr_m['steering_mse']:.4f}/{val_m['steering_mse']:.4f} | "
            f"Dir Acc: {val_m['direction_acc']:.3f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            file_root, file_ext = os.path.splitext(model_path)
            best_model_path = f"{file_root}_best{file_ext}"
            torch.save(policy_model.state_dict(), best_model_path)
            print(f"*** New best model saved to {best_model_path} (Val Loss: {best_val_loss:.4f}) ***")

        torch.save(policy_model.state_dict(), model_path)


# ==========================================
# 6. TESTING LOOP
# ==========================================
def test_policy(model_path, dpt_path, data_dir, num_episodes, pred_dir=None):
    print("--- Phase 3: Testing Driving Policy ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_model = DrivingPolicyNet().to(device)
    checkpoint   = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'policy' in checkpoint:
        policy_model.load_state_dict(checkpoint['policy'])
    else:
        policy_model.load_state_dict(checkpoint)
    policy_model.eval()

    # ---- Offline test ----
    print("\n=> Running Offline Evaluation on Test Split...")

    use_precomputed = (
        pred_dir is not None
        and os.path.isdir(os.path.join(pred_dir, "test"))
    )

    if use_precomputed:
        print(f"[INFO] Using precomputed predictions for offline test: {pred_dir}")
        test_ds = PrecomputedDepthDataset(pred_dir=pred_dir, split="test")
        depth_estimator = None

        def collate_fn(batch):
            combined, actions = zip(*batch)
            return (
                torch.tensor(np.stack(combined), dtype=torch.float32),
                np.stack(actions),
            )
    else:
        depth_estimator = DepthEstimationModel(finetuned_path=dpt_path)
        test_ds = MetaDriveRGBDataset(data_dir=data_dir, split="test")

        def collate_fn(batch):
            rgbs, actions = zip(*batch)
            return np.stack(rgbs), np.stack(actions)

    if len(test_ds) > 0:
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
        test_loss = 0.0
        test_pred, test_true = [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                if use_precomputed:
                    combined_t, actions_np = batch
                    combined  = combined_t.to(device)
                    actions_t = torch.tensor(actions_np, dtype=torch.float32, device=device)
                else:
                    rgb_np, actions_np = batch
                    actions_t = torch.tensor(actions_np, dtype=torch.float32, device=device)
                    combined  = extract_features_frozen(rgb_np, depth_estimator, device)

                pred = policy_model(combined)
                test_loss += custom_driving_loss(pred, actions_t).item()
                test_pred.append(pred.cpu().numpy())
                test_true.append(actions_np)

        avg_test_loss = test_loss / len(test_loader)
        test_m = compute_offline_metrics(np.concatenate(test_pred), np.concatenate(test_true))

        print("\n=== OFFLINE TEST RESULTS ===")
        print(f"Test Loss    : {avg_test_loss:.4f}")
        print(f"Steering MSE : {test_m['steering_mse']:.4f}")
        print(f"Accel MSE    : {test_m['accel_mse']:.4f}")
        print(f"Direction Acc: {test_m['direction_acc']:.3f}")
    else:
        print("[WARNING] No test data found. Skipping offline evaluation.")

    # ---- Online simulation test ----
    print("\n=> Running Online Evaluation (Simulation)...")

    # Always need live DPT for online inference
    if depth_estimator is None:
        depth_estimator = DepthEstimationModel(finetuned_path=dpt_path)

    config = {
        "use_render": True,
        "image_observation": True,
        "sensors": {"rgb": (RGBCamera, 200, 200)},
        "vehicle_config": {"image_source": "rgb"},
        "show_interface": False,
        "image_on_cuda": False,
    }
    env = MetaDriveEnv(config)
    success_flags, route_completions = [], []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done, step_count, anlik_fps = False, 0, 0.0
        last_time = time.time()

        while not done:
            step_count += 1
            rgb_img        = env.engine.get_sensor("rgb").perceive(env.agent)
            combined_tensor = extract_features_frozen(rgb_img[np.newaxis], depth_estimator, device)

            with torch.no_grad():
                pred_action = policy_model(combined_tensor).cpu().numpy()[0]

            if step_count % 2 == 0:
                depth_uint8 = (combined_tensor[0, 0].cpu().numpy() * 255).astype(np.uint8)
                lane_uint8  = (combined_tensor[0, 1].cpu().numpy() * 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
                lane_color  = cv2.cvtColor(lane_uint8, cv2.COLOR_GRAY2BGR)
                vis_size    = (400, 400)
                dashboard   = np.hstack((
                    cv2.resize(depth_color, vis_size, interpolation=cv2.INTER_LINEAR),
                    cv2.resize(lane_color,  vis_size, interpolation=cv2.INTER_NEAREST),
                ))
                cv2.imshow("Agent Perception: Depth (Left) | Lane Mask (Right)", dashboard)
                cv2.waitKey(1)

            for _ in range(3):
                obs, reward, terminated, truncated, info = env.step(pred_action)
                done = terminated or truncated
                if done: break

            cur_time  = time.time()
            elapsed   = cur_time - last_time
            last_time = cur_time
            if elapsed > 0: anlik_fps = 0.9 * anlik_fps + 0.1 * (1.0 / elapsed)
            print(f"EP: {ep+1} | Steer: {pred_action[0]:.2f} | Throttle {pred_action[1]:.2f} | FPS: {anlik_fps:.1f}", end="\r")

        success_flags.append(bool(info.get("arrive_dest", False)))
        route_completions.append(info.get("route_completion", 0.0))
        print(f"\nEpisode {ep+1} done. Success: {success_flags[-1]}")

    print(f"\n=== ONLINE SUMMARY ===\nSuccess: {np.mean(success_flags)*100:.1f}%  Route: {np.mean(route_completions)*100:.1f}%")
    env.close()
    cv2.destroyAllWindows()


# ==========================================
# 7. MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       type=str, required=True, choices=["train", "test", "all"])
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--episodes",   type=int,   default=1)
    parser.add_argument("--data_dir",   type=str,   default="dataset")
    parser.add_argument("--dpt_path",   type=str,   default="models/dpt_finetuned.pth")
    parser.add_argument("--model_path", type=str,   default="models/policy_model.pth")
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=None,
        help=(
            "Path to precomputed DPT predictions (output of train_dpt.py --mode precompute). "
            "When set, policy training/testing skips live DPT inference entirely. "
            "Example: data/processed/dpt_pred"
        ),
    )
    args = parser.parse_args()

    if args.mode in ["train", "all"]:
        train_policy(
            args.epochs, 32, args.model_path,
            args.dpt_path, args.data_dir, args.lr,
            pred_dir=args.pred_dir,
        )
    if args.mode in ["test", "all"]:
        test_policy(
            args.model_path, args.dpt_path, args.data_dir,
            args.episodes, pred_dir=args.pred_dir,
        )