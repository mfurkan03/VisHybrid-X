import os
import glob
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import sys
from pathlib import Path
from tqdm import tqdm

# Import the centralized models
from models import DepthEstimationModel



# ==========================================
# 2. DATASET & LOSS
# ==========================================
class MetaDriveDepthDataset(Dataset):
    def __init__(self, data_dir, split="train", subset_fraction=1.0, augment=False):
        self.augment = augment
        split_dir = os.path.join(data_dir, split)
        self.files      = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
        
        # --- SUBSET LOGIC ---
        if split == "train" and subset_fraction < 1.0:
            import random
            random.seed(42)  # Fixed seed so we grab the same subset across runs
            num_files = max(1, int(len(self.files) * subset_fraction))
            self.files = random.sample(self.files, num_files)
            print(f"[INFO] Training on {subset_fraction*100:.1f}% of data: {num_files} files.")

        self.rgb_frames = []
        self.gt_depths  = []

        for f in self.files:
            try: data = np.load(f, allow_pickle=True)
            except Exception: continue

            rgb_keys      = [k for k in data.files if k.endswith('_rgb')]
            combined_keys = [k for k in data.files if k.endswith('_combined')]
            if not rgb_keys or not combined_keys: continue

            self.rgb_frames.extend(data[rgb_keys[0]])
            self.gt_depths.extend(data[combined_keys[0]][:, 0:1, :, :])

        print(f"[INFO] DepthDataset ({split}): {len(self.gt_depths)} samples loaded from {split_dir}.")

    def __len__(self): return len(self.gt_depths)
    
    def __getitem__(self, idx):
        rgb = self.rgb_frames[idx].copy()
        depth = np.array(self.gt_depths[idx], dtype=np.float32).copy()

        # --- FIX: Resize to a multiple of 14 (196x196 is closest to 200x200) ---
        target_size = (196, 196) # (Width, Height)
        
        # Resize RGB (Interpolation: Linear is good for RGB)
        rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Resize Depth (Interpolation: Nearest avoids averaging artifacts on depth boundaries)
        # Depth is currently (1, H, W), so we extract the 2D plane, resize, and re-expand
        depth_sq = depth[0]
        depth_sq = cv2.resize(depth_sq, target_size, interpolation=cv2.INTER_NEAREST)
        depth = np.expand_dims(depth_sq, axis=0)
        # -----------------------------------------------------------------------

        if self.augment:
            rgb, depth = self._apply_augmentations(rgb, depth)

        return rgb, depth

    def _apply_augmentations(self, rgb, depth):
        import random
        
        # 1. Random Horizontal Flip (50% chance)
        if random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            # Depth is shape (1, H, W). cv2.flip expects (H, W) for 2D.
            depth_sq = cv2.flip(depth[0], 1)
            depth = np.expand_dims(depth_sq, axis=0)

        # 2. Random Color Jitter (80% chance)
        if random.random() > 0.2:
            # Convert to HSV for robust brightness/saturation/hue shifts
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Hue shift (-10 to +10)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
            # Saturation scale (0.7 to 1.3)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
            # Brightness/Value scale (0.7 to 1.3)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.7, 1.3), 0, 255)
            
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # 3. Random Gaussian Blur (30% chance)
        if random.random() > 0.7:
            ksize = random.choice([3, 5])
            rgb = cv2.GaussianBlur(rgb, (ksize, ksize), 0)

        # 4. Random Additive Gaussian Noise (30% chance)
        if random.random() > 0.7:
            noise = np.random.normal(0, random.uniform(5, 15), rgb.shape).astype(np.float32)
            rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 5. Random Cutout / Erasing (20% chance) - simulates occlusion
        if random.random() > 0.8:
            h, w = rgb.shape[:2]
            y = random.randint(0, h // 2)
            x = random.randint(0, w // 2)
            cut_h, cut_w = random.randint(10, h // 4), random.randint(10, w // 4)
            # Add black box to RGB (does NOT modify depth map, encouraging model to infer through occlusion)
            rgb[y:y+cut_h, x:x+cut_w, :] = 0 

        return rgb, depth

def scale_shift_invariant_loss(prediction, target, mask=None):
    """
    Scale and Shift Invariant Loss (SSIL).
    Aligns the prediction to the target dynamically solving for scale and shift (Prediction = s * GT + t)
    """
    if prediction.shape != target.shape:
        import torch.nn.functional as F
        target = F.interpolate(target, size=prediction.shape[-2:], mode='bilinear', align_corners=False)

    if mask is None:
        mask = target > 0 # Ignore zero-depth background pixels

    # Flatten tensors
    prediction = prediction[mask]
    target = target[mask]

    if prediction.numel() < 10:
        return torch.tensor(0.0, device=prediction.device, requires_grad=True)

    # Calculate optimal scale and shift using least squares
    # This prevents the model from being punished for being "inverted" or having a different base scale
    target_mean = target.mean()
    pred_mean = prediction.mean()

    target_var = target - target_mean
    pred_var = prediction - pred_mean

    scale = (target_var * pred_var).sum() / (pred_var.pow(2).sum() + 1e-6)
    shift = target_mean - scale * pred_mean

    # Align prediction
    aligned_prediction = scale * prediction + shift

    # Compute standard L1 or L2 loss on the aligned maps
    loss = torch.nn.functional.l1_loss(aligned_prediction, target)
    return loss

# --- ADDED: patience parameter ---
# --- ADDED: patience parameter ---
def train_dpt(epochs=20, batch_size=64, model_path="dpt_finetuned.pth", data_dir="dataset", lr=1e-5, train_subset=1.0, patience=5):
    print("--- Phase 1: Training Depth Model (DPT) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the centralized model, explicitly passing trainable=True
    depth_estimator = DepthEstimationModel(finetuned_path=None, trainable=True)
    
    train_ds = MetaDriveDepthDataset(data_dir=data_dir, split="train", subset_fraction=train_subset, augment=True)
    val_ds   = MetaDriveDepthDataset(data_dir=data_dir, split="val", subset_fraction=1.0, augment=False)

    def collate_fn(batch):
        rgbs, depths = zip(*batch)
        return np.stack(rgbs), torch.tensor(np.stack(depths), dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # --- Setup Visualization Batch ---
    vis_dir = "epoch_visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    print(f"[INFO] Visualizations will be saved to ./{vis_dir}/")
    
    fixed_vis_rgbs, fixed_vis_depths = next(iter(val_loader))
    fixed_vis_rgbs = fixed_vis_rgbs[:4]
    fixed_vis_depths = fixed_vis_depths[:4]
    # --------------------------------------

    optimizer = optim.AdamW(depth_estimator.model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda')
    import math
    min_val_loss = math.inf
    patience_counter = 0
    
    # --- NEW: Track the specific epoch file to delete later ---
    last_best_epoch_path = None

    for epoch in range(epochs):
        depth_estimator.set_train_mode()
        train_loss = 0.0

        for rgb_np, gt_depth in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            gt_depth = gt_depth.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred_depth = depth_estimator.predict_batch_with_grad(rgb_np)
                loss = scale_shift_invariant_loss(pred_depth, gt_depth)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(depth_estimator.model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        depth_estimator.set_eval_mode()
        val_loss = 0.0
        with torch.no_grad():
            for rgb_np, gt_depth in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                gt_depth = gt_depth.to(device)
                pred_depth = depth_estimator.predict_batch_with_grad(rgb_np) 
                val_loss += scale_shift_invariant_loss(pred_depth, gt_depth).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)

        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6e}")
        scheduler.step()

        # --- UPDATED: Early stopping & Strict Best-Model Keeping ---
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            patience_counter = 0  # Reset counter
            
            # 1. Create a dynamic filename tracking the epoch
            base_dir = os.path.dirname(model_path) or "."
            name, ext = os.path.splitext(os.path.basename(model_path))
            epoch_specific_path = os.path.join(base_dir, f"{name}_ep{epoch+1:02d}{ext}")
            
            # 2. Save the new best model with the epoch name
            torch.save(depth_estimator.model.state_dict(), epoch_specific_path)
            
            # 3. Also overwrite the static model_path so downstream scripts don't break
            torch.save(depth_estimator.model.state_dict(), model_path)
            
            print(f"*** Saved new best DPT checkpoint: {epoch_specific_path} (Val Loss: {min_val_loss:.4f}) ***")
            
            # 4. Delete the previous best epoch file to save storage!
            if last_best_epoch_path and os.path.exists(last_best_epoch_path):
                os.remove(last_best_epoch_path)
                print(f"    -> Removed older checkpoint: {last_best_epoch_path}")
            
            # 5. Update the tracker
            last_best_epoch_path = epoch_specific_path
            
        else:
            patience_counter += 1
            print(f"--- Early Stopping Counter: {patience_counter}/{patience} ---")
            if patience_counter >= patience:
                print(f"[INFO] Early stopping triggered! Validation loss hasn't improved for {patience} epochs.")
                break

        with torch.no_grad():
            pred_vis_depths = depth_estimator.predict_batch_with_grad(fixed_vis_rgbs)
        
        vis_rows = []
        for i in range(len(fixed_vis_rgbs)):
            rgb_img = fixed_vis_rgbs[i].copy()
            if rgb_img.max() <= 1.0:
                rgb_img = (rgb_img * 255).astype(np.uint8)
            
            rgb_img = cv2.resize(rgb_img, (196, 196)) 
            rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

            gt_d = fixed_vis_depths[i, 0].cpu().numpy()
            gt_d = cv2.resize(gt_d, (196, 196), interpolation=cv2.INTER_NEAREST)
            gt_d = (gt_d - gt_d.min()) / (gt_d.max() - gt_d.min() + 1e-6)
            gt_d_color = cv2.applyColorMap((gt_d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

            pred_d = pred_vis_depths[i, 0].cpu().numpy()
            pred_d = cv2.resize(pred_d, (196, 196), interpolation=cv2.INTER_NEAREST)
            pred_d = (pred_d - pred_d.min()) / (pred_d.max() - pred_d.min() + 1e-6)
            pred_d = 1.0 - pred_d 
            pred_d_color = cv2.applyColorMap((pred_d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

            row_img = np.hstack((rgb_bgr, gt_d_color, pred_d_color))
            vis_rows.append(row_img)
            
        grid_img = np.vstack(vis_rows)
        out_file = os.path.join(vis_dir, f"epoch_{epoch+1:03d}.jpg")
        cv2.imwrite(out_file, grid_img)


# ==========================================
# 4. PRECOMPUTE DPT PREDICTIONS
# ==========================================
def precompute_dpt_predictions(
    dpt_path:    str,
    data_dir:    str  = "dataset",
    out_dir:     str  = "data/processed/dpt_pred",
    batch_size:  int  = 32,
    splits:      tuple = ("train", "val"),
):
    print("--- Precomputing DPT Predictions ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the centralized model wrapper just like in training
    depth_estimator = DepthEstimationModel(finetuned_path=dpt_path, trainable=False)
    depth_estimator.set_eval_mode()

    # ---- Helper: compute lane mask from RGB ----
    def _lane_mask(rgb_np: np.ndarray, threshold: int = 180) -> np.ndarray:
        """Returns a float32 array of shape (1, 84, 84), values in {0, 1}."""
        if rgb_np.max() <= 1.0:
            img_u8 = (rgb_np * 255).astype(np.uint8)
        else:
            img_u8 = rgb_np.astype(np.uint8)
        h, w = img_u8.shape[:2]
        roi = img_u8.copy()
        roi[: int(h * 0.55), :] = 0
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        resized = cv2.resize(mask, (84, 84), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
        return resized[np.newaxis]  # (1, 84, 84)

    # ---- Process each split ----
    for split in splits:
        split_in_dir  = os.path.join(data_dir, split)
        split_out_dir = os.path.join(out_dir, split)
        os.makedirs(split_out_dir, exist_ok=True)

        episode_files = sorted(glob.glob(os.path.join(split_in_dir, "*.npz")))
        if not episode_files:
            print(f"[WARNING] No .npz files found in {split_in_dir}. Skipping split '{split}'.")
            continue

        print(f"\n[{split.upper()}] Processing {len(episode_files)} episode file(s) -> {split_out_dir}")

        for ep_path in episode_files:
            ep_name = os.path.basename(ep_path)
            out_path = os.path.join(split_out_dir, ep_name)

            if os.path.exists(out_path):
                print(f"  [SKIP] {ep_name} already exists.")
                continue

            try:
                data = np.load(ep_path, allow_pickle=True)
            except Exception as e:
                print(f"  [ERROR] Could not load {ep_path}: {e}")
                continue

            rgb_keys = [k for k in data.files if k.endswith('_rgb')]
            if not rgb_keys:
                print(f"  [SKIP] No RGB key found in {ep_name}.")
                continue

            rgb_frames = data[rgb_keys[0]]   # (N, H, W, 3) uint8
            actions    = data['action'] if 'action' in data.files else None
            N          = len(rgb_frames)

            depth_preds = np.empty((N, 1, 84, 84), dtype=np.float32)
            lane_masks  = np.empty((N, 1, 84, 84), dtype=np.float32)

            # Run inference in mini-batches
            for start in tqdm(range(0, N, batch_size), desc=f"  {ep_name}", leave=False):
                end   = min(start + batch_size, N)
                batch_rgb = rgb_frames[start:end]
                batch_rescaled = np.array([
                    cv2.resize(frame, (196, 196), interpolation=cv2.INTER_LINEAR) 
                    for frame in batch_rgb
                ])
                # 1. Batched inference matching the training loop
                with torch.no_grad():
                    pred_batch_tensor = depth_estimator.predict_batch_with_grad(batch_rescaled)
                
                # 2. Process each frame identically to the epoch visualization logic
                for i in range(len(batch_rgb)):
                    pred_d = pred_batch_tensor[i, 0].cpu().numpy()
                    
                    # Resize to the required 84x84 policy shape using NEAREST
                    pred_d = cv2.resize(pred_d, (84, 84), interpolation=cv2.INTER_NEAREST)
                    
                    # Normalize to 0-1
                    pred_d = (pred_d - pred_d.min()) / (pred_d.max() - pred_d.min() + 1e-6)
                    
                    # INVERT IT SO IT LOOKS LIKE METADRIVE DEPTH
                    pred_d = 1.0 - pred_d 
                    
                    depth_preds[start + i, 0] = pred_d
                    lane_masks[start + i]  = _lane_mask(batch_rgb[i])

            save_dict = {
                "depth_pred": depth_preds,   # (N, 1, 84, 84) float32
                "lane_mask":  lane_masks,    # (N, 1, 84, 84) float32
            }
            if 'ego_state' in data.files:
               save_dict["ego_state"] = data['ego_state']
            if actions is not None:
                save_dict["action"] = actions

            np.savez_compressed(out_path, **save_dict)
            print(f"  Saved {ep_name}  [{N} frames]  ->  {out_path}")

    print(f"\n[INFO] Precomputation complete. Cached predictions are in: {out_dir}")

# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       type=str, default="train",
                        choices=["train", "precompute"],
                        help="'train' = fine-tune DPT; 'precompute' = cache DPT predictions for policy training")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--data_dir",   type=str,   default="dataset")
    parser.add_argument("--model_path", type=str,   default="models/dpt_finetuned.pth")
    parser.add_argument("--lr",         type=float, default=1e-5)
    parser.add_argument("--train_subset", type=float, default=1.0,
                        help="Fraction of training data to use (e.g. 0.2 for 20%)")
    
    # --- ADDED: Patience argument ---
    parser.add_argument("--patience",   type=int,   default=5,
                        help="Number of epochs to wait for val loss improvement before stopping")
                        
    # Precompute-only args
    parser.add_argument("--out_dir",    type=str,   default="data/processed/dpt_pred",
                        help="Where to write cached DPT predictions (used with --mode precompute)")
    parser.add_argument("--splits",     type=str,   nargs="+", default=["train", "val","test"],
                        help="Which dataset splits to precompute (default: train val)")
    args = parser.parse_args()

    if args.mode == "train":
        # --- ADDED: Pass patience argument to train function ---
        train_dpt(args.epochs, args.batch_size, args.model_path, args.data_dir, args.lr, args.train_subset, args.patience)

    elif args.mode == "precompute":
        precompute_dpt_predictions(
            dpt_path   = args.model_path,
            data_dir   = args.data_dir,
            out_dir    = args.out_dir,
            batch_size = args.batch_size,
            splits     = tuple(args.splits),
        )