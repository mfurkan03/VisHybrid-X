"""
train_dpt.py – fine-tune DepthAnythingV2 on MetaDrive data, then cache predictions.

Usage
-----
# Fine-tune
python src/train_dpt.py --mode train --epochs 5 \
    --data_dir dataset --model_path models/dpt_finetuned.pth

# Cache predictions for policy training
python src/train_dpt.py --mode precompute \
    --model_path models/dpt_finetuned.pth \
    --data_dir dataset --out_dir data/processed/dpt_pred
"""

import argparse
import glob
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models import DepthEstimationModel


# ============================================================
# 1. DATASET
# ============================================================
class MetaDriveDepthDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train",
                 subset_fraction: float = 1.0, augment: bool = False):
        self.augment  = augment
        split_dir     = os.path.join(data_dir, split)
        files         = sorted(glob.glob(os.path.join(split_dir, "*.npz")))

        if split == "train" and subset_fraction < 1.0:
            random.seed(42)
            num_files = max(1, int(len(files) * subset_fraction))
            files     = random.sample(files, num_files)
            print(f"[INFO] Training on {subset_fraction*100:.1f}% of data: {num_files} files.")

        self.rgb_frames: list = []
        self.gt_depths:  list = []

        for f in files:
            try:
                data = np.load(f, allow_pickle=True)
            except Exception:
                continue
            rgb_keys      = [k for k in data.files if k.endswith("_rgb")]
            depth_keys    = [k for k in data.files if k.endswith("_depth")]
            if not rgb_keys or not depth_keys:
                continue
            self.rgb_frames.extend(data[rgb_keys[0]])
            
            depth_data = data[depth_keys[0]]
            if depth_data.ndim == 3: # (N, H, W)
                depth_data = np.expand_dims(depth_data, axis=1)
            elif depth_data.ndim == 4 and depth_data.shape[-1] == 1: # (N, H, W, 1)
                depth_data = np.transpose(depth_data, (0, 3, 1, 2))
            
            self.gt_depths.extend(depth_data)

        print(f"[INFO] DepthDataset ({split}): {len(self.gt_depths)} samples from {split_dir}.")

    def __len__(self):
        return len(self.gt_depths)

    def __getitem__(self, idx):
        depth = np.array(self.gt_depths[idx], dtype=np.float32).copy()
        depth = np.expand_dims(depth, axis=0)

        if self.augment:
            rgb, depth = _augment(rgb, depth)
        return rgb, depth


def _augment(rgb: np.ndarray, depth: np.ndarray):
    # Horizontal flip
    if random.random() > 0.5:
        rgb   = cv2.flip(rgb, 1)
        depth = np.expand_dims(cv2.flip(depth[0], 1), axis=0)

    # Colour jitter
    if random.random() > 0.2:
        hsv            = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0]   = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
        hsv[:, :, 1]   = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
        hsv[:, :, 2]   = np.clip(hsv[:, :, 2] * random.uniform(0.7, 1.3), 0, 255)
        rgb             = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Gaussian blur
    if random.random() > 0.7:
        ksize = random.choice([3, 5])
        rgb   = cv2.GaussianBlur(rgb, (ksize, ksize), 0)

    # Additive noise
    if random.random() > 0.7:
        noise = np.random.normal(0, random.uniform(5, 15), rgb.shape).astype(np.float32)
        rgb   = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Cutout / random erasing
    if random.random() > 0.8:
        h, w     = rgb.shape[:2]
        y, x     = random.randint(0, h // 2), random.randint(0, w // 2)
        cut_h    = random.randint(10, h // 4)
        cut_w    = random.randint(10, w // 4)
        rgb[y:y + cut_h, x:x + cut_w, :] = 0

    return rgb, depth


# ============================================================
# 2. LOSS
# ============================================================
def scale_shift_invariant_loss(prediction, target, mask=None):
    """
    Scale and Shift Invariant Loss (SSIL).
    Aligns prediction to target via least-squares scale/shift before computing L1.
    """

    if mask is None:
        mask = target > 0

    prediction = prediction[mask]
    target     = target[mask]

    if prediction.numel() < 10:
        return torch.tensor(0.0, device=prediction.device, requires_grad=True)

    target_mean = target.mean()
    pred_mean   = prediction.mean()
    target_var  = target     - target_mean
    pred_var    = prediction - pred_mean

    scale           = (target_var * pred_var).sum() / (pred_var.pow(2).sum() + 1e-6)
    shift           = target_mean - scale * pred_mean
    aligned_pred    = scale * prediction + shift

    return torch.nn.functional.l1_loss(aligned_pred, target)


# ============================================================
# 3. TRAINING
# ============================================================
def train_dpt(
    epochs:         int   = 20,
    batch_size:     int   = 64,
    model_path:     str   = "models/dpt_finetuned.pth",
    data_dir:       str   = "dataset",
    lr:             float = 1e-5,
    train_subset:   float = 1.0,
    patience:       int   = 5,
):
    print("--- Phase 1: Training Depth Model (DPT) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    depth_estimator = DepthEstimationModel(finetuned_path=None, trainable=True)

    train_ds = MetaDriveDepthDataset(data_dir, split="train", subset_fraction=train_subset, augment=True)
    val_ds   = MetaDriveDepthDataset(data_dir, split="val",   subset_fraction=1.0,          augment=False)

    def collate_fn(batch):
        rgbs, depths = zip(*batch)
        return np.stack(rgbs), torch.tensor(np.stack(depths), dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Fixed validation batch for per-epoch visualisation
    vis_dir = "epoch_visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    fixed_vis_rgbs, fixed_vis_depths = next(iter(val_loader))
    fixed_vis_rgbs   = fixed_vis_rgbs[:4]
    fixed_vis_depths = fixed_vis_depths[:4]
    print(f"[INFO] Visualizations will be saved to ./{vis_dir}/")

    optimizer = optim.AdamW(depth_estimator.model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    scaler    = torch.amp.GradScaler("cuda")

    min_val_loss         = math.inf
    patience_counter     = 0
    last_best_epoch_path = None

    for epoch in range(epochs):
        depth_estimator.set_train_mode()
        train_loss = 0.0

        for rgb_np, gt_depth in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            gt_depth = gt_depth.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                pred_depth = depth_estimator.predict_batch_with_grad(rgb_np)
                loss       = scale_shift_invariant_loss(pred_depth, gt_depth)
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
                gt_depth  = gt_depth.to(device)
                val_loss += scale_shift_invariant_loss(
                    depth_estimator.predict_batch_with_grad(rgb_np), gt_depth
                ).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train: {avg_train:.4f} | "
              f"Val: {avg_val:.4f} | LR: {scheduler.get_last_lr()[0]:.6e}")
        scheduler.step()

        if avg_val < min_val_loss:
            min_val_loss     = avg_val
            patience_counter = 0
            base_dir = os.path.dirname(model_path) or "."
            name, ext = os.path.splitext(os.path.basename(model_path))
            epoch_path = os.path.join(base_dir, f"{name}_ep{epoch+1:02d}{ext}")

            torch.save(depth_estimator.model.state_dict(), epoch_path)
            torch.save(depth_estimator.model.state_dict(), model_path)
            print(f"*** Best DPT checkpoint: {epoch_path}  (Val: {min_val_loss:.4f}) ***")

            if last_best_epoch_path and os.path.exists(last_best_epoch_path):
                os.remove(last_best_epoch_path)
                print(f"    -> Removed older checkpoint: {last_best_epoch_path}")
            last_best_epoch_path = epoch_path
        else:
            patience_counter += 1
            print(f"--- Early Stopping Counter: {patience_counter}/{patience} ---")
            if patience_counter >= patience:
                print(f"[INFO] Early stopping triggered after {patience} epochs without improvement.")
                break

        # Save visualisation grid
        with torch.no_grad():
            pred_vis = depth_estimator.predict_batch_with_grad(fixed_vis_rgbs)
        _save_vis_grid(fixed_vis_rgbs, fixed_vis_depths, pred_vis,
                       os.path.join(vis_dir, f"epoch_{epoch+1:03d}.jpg"))


def _save_vis_grid(fixed_vis_rgbs, fixed_vis_depths, pred_vis, out_file):
    image_size = 196
    vis_rows = []
    for i in range(len(fixed_vis_rgbs)):
        rgb_img = fixed_vis_rgbs[i].copy()
        if rgb_img.max() <= 1.0:
            rgb_img = (rgb_img * 255).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(cv2.resize(rgb_img, (image_size, image_size)), cv2.COLOR_RGB2BGR)

        gt_d = fixed_vis_depths[i, 0].cpu().numpy()
        gt_d = cv2.resize(gt_d, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        gt_d = (gt_d - gt_d.min()) / (gt_d.max() - gt_d.min() + 1e-6)
        gt_d_color = cv2.applyColorMap((gt_d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

        pred_d = pred_vis[i, 0].cpu().numpy()
        pred_d = cv2.resize(pred_d, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        pred_d = (pred_d - pred_d.min()) / (pred_d.max() - pred_d.min() + 1e-6)
        pred_d = 1.0 - pred_d
        pred_d_color = cv2.applyColorMap((pred_d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

        vis_rows.append(np.hstack((rgb_bgr, gt_d_color, pred_d_color)))

    cv2.imwrite(out_file, np.vstack(vis_rows))


# ============================================================
# 4. PRECOMPUTE DPT PREDICTIONS
# ============================================================
def _process_episode(ep_path: str, out_path: str, dpt_path: str, batch_size: int) -> str:
    """
    Worker function: loads its own model instance and processes a single episode file.
    Returns a status string for logging.
    """
    from models import DepthEstimationModel  # local import so each worker loads fresh

    ep_name = os.path.basename(ep_path)

    try:
        data = np.load(ep_path, allow_pickle=True)
    except Exception as e:
        return f"  [ERROR] {ep_path}: {e}"

    rgb_keys = [k for k in data.files if k.endswith("_rgb")]
    if not rgb_keys:
        return f"  [SKIP] No RGB key in {ep_name}."

    depth_estimator = DepthEstimationModel(finetuned_path=dpt_path, trainable=False)
    depth_estimator.set_eval_mode()

    rgb_frames  = data[rgb_keys[0]]
    actions     = data.get("action")  # may be None
    N           = len(rgb_frames)
    if N == 0:
        return f"  [SKIP] Empty RGB array in {ep_name}."
        
    orig_H, orig_W = rgb_frames.shape[1:3]
    depth_preds = np.empty((N, 1, orig_H, orig_W), dtype=np.float32)

    for start in range(0, N, batch_size):
        end   = min(start + batch_size, N)
        batch = rgb_frames[start:end]
        with torch.no_grad():
            pred_batch = depth_estimator.predict_batch_with_grad(batch)

        for i in range(len(batch)):
            pred_d = pred_batch[i, 0].cpu().numpy()
            pred_d = (pred_d - pred_d.min()) / (pred_d.max() - pred_d.min() + 1e-6)
            pred_d = 1.0 - pred_d
            depth_preds[start + i, 0] = pred_d

    save_dict = {"depth_pred": depth_preds, "rgb": rgb_frames}
    if "ego_state" in data.files:
        save_dict["ego_state"] = data["ego_state"]
    if "ego_state_full" in data.files:
        save_dict["ego_state_full"] = data["ego_state_full"]
    if actions is not None:
        save_dict["action"] = actions

    np.savez_compressed(out_path, **save_dict)
    return f"  Saved {ep_name}  [{N} frames]  ->  {out_path}"


def precompute_dpt_predictions(
    dpt_path:    str,
    data_dir:    str   = "dataset",
    out_dir:     str   = "data/processed/dpt_pred",
    batch_size:  int   = 32,
    splits:      tuple = ("train", "val"),
    num_workers: int   = 4,
):
    print("--- Precomputing DPT Predictions ---")

    for split in splits:
        split_in_dir  = os.path.join(data_dir, split)
        split_out_dir = os.path.join(out_dir, split)
        os.makedirs(split_out_dir, exist_ok=True)

        episode_files = sorted(glob.glob(os.path.join(split_in_dir, "*.npz")))
        if not episode_files:
            print(f"[WARNING] No .npz files in {split_in_dir}. Skipping '{split}'.")
            continue
        print(f"\n[{split.upper()}] {len(episode_files)} file(s) -> {split_out_dir}")

        # Build work list, skipping already-processed episodes
        work = []
        for ep_path in episode_files:
            ep_name  = os.path.basename(ep_path)
            out_path = os.path.join(split_out_dir, ep_name)
            if os.path.exists(out_path):
                print(f"  [SKIP] {ep_name} already exists.")
            else:
                work.append((ep_path, out_path))

        if not work:
            continue

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_episode, ep_path, out_path, dpt_path, batch_size): ep_path
                for ep_path, out_path in work
            }
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"  [{split}] episodes"):
                print(future.result())

    print(f"\n[INFO] Precomputation complete. Cached predictions are in: {out_dir}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",         type=str, default="train",
                        choices=["train", "precompute"])
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--data_dir",     type=str,   default="dataset")
    parser.add_argument("--model_path",   type=str,   default="models/dpt_finetuned.pth")
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--train_subset", type=float, default=1.0,
                        help="Fraction of training data to use (e.g. 0.2 for 20%%)")
    parser.add_argument("--patience",     type=int,   default=5,
                        help="Early stopping patience (epochs without val improvement)")
    parser.add_argument("--out_dir",      type=str,   default="data/processed/dpt_pred",
                        help="Destination for cached DPT predictions (--mode precompute)")
    parser.add_argument("--splits",       type=str,   nargs="+", default=["train", "val", "test"])
    parser.add_argument("--num_workers",  type=int,   default=4,
                        help="Number of parallel worker processes for precomputation")
    args = parser.parse_args()

    if args.mode == "train":
        train_dpt(args.epochs, args.batch_size, args.model_path,
                  args.data_dir, args.lr, args.train_subset, args.patience)
    elif args.mode == "precompute":
        precompute_dpt_predictions(
            dpt_path    = args.model_path,
            data_dir    = args.data_dir,
            out_dir     = args.out_dir,
            batch_size  = args.batch_size,
            splits      = tuple(args.splits),
            num_workers = args.num_workers,
        )