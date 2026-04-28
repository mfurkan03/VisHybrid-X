"""
policy/trainer.py – shared epoch loop, data loader builder, and feature extractor.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import EGO_DIM
from policy.datasets import MetaDriveRGBDataset, PrecomputedDepthDataset
from policy.losses import custom_driving_loss, compute_offline_metrics
from utils.checkpoints import save_checkpoint


# ============================================================
# FEATURE EXTRACTION  (live DPT inference path)
# ============================================================
def get_lane_mask_visual(rgb_image: np.ndarray, threshold_value: int = 180) -> np.ndarray:
    img_uint8 = (rgb_image * 255.0).astype(np.uint8) if rgb_image.max() <= 1.0 else rgb_image.astype(np.uint8)
    h, w      = img_uint8.shape[:2]
    roi       = img_uint8.copy()
    roi[0:int(h * 0.55), :] = 0
    gray      = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    _, mask   = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return mask


def extract_features_frozen(rgb_batch, depth_estimator, device):
    rescaled = cv2.resize(rgb_batch[0], (196, 196), interpolation=cv2.INTER_LINEAR)
    with torch.no_grad():
        depth_tensors = depth_estimator.predict_batch_with_grad(np.expand_dims(rescaled, 0))

    for i in range(depth_tensors.shape[0]):
        d = depth_tensors[i, 0]
        depth_tensors[i, 0] = 1.0 - (d - d.min()) / (d.max() - d.min() + 1e-6)

    lane_list = []
    for rgb in rgb_batch:
        mask = get_lane_mask_visual(rgb)
        lane_list.append((cv2.resize(mask, (84, 84)) / 255.0).astype(np.float32))
    lane_tensor = torch.tensor(np.stack(lane_list), device=device).unsqueeze(1)

    combined    = torch.cat([depth_tensors, lane_tensor], dim=1)
    ego_zeros   = torch.zeros(combined.shape[0], EGO_DIM, device=device)
    return combined, ego_zeros


# ============================================================
# DATA LOADER BUILDER
# ============================================================
def build_loaders(use_precomputed: bool, pred_dir, data_dir, batch_size, depth_estimator):
    if use_precomputed:
        train_ds = PrecomputedDepthDataset(pred_dir=pred_dir, split="train")
        val_ds   = PrecomputedDepthDataset(pred_dir=pred_dir, split="val")

        def collate_fn(batch):
            combined, actions, egos = zip(*batch)
            return (
                torch.tensor(np.stack(combined), dtype=torch.float32),
                np.stack(actions),
                np.stack(egos),
            )
    else:
        train_ds = MetaDriveRGBDataset(data_dir=data_dir, split="train")
        val_ds   = MetaDriveRGBDataset(data_dir=data_dir, split="val")

        def collate_fn(batch):
            rgbs, actions, egos = zip(*batch)
            return np.stack(rgbs), np.stack(actions), np.stack(egos)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader


# ============================================================
# SINGLE EPOCH
# ============================================================
def run_epoch(policy_model, loader, optimizer, device,
              use_precomputed, depth_estimator, is_train, desc):
    """Run one training or validation epoch. Returns (avg_loss, preds, trues)."""
    policy_model.train() if is_train else policy_model.eval()
    total_loss          = 0.0
    all_pred, all_true  = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=desc, leave=False):
            if use_precomputed:
                combined_t, actions_np, ego_np = batch
                combined  = combined_t.to(device)
                actions_t = torch.tensor(actions_np, dtype=torch.float32, device=device)
                ego_t     = torch.tensor(ego_np,     dtype=torch.float32, device=device)
            else:
                rgb_np, actions_np, ego_np = batch
                actions_t   = torch.tensor(actions_np, dtype=torch.float32, device=device)
                combined, _ = extract_features_frozen(rgb_np, depth_estimator, device)
                ego_t       = torch.tensor(ego_np, dtype=torch.float32, device=device)

            if is_train:
                optimizer.zero_grad()

            pred = policy_model(combined, ego_t)
            loss = custom_driving_loss(pred, actions_t)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_pred.append(pred.detach().cpu().numpy())
            all_true.append(actions_np)

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, np.concatenate(all_pred), np.concatenate(all_true)


# ============================================================
# FULL TRAINING LOOP
# ============================================================
def train_loop(
    policy_model,
    device,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    epochs:          int,
    start_epoch:     int,
    best_val_loss:   float,
    model_path:      str,
    use_precomputed: bool,
    depth_estimator,
    tag:             str = "Train",
) -> float:
    """Shared epoch loop used by train_policy and finetune_policy."""
    file_root, file_ext = os.path.splitext(model_path)
    best_path           = f"{file_root}_best{file_ext}"

    for epoch in range(start_epoch, start_epoch + epochs):
        avg_train, tr_pred, tr_true = run_epoch(
            policy_model, train_loader, optimizer, device,
            use_precomputed, depth_estimator, is_train=True,
            desc=f"[{tag}] Epoch {epoch+1}/{start_epoch+epochs} [Train]",
        )
        avg_val, val_pred, val_true = run_epoch(
            policy_model, val_loader, optimizer, device,
            use_precomputed, depth_estimator, is_train=False,
            desc=f"[{tag}] Epoch {epoch+1}/{start_epoch+epochs} [Val]",
        )

        tr_m  = compute_offline_metrics(tr_pred,  tr_true)
        val_m = compute_offline_metrics(val_pred, val_true)
        print(
            f"[{tag}] Epoch [{epoch+1:02d}] "
            f"Loss Tr/Val: {avg_train:.4f}/{avg_val:.4f} | "
            f"Steer MSE Tr/Val: {tr_m['steering_mse']:.4f}/{val_m['steering_mse']:.4f} | "
            f"Dir Acc: {val_m['direction_acc']:.3f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        scheduler.step()

        save_checkpoint(policy_model, optimizer, scheduler, epoch, avg_val, model_path)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_checkpoint(policy_model, optimizer, scheduler, epoch, avg_val, best_path)
            print(f"*** Best model saved → {best_path}  (Val Loss: {best_val_loss:.4f}) ***")

    return best_val_loss


# late import to avoid circular dep
import os  # noqa: E402