"""
policy/trainer.py – shared epoch loop, data loader builder, and feature extractor.
"""

import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import EGO_DIM
from policy.datasets import MetaDriveRGBDataset, PrecomputedDepthDataset
from policy.losses import (custom_driving_loss, compute_offline_metrics,
                           compute_predictive_metrics, compute_heading_metrics)
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


import torch
import torch.nn.functional as F
import numpy as np

def apply_lane_mask(
    depth_tensor: torch.Tensor,
    rgb_batch: np.ndarray,
    device: torch.device,
    current_epoch: int = 999,
    curriculum_epochs: int = 10,
    fully_masked_epochs: int = 3,
    image_size: int = None
) -> torch.Tensor:
    """
    Applies lane masking at original resolution and interpolates the final result.
    """
    
    # 1. Calculate Alpha for Curriculum Learning
    if current_epoch < fully_masked_epochs:
        alpha = 0.0
    elif current_epoch >= curriculum_epochs:
        alpha = 1.0
    else:
        alpha = (current_epoch - fully_masked_epochs) / max(1, curriculum_epochs - fully_masked_epochs)

    # 2. Pre-process RGB Batch (Maintain original resolution for now)
    rgb_tensor = torch.from_numpy(rgb_batch).float().to(device)
    
    if rgb_tensor.max() > 1.0:
        rgb_tensor /= 255.0
        
    # Convert (B, H, W, C) -> (B, C, H, W)
    rgb_tensor = rgb_tensor.permute(0, 3, 1, 2) 

    # 3. Lane Masking and Blending (at original resolution)
    blended_list = []
    
    for i in range(rgb_tensor.shape[0]):
        # Extract single image for mask generation
        img_torch = rgb_tensor[i]
        img_np = (img_torch.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # Generate mask (Expected output: H, W)
        mask = get_lane_mask_visual(img_np) 
        mask_tensor = torch.from_numpy(mask).float().to(device).unsqueeze(0) / 255.0
        
        # Apply blending logic
        # Final = (Original * Mask) + (Original * (1 - Mask) * Alpha)
        blended = img_torch * mask_tensor + img_torch * (1.0 - mask_tensor) * alpha
        blended_list.append(blended)

    blended_batch = torch.stack(blended_list)

    # 4. Concatenate Depth (C=1) and Blended RGB (C=3) -> (B, 4, H, W)
    combined_tensor = torch.cat([depth_tensor, blended_batch], dim=1)

    # 5. Final Interpolation
    # We resize the concatenated tensor once at the very end
    if image_size:
        combined_tensor = F.interpolate(
            combined_tensor, 
            size=(image_size, image_size), 
            mode='bilinear', 
            align_corners=False
        )

    return combined_tensor

def extract_features_frozen(
    rgb_batch,
    depth_estimator,
    device,
    current_epoch: int = 999,
    curriculum_epochs: int = 10,
    fully_masked_epochs: int = 3,
    image_size: int = None
):
    with torch.no_grad():
        depth_tensors = depth_estimator.predict_batch_with_grad(np.expand_dims(rgb_batch[0], 0))

    for i in range(depth_tensors.shape[0]):
        d = depth_tensors[i, 0]
        depth_tensors[i, 0] = 1.0 - (d - d.min()) / (d.max() - d.min() + 1e-6)

    combined  = apply_lane_mask(
        depth_tensors, rgb_batch, device,
        current_epoch=current_epoch,
        curriculum_epochs=curriculum_epochs,
        fully_masked_epochs=fully_masked_epochs,
        image_size = image_size
    )

    ego_zeros = torch.zeros(combined.shape[0], EGO_DIM, device=device)
    return combined, ego_zeros


# ============================================================
# DATA LOADER BUILDER
# ============================================================
def build_loaders(use_precomputed: bool, pred_dir, data_dir, batch_size, depth_estimator):
    if use_precomputed:
        train_ds = PrecomputedDepthDataset(pred_dir=pred_dir, split="train")
        val_ds   = PrecomputedDepthDataset(pred_dir=pred_dir, split="val")

        def collate_fn(batch):
            depths, rgbs, actions, egos, ego_fulls = zip(*batch)
            return (
                torch.tensor(np.stack(depths), dtype=torch.float32),
                np.stack(rgbs),
                np.stack(actions),
                np.stack(egos),
                np.stack(ego_fulls),
            )
    else:
        train_ds = MetaDriveRGBDataset(data_dir=data_dir, split="train")
        val_ds   = MetaDriveRGBDataset(data_dir=data_dir, split="val")

        def collate_fn(batch):
            rgbs, actions, egos = zip(*batch)
            n = len(actions)
            return (np.stack(rgbs), np.stack(actions), np.stack(egos),
                    np.zeros((n, 5), dtype=np.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader


# ============================================================
# SINGLE EPOCH
# ============================================================
def run_epoch(policy_model, loader, optimizer, device,
              use_precomputed, depth_estimator, is_train, desc,
              current_epoch: int = 999, curriculum_epochs: int = 10, fully_masked_epochs: int = 3,
              image_size: int = None):
    """Run one training or validation epoch. Returns (avg_loss, preds, trues, ego_states, ego_fulls)."""
    policy_model.train() if is_train else policy_model.eval()
    total_loss                           = 0.0
    all_pred, all_true, all_ego, all_ego_full = [], [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=desc, leave=False):
            if use_precomputed:
                depth_t, rgb_np, actions_np, ego_np, ego_full_np = batch
                depth_t   = depth_t.to(device)
                actions_t = torch.tensor(actions_np, dtype=torch.float32, device=device)
                ego_t     = torch.tensor(ego_np,     dtype=torch.float32, device=device)
                combined  = apply_lane_mask(
                    depth_t, rgb_np, device,
                    current_epoch=current_epoch,
                    curriculum_epochs=curriculum_epochs,
                    fully_masked_epochs=fully_masked_epochs,
                    image_size=image_size
                )
            else:
                rgb_np, actions_np, ego_np, ego_full_np = batch
                actions_t   = torch.tensor(actions_np, dtype=torch.float32, device=device)
                combined, _ = extract_features_frozen(
                    rgb_np, depth_estimator, device,
                    current_epoch=current_epoch,
                    curriculum_epochs=curriculum_epochs,
                    fully_masked_epochs=fully_masked_epochs,
                    image_size=image_size
                )
                ego_t = torch.tensor(ego_np, dtype=torch.float32, device=device)

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
            all_ego.append(ego_np)
            all_ego_full.append(ego_full_np)

    avg_loss = total_loss / max(len(loader), 1)
    return (avg_loss,
            np.concatenate(all_pred), np.concatenate(all_true),
            np.concatenate(all_ego),  np.concatenate(all_ego_full))


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
    curriculum_epochs: int = 10,
    fully_masked_epochs: int = 3,
    image_size: int = None,
) -> float:
    """Shared epoch loop used by train_policy and finetune_policy."""
    file_root, file_ext = os.path.splitext(model_path)
    best_path           = f"{file_root}_best{file_ext}"

    for epoch in range(start_epoch, start_epoch + epochs):
        avg_train, tr_pred, tr_true, tr_ego, _ = run_epoch(
            policy_model, train_loader, optimizer, device,
            use_precomputed, depth_estimator, is_train=True,
            desc=f"[{tag}] Epoch {epoch+1}/{start_epoch+epochs} [Train]",
            current_epoch=epoch,
            curriculum_epochs=curriculum_epochs,
            fully_masked_epochs=fully_masked_epochs,
            image_size=image_size
        )
        avg_val, val_pred, val_true, val_ego, val_ego_full = run_epoch(
            policy_model, val_loader, optimizer, device,
            use_precomputed, depth_estimator, is_train=False,
            desc=f"[{tag}] Epoch {epoch+1}/{start_epoch+epochs} [Val]",
            current_epoch=epoch,
            curriculum_epochs=curriculum_epochs,
            fully_masked_epochs=fully_masked_epochs,
            image_size=image_size
        )

        tr_m   = compute_offline_metrics(tr_pred, tr_true)
        val_m  = compute_offline_metrics(val_pred, val_true)
        val_pm = compute_predictive_metrics(val_pred, val_true, val_ego)
        val_hm = compute_heading_metrics(val_pred, val_true, val_ego_full)
        print(
            f"[{tag}] Epoch [{epoch+1:02d}] "
            f"Loss Tr/Val: {avg_train:.4f}/{avg_val:.4f} | "
            f"Str MAE Tr/Val: {tr_m['steering_mae']:.4f}/{val_m['steering_mae']:.4f} | "
            f"Str Dir Acc: {val_m['steering_dir_acc']:.3f} | "
            f"Brake Acc: {val_m['brake_acc']:.3f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        print(
            f"         P95 Steer Err: {val_pm['steer_p95_error']:.4f} | "
            f"Turn MAE: {val_pm['active_turn_mae']:.4f} | "
            f"Crit Turn MAE: {val_pm['critical_turn_mae']:.4f} | "
            f"Jitter: {val_pm['jitter_ratio']:.3f} | "
            f"OOB: {val_pm['out_of_bounds_rate']:.3f} | "
            f"SpeedWt MAE: {val_pm['speed_weighted_steer_mae']:.4f} | "
            f"BrakeAntic: {val_pm['pre_brake_anticipation']:.3f}"
        )
        if val_hm:
            print(
                f"         HeadDirAcc: {val_hm['heading_dir_acc']:.3f} | "
                f"HeadMAE: {val_hm['heading_delta_mae']:.4f} | "
                f"WinDiv Mean/P95: {val_hm['window_heading_div_mean']:.4f}/{val_hm['window_heading_div_p95']:.4f}"
            )
        if epoch>fully_masked_epochs:
            scheduler.step()

        save_checkpoint(policy_model, optimizer, scheduler, epoch, avg_val, model_path)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_checkpoint(policy_model, optimizer, scheduler, epoch, avg_val, best_path)
            print(f"*** Best model saved → {best_path}  (Val Loss: {best_val_loss:.4f}) ***")

    return best_val_loss