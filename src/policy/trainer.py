"""
policy/trainer.py – shared epoch loop, data loader builder, and feature extractor.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import EGO_DIM
from policy.datasets import MetaDriveRGBDataset, PrecomputedDepthDataset
from policy.losses import custom_driving_loss, compute_offline_metrics
from utils.checkpoints import save_checkpoint


# ============================================================
# FEATURE EXTRACTION  (live DPT inference path)
# ============================================================
def _batch_lane_mask(rgb_tensor: torch.Tensor, threshold: float = 180 / 255.0) -> torch.Tensor:
    """Vectorized lane mask for a whole batch — no per-image loops, no OpenCV."""
    _, _, H, _ = rgb_tensor.shape
    gray = 0.299 * rgb_tensor[:, 0] + 0.587 * rgb_tensor[:, 1] + 0.114 * rgb_tensor[:, 2]
    gray[:, :int(H * 0.55), :] = 0.0
    return (gray >= threshold).float().unsqueeze(1)  # (B, 1, H, W)


def apply_lane_mask(
    depth_tensor: torch.Tensor,
    rgb_batch: np.ndarray,
    device: torch.device,
    current_epoch: int = 999,
    curriculum_epochs: int = 10,
    fully_masked_epochs: int = 3,
    image_size: int = None
) -> torch.Tensor:
    if current_epoch < fully_masked_epochs:
        alpha = 0.0
    elif current_epoch >= curriculum_epochs:
        alpha = 1.0
    else:
        alpha = (current_epoch - fully_masked_epochs) / max(1, curriculum_epochs - fully_masked_epochs)

    rgb_tensor = torch.from_numpy(rgb_batch).float().to(device)
    if rgb_tensor.max() > 1.0:
        rgb_tensor = rgb_tensor / 255.0
    rgb_tensor = rgb_tensor.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

    if image_size:
        rgb_tensor   = F.interpolate(rgb_tensor,   size=(image_size, image_size), mode='bilinear', align_corners=False)
        depth_tensor = F.interpolate(depth_tensor, size=(image_size, image_size), mode='bilinear', align_corners=False)

    mask    = _batch_lane_mask(rgb_tensor)                          # (B, 1, H, W)
    blended = rgb_tensor * mask + rgb_tensor * (1.0 - mask) * alpha

    return torch.cat([depth_tensor, blended], dim=1)               # (B, 4, H, W)


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
# MODULE-LEVEL COLLATE FUNCTIONS  (must be at module level for
# pickling when num_workers > 0 on Windows spawn)
# ============================================================
def _collate_precomputed(batch):
    depths, rgbs, actions, egos = zip(*batch)
    return (
        torch.tensor(np.stack(depths), dtype=torch.float32),
        np.stack(rgbs),
        np.stack(actions),
        np.stack(egos),
    )


def _collate_rgb(batch):
    rgbs, actions, egos = zip(*batch)
    return np.stack(rgbs), np.stack(actions), np.stack(egos)


# ============================================================
# DATA LOADER BUILDER
# ============================================================
def build_loaders(use_precomputed: bool, pred_dir, data_dir, batch_size):
    if use_precomputed:
        train_ds   = PrecomputedDepthDataset(pred_dir=pred_dir, split="train")
        val_ds     = PrecomputedDepthDataset(pred_dir=pred_dir, split="val")
        collate_fn = _collate_precomputed
    else:
        train_ds   = MetaDriveRGBDataset(data_dir=data_dir, split="train")
        val_ds     = MetaDriveRGBDataset(data_dir=data_dir, split="val")
        collate_fn = _collate_rgb

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader


# ============================================================
# SINGLE EPOCH
# ============================================================
def run_epoch(policy_model, loader, optimizer, device,
              use_precomputed, depth_estimator, is_train, desc,
              current_epoch: int = 999, curriculum_epochs: int = 10, fully_masked_epochs: int = 3,
              image_size: int = None, scaler=None):
    """Run one training or validation epoch. Returns (avg_loss, preds, trues)."""
    policy_model.train() if is_train else policy_model.eval()
    total_loss         = 0.0
    all_pred, all_true = [], []
    use_amp            = device.type == "cuda"

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=desc, leave=False):
            if use_precomputed:
                depth_t, rgb_np, actions_np, ego_np = batch
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
                rgb_np, actions_np, ego_np = batch
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

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = policy_model(combined, ego_t)
                loss = custom_driving_loss(pred, actions_t)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
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
    curriculum_epochs: int = 10,
    fully_masked_epochs: int = 3,
    image_size: int = None,
) -> float:
    """Shared epoch loop used by train_policy and finetune_policy."""
    file_root, file_ext = os.path.splitext(model_path)
    best_path           = f"{file_root}_best{file_ext}"
    use_amp             = device.type == "cuda"
    scaler              = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(start_epoch, start_epoch + epochs):
        avg_train, tr_pred, tr_true = run_epoch(
            policy_model, train_loader, optimizer, device,
            use_precomputed, depth_estimator, is_train=True,
            desc=f"[{tag}] Epoch {epoch+1}/{start_epoch+epochs} [Train]",
            current_epoch=epoch,
            curriculum_epochs=curriculum_epochs,
            fully_masked_epochs=fully_masked_epochs,
            image_size=image_size,
            scaler=scaler,
        )
        avg_val, val_pred, val_true = run_epoch(
            policy_model, val_loader, optimizer, device,
            use_precomputed, depth_estimator, is_train=False,
            desc=f"[{tag}] Epoch {epoch+1}/{start_epoch+epochs} [Val]",
            current_epoch=epoch,
            curriculum_epochs=curriculum_epochs,
            fully_masked_epochs=fully_masked_epochs,
            image_size=image_size,
        )

        tr_m  = compute_offline_metrics(tr_pred,  tr_true)
        val_m = compute_offline_metrics(val_pred, val_true)
        print(
            f"[{tag}] Epoch [{epoch+1:02d}] "
            f"Loss Tr/Val: {avg_train:.4f}/{avg_val:.4f} | "
            f"Str MAE Tr/Val: {tr_m['steering_mae']:.4f}/{val_m['steering_mae']:.4f} | "
            f"Str Dir Acc: {val_m['steering_dir_acc']:.3f} | "
            f"Brake Acc: {val_m['brake_acc']:.3f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        if epoch>fully_masked_epochs:
            scheduler.step()

        save_checkpoint(policy_model, optimizer, scheduler, epoch, avg_val, model_path)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_checkpoint(policy_model, optimizer, scheduler, epoch, avg_val, best_path)
            print(f"*** Best model saved → {best_path}  (Val Loss: {best_val_loss:.4f}) ***")

    return best_val_loss