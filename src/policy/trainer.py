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
from policy.losses import custom_driving_loss, compute_offline_metrics,compute_predictive_metrics
from utils.checkpoints import save_checkpoint


# ============================================================
# FEATURE EXTRACTION  (live DPT inference path)
# ============================================================
def _batch_lane_mask(rgb_tensor: torch.Tensor, threshold: float = 180 / 255.0) -> torch.Tensor:
    """Vectorized lane mask for a whole batch — no per-image loops, no OpenCV."""
    _, _, H, _ = rgb_tensor.shape
    r, g, b = rgb_tensor[:, 0], rgb_tensor[:, 1], rgb_tensor[:, 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray[:, :int(H * 0.55), :] = 0.0
    white_mask = gray >= threshold

    # Golden + greyish-yellow: R > G is the core discriminator (road grey has R≈G, grass has G>R).
    # (r - g) < 0.35 excludes orange/red. r > b avoids cool greys. Small margin on r > g + 0.02
    # prevents floating-point ties with pure grey pixels from sneaking through.
    yellow_mask = (r > g + 0.02) & (r > b) & ((r - g) < 0.35) & (r > 0.28)
    yellow_mask[:, :int(H * 0.55), :] = False

    return (white_mask | yellow_mask).float().unsqueeze(1)  # (B, 1, H, W)


def apply_lane_mask(
    depth_tensor: torch.Tensor,
    rgb_batch: np.ndarray,
    device: torch.device,
    current_epoch: int = 999,
    curriculum_epochs: int = 10,
    fully_masked_epochs: int = 3,
    image_size: int = None,
    always_lane_masked: bool = False,
) -> torch.Tensor:
    if always_lane_masked and curriculum_epochs > 0:
        alpha = 0.0
    elif current_epoch < fully_masked_epochs:
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

    mask = _batch_lane_mask(rgb_tensor)                              # (B, 1, H, W)

    blended = rgb_tensor * mask + rgb_tensor * (1.0 - mask) * alpha

    return torch.cat([depth_tensor, blended], dim=1)               # (B, 4, H, W)


def extract_features_frozen(
    rgb_batch,
    depth_estimator,
    device,
    current_epoch: int = 999,
    curriculum_epochs: int = 10,
    fully_masked_epochs: int = 3,
    image_size: int = None,
    always_lane_masked: bool = False,
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
        image_size=image_size,
        always_lane_masked=always_lane_masked,
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
              image_size: int = None, scaler=None, lane_mask_prob: float = 0.0):
    """Run one training or validation epoch. Returns (avg_loss, preds, trues)."""
    policy_model.train() if is_train else policy_model.eval()
    total_loss         = 0.0
    all_pred, all_true = [], []
    use_amp            = device.type == "cuda"

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=desc, leave=False):
            force_masked = curriculum_epochs > 0 and np.random.random() < lane_mask_prob
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
                    image_size=image_size,
                    always_lane_masked=force_masked,
                )
            else:
                rgb_np, actions_np, ego_np = batch
                actions_t   = torch.tensor(actions_np, dtype=torch.float32, device=device)
                combined, _ = extract_features_frozen(
                    rgb_np, depth_estimator, device,
                    current_epoch=current_epoch,
                    curriculum_epochs=curriculum_epochs,
                    fully_masked_epochs=fully_masked_epochs,
                    image_size=image_size,
                    always_lane_masked=force_masked,
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
    lane_mask_prob: float = 0.0,
) -> float:
    """Shared epoch loop used by train_policy and finetune_policy."""
    file_root, file_ext = os.path.splitext(model_path)
    best_path           = f"{file_root}_best{file_ext}"
    use_amp             = device.type == "cuda"
    scaler              = torch.amp.GradScaler("cuda", enabled=use_amp)

    patience            = 8
    no_improve_count    = 0
    es_best_val_loss    = float("inf")

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
            lane_mask_prob=lane_mask_prob,
        )
        avg_val, val_pred, val_true = run_epoch(
            policy_model, val_loader, optimizer, device,
            use_precomputed, depth_estimator, is_train=False,
            desc=f"[{tag}] Epoch {epoch+1}/{start_epoch+epochs} [Val]",
            current_epoch=epoch,
            curriculum_epochs=curriculum_epochs,
            fully_masked_epochs=fully_masked_epochs,
            image_size=image_size,
            lane_mask_prob=0,
        )

        tr_m  = compute_offline_metrics(tr_pred,  tr_true)
        val_m = compute_offline_metrics(val_pred, val_true)
        val_pred_m = compute_predictive_metrics(val_pred, val_true) # NEW

        print(
            f"[{tag}] Epoch [{epoch+1:02d}] "
            f"Loss Tr/Val: {avg_train:.4f}/{avg_val:.4f} | "
            f"Str MAE Tr/Val: {tr_m['steering_mae']:.4f}/{val_m['steering_mae']:.4f} | "
            f"Str Dir Acc: {val_m['steering_dir_acc']:.3f} | "
            f"Brake Acc: {val_m['brake_acc']:.3f} | "
            f"Turn MAE: {val_pred_m['active_turn_mae']:.4f} | "  
            f"Jitter: {val_pred_m['jitter_ratio']:.2f}x | "      
            f"95th Pctl Error:  {val_pred_m['steer_95th_pctl_err']:.4f}"
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        
        scheduler.step()

        save_checkpoint(policy_model, optimizer, scheduler, epoch, avg_val, model_path)
        if avg_val < best_val_loss and epoch>=fully_masked_epochs+curriculum_epochs:
            best_val_loss = avg_val
            save_checkpoint(policy_model, optimizer, scheduler, epoch, avg_val, best_path)
            print(f"*** Best model saved → {best_path}  (Val Loss: {best_val_loss:.4f}) ***")
        if epoch >fully_masked_epochs+curriculum_epochs:
            if avg_val < es_best_val_loss:
                es_best_val_loss = avg_val
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Early stopping: no val loss improvement for {patience} epochs.")
                    break

    return best_val_loss