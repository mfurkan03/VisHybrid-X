"""
policy/trainer.py – shared epoch loop, data loader builder, and feature extractor.
"""

import csv
import os
import cv2

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import EGO_DIM
from policy.datasets import MetaDriveRGBDataset, PrecomputedDepthDataset
from policy.losses import (custom_driving_loss_beta, compute_offline_metrics,
                           compute_predictive_metrics, compute_heading_metrics)
from utils.checkpoints import save_checkpoint
from utils.seed import worker_init_fn

# ============================================================
# FEATURE EXTRACTION  (live DPT inference path)
# ============================================================
def get_lane_mask_visual(rgb_image: np.ndarray, threshold_value: int = 180) -> np.ndarray:
    img_uint8 = (rgb_image * 255.0).astype(np.uint8) if rgb_image.max() <= 1.0 else rgb_image.astype(np.uint8)
    h, w      = img_uint8.shape[:2]
    roi       = img_uint8.copy()
    roi[0:int(h * 0.55), :] = 0
    gray      = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    _, white  = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    hsv       = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    yellow    = cv2.inRange(hsv, np.array([15, 80, 165]), np.array([35, 255, 255]))
    return cv2.bitwise_or(white, yellow)

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
    """
    Applies lane masking at original resolution and interpolates the final result.
    """

    # 1. Calculate Alpha for Curriculum Learning
    if always_lane_masked:
        alpha = 0.0
    elif current_epoch < fully_masked_epochs:
        alpha = 0.0
    elif current_epoch >= fully_masked_epochs + curriculum_epochs:
        alpha = 1.0
    else:
        alpha = (current_epoch - fully_masked_epochs) / max(1, curriculum_epochs)

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

def apply_augmentations(
    combined: torch.Tensor,
    actions_np: np.ndarray,
    ego_np: np.ndarray,
    prob_pixel_noise: float = 1.0,
    prob_hflip: float = 0.5,
    prob_grayscale: float = 0.1,
) -> tuple:
    """
    Per-sample stochastic augmentations applied to the combined (4, H, W) tensor.

    - Pixel noise  : Gaussian noise (std=0.05) on 10 % of pixels, RGB channels only.
    - Horizontal flip : flips image + negates steer, last_steer, heading_delta,
                        and swaps the navi_left/navi_right command (a left turn
                        in the mirrored frame becomes a right turn).
    - Grayscale    : RGB → luminance, depth channel untouched.

    prob_* is the per-sample probability; 0.0 disables the augmentation entirely.
    """
    actions_out = actions_np.copy()
    ego_out     = ego_np.copy()
    imgs        = list(combined.unbind(0))

    for i, img in enumerate(imgs):
        img = img.clone()
        H, W = img.shape[1], img.shape[2]

        if prob_pixel_noise > 0.0 and np.random.random() < prob_pixel_noise:
            n_pix  = max(1, int(0.10 * H * W))
            ys     = np.random.randint(0, H, n_pix)
            xs     = np.random.randint(0, W, n_pix)
            noise  = torch.zeros_like(img)
            noise[1:4, ys, xs] = torch.tensor(
                np.random.normal(0.0, 0.05, (3, n_pix)).astype(np.float32),
                device=img.device,
            )
            img = (img + noise).clamp(0.0, 1.0)

        if prob_hflip > 0.0 and np.random.random() < prob_hflip:
            img = torch.flip(img, dims=[2])
            actions_out[i, 0] *= -1
            if ego_out.shape[1] > 1:
                ego_out[i, 1] *= -1   # last_steer
            if ego_out.shape[1] > 2:
                ego_out[i, 2] *= -1   # heading_delta
            if ego_out.shape[1] > 4:
                # swap navi_left (idx 3) <-> navi_right (idx 4)
                ego_out[i, 3], ego_out[i, 4] = ego_out[i, 4], ego_out[i, 3]

        if prob_grayscale > 0.0 and np.random.random() < prob_grayscale:
            gray   = 0.299 * img[1] + 0.587 * img[2] + 0.114 * img[3]
            img[1] = gray
            img[2] = gray
            img[3] = gray

        imgs[i] = img

    return torch.stack(imgs), actions_out, ego_out


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
def build_loaders(use_precomputed: bool, pred_dir, data_dir, batch_size, depth_estimator,
                  seed: int = 0, nav_dir: str = None):
    if use_precomputed:
        train_ds = PrecomputedDepthDataset(pred_dir=pred_dir, split="train", nav_dir=nav_dir)
        val_ds   = PrecomputedDepthDataset(pred_dir=pred_dir, split="val",   nav_dir=nav_dir)

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
        train_ds = MetaDriveRGBDataset(data_dir=data_dir, split="train", nav_dir=nav_dir)
        val_ds   = MetaDriveRGBDataset(data_dir=data_dir, split="val",   nav_dir=nav_dir)

        def collate_fn(batch):
            rgbs, actions, egos = zip(*batch)
            n = len(actions)
            return (np.stack(rgbs), np.stack(actions), np.stack(egos),
                    np.zeros((n, 5), dtype=np.float32))

    from torch.utils.data import WeightedRandomSampler
    actions_np = np.array(train_ds.actions)
    steer_mag  = np.abs(actions_np[:, 0])

    bins        = np.digitize(steer_mag, [0.05, 0.2])  # 0: straight, 1: turning, 2: intersection
    bin_targets = {0: 0.48, 1: 0.45, 2: 0.05}
    weights     = np.zeros(len(steer_mag), dtype=np.float64)
    for b in np.unique(bins):
        mask          = bins == b
        weights[mask] = bin_targets[b] / mask.sum()

    sampler = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float64),
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              collate_fn=collate_fn, worker_init_fn=worker_init_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, worker_init_fn=worker_init_fn)
    return train_loader, val_loader


# ============================================================
# SINGLE EPOCH
# ============================================================
def run_epoch(policy_model, loader, optimizer, device,
              use_precomputed, depth_estimator, is_train, desc,
              current_epoch: int = 999, curriculum_epochs: int = 10, fully_masked_epochs: int = 3,
              image_size: int = None, always_lane_masked: bool = False,
              prob_pixel_noise: float = 0.0, prob_hflip: float = 0.0, prob_grayscale: float = 0.0):
    """Run one training or validation epoch. Returns (avg_loss, preds, trues, ego_states, ego_fulls)."""
    policy_model.train() if is_train else policy_model.eval()
    total_loss                           = 0.0
    all_pred, all_true, all_ego, all_ego_full = [], [], [], []

    aug_active = is_train and current_epoch >= fully_masked_epochs

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=desc, leave=False):
            if use_precomputed:
                depth_t, rgb_np, actions_np, ego_np, ego_full_np = batch
                depth_t  = depth_t.to(device)
                combined = apply_lane_mask(
                    depth_t, rgb_np, device,
                    current_epoch=current_epoch,
                    curriculum_epochs=curriculum_epochs,
                    fully_masked_epochs=fully_masked_epochs,
                    image_size=image_size,
                    always_lane_masked=always_lane_masked,
                )
            else:
                rgb_np, actions_np, ego_np, ego_full_np = batch
                combined, _ = extract_features_frozen(
                    rgb_np, depth_estimator, device,
                    current_epoch=current_epoch,
                    curriculum_epochs=curriculum_epochs,
                    fully_masked_epochs=fully_masked_epochs,
                    image_size=image_size,
                    always_lane_masked=always_lane_masked,
                )

            if aug_active:
                combined, actions_np, ego_np = apply_augmentations(
                    combined, actions_np, ego_np,
                    prob_pixel_noise=prob_pixel_noise,
                    prob_hflip=prob_hflip,
                    prob_grayscale=prob_grayscale,
                )

            actions_t  = torch.tensor(actions_np, dtype=torch.float32, device=device)
            ego_t      = torch.tensor(ego_np,     dtype=torch.float32, device=device)
            # Map [-1,1] → [0,1]; clip raw MetaDrive actions that fall outside [-1,1]
            actions_01 = ((actions_t.clamp(-1.0, 1.0) + 1.0) / 2.0)

            if is_train:
                optimizer.zero_grad()

            pred_alpha, pred_beta = policy_model(combined, ego_t)
            loss = custom_driving_loss_beta(pred_alpha, pred_beta, actions_01)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                mean_01    = pred_alpha / (pred_alpha + pred_beta)   # Beta mean; exact for throttle (= mu), close to mode for steer
                pred_mean  = (mean_01 * 2.0 - 1.0).cpu().numpy()   # back to [-1, 1]
            all_pred.append(pred_mean)
            all_true.append(actions_np)
            all_ego.append(ego_np)
            all_ego_full.append(ego_full_np)

    avg_loss = total_loss / max(len(loader), 1)
    if not all_pred:
        empty = np.zeros((0, 2), dtype=np.float32)
        return avg_loss, empty, empty, np.zeros((0, EGO_DIM), dtype=np.float32), np.zeros((0, 5), dtype=np.float32)
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
    curriculum_epochs: int = None,
    fully_masked_epochs: int = None,
    image_size: int = None,
    always_lane_masked: bool = False,
    early_stopping_patience: int = None,
    early_stopping_min_delta: float = None,
    prob_pixel_noise: float = 0.0,
    prob_hflip: float = 0.0,
    prob_grayscale: float = 0.0,
) -> float:
    """Shared epoch loop used by train_policy and finetune_policy."""
    file_root, file_ext = os.path.splitext(model_path)
    best_path           = f"{file_root}_best{file_ext}"

    no_improve_count = 0
    best_epoch       = -1
    best_metrics     = None

    csv_path    = f"{file_root}_metrics.csv"
    csv_headers = [
        "epoch", "train_loss", "val_loss", "lr",
        "steering_mae_train", "accel_mae_train",
        "steering_mae_val", "accel_mae_val", "steering_mse_val", "accel_mse_val",
        "steering_dir_acc_val", "direction_acc_val", "brake_acc_val", "steering_corr_val",
        "steer_p95_error", "active_turn_mae", "critical_turn_mae",
        "jitter_ratio", "out_of_bounds_rate", "speed_weighted_steer_mae", "pre_brake_anticipation",
        "heading_dir_acc", "heading_delta_mae", "window_heading_div_mean", "window_heading_div_p95",
    ]
    write_header = not os.path.exists(csv_path)
    csv_file     = open(csv_path, "a", newline="")
    csv_writer   = csv.DictWriter(csv_file, fieldnames=csv_headers)
    if write_header:
        csv_writer.writeheader()

    for epoch in range(start_epoch, start_epoch + epochs):
        avg_train, tr_pred, tr_true, tr_ego, _ = run_epoch(
            policy_model, train_loader, optimizer, device,
            use_precomputed, depth_estimator, is_train=True,
            desc=f"[{tag}] Epoch {epoch+1}/{start_epoch+epochs} [Train]",
            current_epoch=epoch,
            curriculum_epochs=curriculum_epochs,
            fully_masked_epochs=fully_masked_epochs,
            image_size=image_size,
            always_lane_masked=always_lane_masked,
            prob_pixel_noise=prob_pixel_noise,
            prob_hflip=prob_hflip,
            prob_grayscale=prob_grayscale,
        )
        avg_val, val_pred, val_true, val_ego, val_ego_full = run_epoch(
            policy_model, val_loader, optimizer, device,
            use_precomputed, depth_estimator, is_train=False,
            desc=f"[{tag}] Epoch {epoch+1}/{start_epoch+epochs} [Val]",
            current_epoch=epoch,
            curriculum_epochs=curriculum_epochs,
            fully_masked_epochs=fully_masked_epochs,
            image_size=image_size,
            always_lane_masked=always_lane_masked,
        )

        tr_m   = compute_offline_metrics(tr_pred, tr_true)
        val_empty = len(val_pred) == 0
        val_m  = compute_offline_metrics(val_pred, val_true) if not val_empty else {k: 0.0 for k in ["steering_mae","accel_mae","steering_mse","accel_mse","steering_dir_acc","direction_acc","brake_acc","steering_corr"]}
        val_pm = compute_predictive_metrics(val_pred, val_true, val_ego) if not val_empty else {k: 0.0 for k in ["steer_p95_error","active_turn_mae","critical_turn_mae","jitter_ratio","out_of_bounds_rate","speed_weighted_steer_mae","pre_brake_anticipation"]}
        val_hm = compute_heading_metrics(val_pred, val_true, val_ego_full) if not val_empty else {}

        # ── Per-epoch mean output vs expert (diagnose collapsed/biased predictions) ──
        tr_pred_accel  = tr_pred[:, 1];  tr_true_accel  = tr_true[:, 1]
        val_pred_accel = val_pred[:, 1] if not val_empty else np.array([])
        val_true_accel = val_true[:, 1] if not val_empty else np.array([])
        tr_brake_mask  = tr_true_accel < -0.05
        print(
            f"[{tag}] Epoch [{epoch+1:02d}] "
            f"Loss Tr/Val: {avg_train:.4f}/{avg_val:.4f} | "
            f"Str MAE Tr/Val: {tr_m['steering_mae']:.4f}/{val_m['steering_mae']:.4f} | "
            f"Str Dir Acc: {val_m['steering_dir_acc']:.3f} | "
            f"Brake Acc: {val_m['brake_acc']:.3f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        print(
            f"         [ACCEL]  Expert mean: {tr_true_accel.mean():+.4f}  "
            f"Pred mean: {tr_pred_accel.mean():+.4f}  "
            f"Pred min/max: {tr_pred_accel.min():+.4f}/{tr_pred_accel.max():+.4f}  "
            f"Pred<-0.05: {(tr_pred_accel<-0.05).mean()*100:.3f}%  "
            f"(expert brake samples: {tr_brake_mask.sum()}/{len(tr_brake_mask)})"
        )
        if tr_brake_mask.any():
            print(
                f"         [BRAKE]  On brake obs: expert={tr_true_accel[tr_brake_mask].mean():+.4f}  "
                f"pred={tr_pred_accel[tr_brake_mask].mean():+.4f}"
            )
        if not val_empty and len(val_pred_accel):
            print(
                f"         [VAL]    Expert mean: {val_true_accel.mean():+.4f}  "
                f"Pred mean: {val_pred_accel.mean():+.4f}  "
                f"Pred<-0.05: {(val_pred_accel<-0.05).mean()*100:.3f}%"
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
        current_lr = scheduler.get_last_lr()[0]
        csv_writer.writerow({
            "epoch":                  epoch + 1,
            "train_loss":             round(avg_train, 6),
            "val_loss":               round(avg_val,   6),
            "lr":                     current_lr,
            "steering_mae_train":     round(tr_m["steering_mae"],      6),
            "accel_mae_train":        round(tr_m["accel_mae"],         6),
            "steering_mae_val":       round(val_m["steering_mae"],     6),
            "accel_mae_val":          round(val_m["accel_mae"],        6),
            "steering_mse_val":       round(val_m["steering_mse"],     6),
            "accel_mse_val":          round(val_m["accel_mse"],        6),
            "steering_dir_acc_val":   round(val_m["steering_dir_acc"], 6),
            "direction_acc_val":      round(val_m["direction_acc"],    6),
            "brake_acc_val":          round(val_m["brake_acc"],        6),
            "steering_corr_val":      round(val_m["steering_corr"],    6),
            "steer_p95_error":        round(val_pm["steer_p95_error"],           6),
            "active_turn_mae":        round(val_pm["active_turn_mae"],           6),
            "critical_turn_mae":      round(val_pm["critical_turn_mae"],         6),
            "jitter_ratio":           round(val_pm["jitter_ratio"],              6),
            "out_of_bounds_rate":     round(val_pm["out_of_bounds_rate"],        6),
            "speed_weighted_steer_mae": round(val_pm["speed_weighted_steer_mae"],6),
            "pre_brake_anticipation": round(val_pm["pre_brake_anticipation"],    6),
            "heading_dir_acc":          round(val_hm["heading_dir_acc"],           6) if val_hm else "",
            "heading_delta_mae":        round(val_hm["heading_delta_mae"],         6) if val_hm else "",
            "window_heading_div_mean":  round(val_hm["window_heading_div_mean"],   6) if val_hm else "",
            "window_heading_div_p95":   round(val_hm["window_heading_div_p95"],    6) if val_hm else "",
        })
        csv_file.flush()

        scheduler.step()
        past_curriculum = epoch >= fully_masked_epochs + curriculum_epochs

        save_checkpoint(policy_model, optimizer, scheduler, epoch, avg_val, model_path)
        if val_empty:
            continue
        if past_curriculum and avg_val < best_val_loss - early_stopping_min_delta:
            best_val_loss    = avg_val
            best_epoch       = epoch + 1
            best_metrics     = dict(
                train_loss=avg_train, val_loss=avg_val,
                tr_m=tr_m, val_m=val_m, val_pm=val_pm, val_hm=val_hm,
            )
            no_improve_count = 0
            save_checkpoint(policy_model, optimizer, scheduler, epoch, avg_val, best_path)
            print(f"*** Best model saved → {best_path}  (Val Loss: {best_val_loss:.4f}) ***")
        elif past_curriculum:
            no_improve_count += 1
            if early_stopping_patience > 0:
                print(f"    [EarlyStopping] No improvement for {no_improve_count}/{early_stopping_patience} epochs")
            if early_stopping_patience > 0 and no_improve_count >= early_stopping_patience:
                print(f"[{tag}] Early stopping triggered at epoch {epoch+1}.")
                break

    csv_file.close()
    print(f"[{tag}] Metrics saved → {csv_path}")

    sep = "=" * 64
    if best_metrics:
        bm  = best_metrics
        print(f"\n{sep}")
        print(f"[{tag}] TRAINING COMPLETE  —  Best checkpoint: epoch {best_epoch}")
        print(sep)
        print(f"  Loss  Train / Val    : {bm['train_loss']:.4f} / {bm['val_loss']:.4f}")
        print(f"  Steer MAE  Tr / Val  : {bm['tr_m']['steering_mae']:.4f} / {bm['val_m']['steering_mae']:.4f}")
        print(f"  Accel MAE  Tr / Val  : {bm['tr_m']['accel_mae']:.4f} / {bm['val_m']['accel_mae']:.4f}")
        print(f"  Steer MSE  (val)     : {bm['val_m']['steering_mse']:.4f}")
        print(f"  Accel MSE  (val)     : {bm['val_m']['accel_mse']:.4f}")
        print(f"  Steer Dir Acc (val)  : {bm['val_m']['steering_dir_acc']*100:.1f}%")
        print(f"  Accel Dir Acc (val)  : {bm['val_m']['direction_acc']*100:.1f}%")
        print(f"  Braking Acc   (val)  : {bm['val_m']['brake_acc']*100:.1f}%")
        print(f"  Steering Corr (val)  : {bm['val_m']['steering_corr']:.4f}")
        print(f"  --- Predictive ---")
        print(f"  P95 Steer Err        : {bm['val_pm']['steer_p95_error']:.4f}")
        print(f"  Active Turn MAE      : {bm['val_pm']['active_turn_mae']:.4f}")
        print(f"  Critical Turn MAE    : {bm['val_pm']['critical_turn_mae']:.4f}")
        print(f"  Jitter Ratio         : {bm['val_pm']['jitter_ratio']:.3f}  (1.0=expert)")
        print(f"  Out-of-Bounds Rate   : {bm['val_pm']['out_of_bounds_rate']*100:.1f}%")
        print(f"  Speed-Weighted MAE   : {bm['val_pm']['speed_weighted_steer_mae']:.4f}")
        print(f"  Pre-Brake Anticipation: {bm['val_pm']['pre_brake_anticipation']*100:.1f}%")
        if bm['val_hm']:
            print(f"  --- Heading ---")
            print(f"  Heading Dir Acc      : {bm['val_hm']['heading_dir_acc']*100:.1f}%")
            print(f"  Heading Delta MAE    : {bm['val_hm']['heading_delta_mae']:.4f}")
            print(f"  Window Div Mean/P95  : {bm['val_hm']['window_heading_div_mean']:.4f} / {bm['val_hm']['window_heading_div_p95']:.4f}")
        print(sep)
    else:
        print(f"\n{sep}")
        print(f"[{tag}] TRAINING COMPLETE  —  No best checkpoint saved (curriculum not yet passed or no improvement).")
        print(sep)

    return best_val_loss