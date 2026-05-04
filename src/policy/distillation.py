"""
policy/distillation.py – Teacher-Student knowledge distillation training loops.

Both teacher and student are DrivingPolicyNet(in_channels=4), so their visual streams
are directly comparable for feature distillation.  The only difference is the input:

  Teacher input : depth + lane-masked RGB (3 ch, non-lane pixels zeroed out — always)
  Student input : depth + curriculum-blended RGB (3 ch, transitioning from masked → raw)

Both inputs are produced by apply_lane_mask(); the teacher always sets
always_lane_masked=True so alpha=0 throughout training.

    L = L_imitation + λ_output · L_output_dist + λ_feature · L_feature_dist

  L_imitation   : smooth-L1 between student actions and expert ground-truth
  L_output_dist : smooth-L1 between student actions and teacher's actions (soft targets)
  L_feature_dist: MSE between student's visual bottleneck and teacher's visual bottleneck

Setting λ_output=0 and λ_feature=0 recovers the exact baseline behaviour.
"""

import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from policy.losses import custom_driving_loss, compute_offline_metrics, compute_predictive_metrics
from policy.trainer import _apply_pixel_noise, apply_lane_mask
from utils.checkpoints import save_checkpoint


# ============================================================
# DISTILLATION LOSS
# ============================================================

class DistillationLoss:
    """
    Three-term loss for student training.

    Args:
        lambda_output  : weight for output distillation (student vs teacher actions).
                         Set to 0 to disable.
        lambda_feature : weight for feature distillation (student vs teacher bottleneck).
                         Set to 0 to disable.
    """

    def __init__(self, lambda_output: float = 1.0, lambda_feature: float = 0.1):
        self.lambda_output  = lambda_output
        self.lambda_feature = lambda_feature

    def __call__(
        self,
        student_pred:   torch.Tensor,
        student_feat:   torch.Tensor,
        teacher_pred:   torch.Tensor,
        teacher_feat:   torch.Tensor,
        expert_actions: torch.Tensor,
    ):
        L_imitation    = custom_driving_loss(student_pred, expert_actions)
        L_output_dist  = F.smooth_l1_loss(student_pred, teacher_pred.detach())
        L_feature_dist = F.mse_loss(student_feat, teacher_feat.detach())
        total = (
            L_imitation
            + self.lambda_output  * L_output_dist
            + self.lambda_feature * L_feature_dist
        )
        breakdown = {
            "imitation":    L_imitation.item(),
            "output_dist":  L_output_dist.item(),
            "feature_dist": L_feature_dist.item(),
        }
        return total, breakdown


# ============================================================
# TEACHER EPOCH  (lane-mask input, standard imitation loss)
# ============================================================

def run_teacher_epoch(
    teacher, loader, optimizer, device,
    is_train: bool, desc: str,
    image_size: int = None,
    scaler=None,
    pixel_noise_frac: float = 0.0,
    use_ego: bool = True,
    use_depth: bool = True,
):
    """One training or validation epoch for the teacher (lane-mask input)."""
    teacher.train() if is_train else teacher.eval()
    total_loss         = 0.0
    all_pred, all_true = [], []
    use_amp            = device.type == "cuda"

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=desc, leave=False):
            depth_t, rgb_np, actions_np, ego_np = batch
            depth_t   = depth_t.to(device)
            actions_t = torch.tensor(actions_np, dtype=torch.float32, device=device)
            ego_t     = torch.tensor(ego_np,     dtype=torch.float32, device=device)
            if not use_ego:
                ego_t = torch.zeros_like(ego_t)
            teacher_in = apply_lane_mask(depth_t, rgb_np, device, always_lane_masked=True,
                                         image_size=image_size, use_depth=use_depth)

            if is_train:
                if pixel_noise_frac > 0:
                    teacher_in = _apply_pixel_noise(teacher_in, pixel_noise_frac)
                optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = teacher(teacher_in, ego_t)
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

    return total_loss / max(len(loader), 1), np.concatenate(all_pred), np.concatenate(all_true)


# ============================================================
# STUDENT EPOCH  (curriculum depth+RGB input + distillation)
# ============================================================

def run_student_epoch(
    student, teacher, loader, optimizer, device,
    is_train: bool, desc: str,
    dist_loss: DistillationLoss,
    current_epoch:       int   = 999,
    curriculum_epochs:   int   = 10,
    fully_masked_epochs: int   = 3,
    image_size:          int   = None,
    scaler=None,
    lane_mask_prob:      float = 0.0,
    pixel_noise_frac:    float = 0.0,
    use_ego:             bool  = True,
    use_depth:           bool  = True,
):
    """
    One training or validation epoch for the student.

    Student input : curriculum-blended depth+RGB (4-ch) or RGB-only (3-ch) when use_depth=False.
    Teacher input : same channel config, always fully lane-masked.
    """
    student.train() if is_train else student.eval()
    teacher.eval()
    total_loss         = 0.0
    all_pred, all_true = [], []
    bd_acc             = {"imitation": 0.0, "output_dist": 0.0, "feature_dist": 0.0}
    use_amp            = device.type == "cuda"

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=desc, leave=False):
            depth_t, rgb_np, actions_np, ego_np = batch
            depth_t   = depth_t.to(device)
            actions_t = torch.tensor(actions_np, dtype=torch.float32, device=device)
            ego_t     = torch.tensor(ego_np,     dtype=torch.float32, device=device)
            if not use_ego:
                ego_t = torch.zeros_like(ego_t)

            # Student input: same curriculum pipeline as the baseline training
            force_masked = curriculum_epochs > 0 and np.random.random() < lane_mask_prob
            combined = apply_lane_mask(
                depth_t, rgb_np, device,
                current_epoch=current_epoch,
                curriculum_epochs=curriculum_epochs,
                fully_masked_epochs=fully_masked_epochs,
                image_size=image_size,
                always_lane_masked=force_masked,
                use_depth=use_depth,
            )

            # Teacher input: same channel config, always fully masked
            teacher_in = apply_lane_mask(depth_t, rgb_np, device, always_lane_masked=True,
                                         image_size=image_size, use_depth=use_depth)

            if is_train:
                if pixel_noise_frac > 0:
                    combined = _apply_pixel_noise(combined, pixel_noise_frac)
                optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                with torch.no_grad():
                    t_pred, t_feat = teacher(teacher_in, ego_t, return_features=True)
                s_pred, s_feat = student(combined, ego_t, return_features=True)
                loss, breakdown = dist_loss(s_pred, s_feat, t_pred, t_feat, actions_t)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item()
            all_pred.append(s_pred.detach().cpu().numpy())
            all_true.append(actions_np)
            for k in bd_acc:
                bd_acc[k] += breakdown[k]

    n      = max(len(loader), 1)
    avg_bd = {k: v / n for k, v in bd_acc.items()}
    return total_loss / n, np.concatenate(all_pred), np.concatenate(all_true), avg_bd


# ============================================================
# CSV LOGGING
# ============================================================

_TEACHER_CSV_FIELDS = [
    "epoch", "train_loss", "val_loss",
    "train_steering_mae", "val_steering_mae",
    "val_steering_dir_acc", "val_brake_acc",
    "val_active_turn_mae", "val_jitter_ratio", "val_steer_95th_pctl_err",
    "lr",
]

_STUDENT_CSV_FIELDS = [
    "epoch", "train_loss", "val_loss",
    "train_steering_mae", "val_steering_mae",
    "val_steering_dir_acc", "val_brake_acc",
    "val_active_turn_mae", "val_jitter_ratio", "val_steer_95th_pctl_err",
    "loss_imitation", "loss_output_dist", "loss_feature_dist",
    "lr",
]


def _log_csv(csv_path: str, fields: list, row: dict) -> None:
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# TEACHER TRAINING LOOP
# ============================================================

def train_teacher_loop(
    teacher, device,
    train_loader, val_loader,
    optimizer, scheduler,
    epochs:           int,
    model_path:       str,
    image_size:       int   = None,
    pixel_noise_frac: float = 0.0,
    patience:         int   = 8,
    use_ego:          bool  = True,
    use_depth:        bool  = True,
) -> float:
    """Full training loop for the teacher on lane masks."""
    file_root, file_ext = os.path.splitext(model_path)
    best_path    = f"{file_root}_best{file_ext}"
    csv_path     = f"{file_root}_metrics.csv"
    use_amp      = device.type == "cuda"
    scaler       = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss    = float("inf")
    no_improve_count = 0

    for epoch in range(epochs):
        avg_train, tr_pred, tr_true = run_teacher_epoch(
            teacher, train_loader, optimizer, device,
            is_train=True,
            desc=f"[Teacher] Epoch {epoch+1}/{epochs} [Train]",
            image_size=image_size, scaler=scaler,
            pixel_noise_frac=pixel_noise_frac,
            use_ego=use_ego, use_depth=use_depth,
        )
        avg_val, val_pred, val_true = run_teacher_epoch(
            teacher, val_loader, optimizer, device,
            is_train=False,
            desc=f"[Teacher] Epoch {epoch+1}/{epochs} [Val]",
            image_size=image_size,
            use_ego=use_ego, use_depth=use_depth,
        )

        tr_m   = compute_offline_metrics(tr_pred,  tr_true)
        val_m  = compute_offline_metrics(val_pred, val_true)
        pred_m = compute_predictive_metrics(val_pred, val_true)
        lr     = scheduler.get_last_lr()[0]

        print(
            f"[Teacher] Epoch [{epoch+1:02d}/{epochs}] "
            f"Loss Tr/Val: {avg_train:.4f}/{avg_val:.4f} | "
            f"Steer MAE Tr/Val: {tr_m['steering_mae']:.4f}/{val_m['steering_mae']:.4f} | "
            f"Dir Acc: {val_m['steering_dir_acc']:.3f} | "
            f"Brake Acc: {val_m['brake_acc']:.3f} | "
            f"LR: {lr:.2e}"
        )

        _log_csv(csv_path, _TEACHER_CSV_FIELDS, {
            "epoch":                   epoch + 1,
            "train_loss":              round(avg_train, 6),
            "val_loss":                round(avg_val,   6),
            "train_steering_mae":      round(tr_m["steering_mae"],          6),
            "val_steering_mae":        round(val_m["steering_mae"],         6),
            "val_steering_dir_acc":    round(val_m["steering_dir_acc"],     6),
            "val_brake_acc":           round(val_m["brake_acc"],            6),
            "val_active_turn_mae":     round(pred_m["active_turn_mae"],     6),
            "val_jitter_ratio":        round(pred_m["jitter_ratio"],        6),
            "val_steer_95th_pctl_err": round(pred_m["steer_95th_pctl_err"], 6),
            "lr":                      lr,
        })

        scheduler.step()
        save_checkpoint(teacher, optimizer, scheduler, epoch, avg_val, model_path)

        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            no_improve_count = 0
            save_checkpoint(teacher, optimizer, scheduler, epoch, avg_val, best_path)
            print(f"*** Best teacher saved → {best_path}  (Val Loss: {best_val_loss:.4f}) ***")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping: no teacher val improvement for {patience} epochs.")
                break

    return best_val_loss


# ============================================================
# STUDENT TRAINING LOOP
# ============================================================

def train_student_loop(
    student, teacher, device,
    train_loader, val_loader,
    optimizer, scheduler,
    epochs:              int,
    start_epoch:         int,
    best_val_loss:       float,
    model_path:          str,
    dist_loss:           DistillationLoss,
    curriculum_epochs:   int   = 10,
    fully_masked_epochs: int   = 3,
    image_size:          int   = None,
    lane_mask_prob:      float = 0.0,
    pixel_noise_frac:    float = 0.05,
    patience:            int   = 8,
    use_ego:             bool  = True,
    use_depth:           bool  = True,
) -> float:
    """
    Full distillation training loop for the student (DrivingPolicyNet).

    Mirrors train_loop() from trainer.py but adds teacher distillation losses on top.
    The student uses the same curriculum pipeline as baseline offline training.
    """
    file_root, file_ext = os.path.splitext(model_path)
    best_path    = f"{file_root}_best{file_ext}"
    csv_path     = f"{file_root}_metrics.csv"
    use_amp      = device.type == "cuda"
    scaler       = torch.amp.GradScaler("cuda", enabled=use_amp)

    no_improve_count = 0
    es_best_val      = float("inf")

    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    for epoch in range(start_epoch, start_epoch + epochs):
        avg_train, tr_pred, tr_true, tr_bd = run_student_epoch(
            student, teacher, train_loader, optimizer, device,
            is_train=True,
            desc=f"[Student] Epoch {epoch+1}/{start_epoch+epochs} [Train]",
            dist_loss=dist_loss,
            current_epoch=epoch,
            curriculum_epochs=curriculum_epochs,
            fully_masked_epochs=fully_masked_epochs,
            image_size=image_size, scaler=scaler,
            lane_mask_prob=lane_mask_prob,
            pixel_noise_frac=pixel_noise_frac,
            use_ego=use_ego, use_depth=use_depth,
        )
        avg_val, val_pred, val_true, val_bd = run_student_epoch(
            student, teacher, val_loader, optimizer, device,
            is_train=False,
            desc=f"[Student] Epoch {epoch+1}/{start_epoch+epochs} [Val]",
            dist_loss=dist_loss,
            current_epoch=epoch,
            curriculum_epochs=curriculum_epochs,
            fully_masked_epochs=fully_masked_epochs,
            image_size=image_size,
            lane_mask_prob=0,
            use_ego=use_ego, use_depth=use_depth,
        )

        tr_m   = compute_offline_metrics(tr_pred,  tr_true)
        val_m  = compute_offline_metrics(val_pred, val_true)
        pred_m = compute_predictive_metrics(val_pred, val_true)
        lr     = scheduler.get_last_lr()[0]

        print(
            f"[Student] Epoch [{epoch+1:02d}] "
            f"Loss Tr/Val: {avg_train:.4f}/{avg_val:.4f} | "
            f"Steer MAE Tr/Val: {tr_m['steering_mae']:.4f}/{val_m['steering_mae']:.4f} | "
            f"Dir Acc: {val_m['steering_dir_acc']:.3f} | "
            f"Brake Acc: {val_m['brake_acc']:.3f} | "
            f"Imit/OutDist/FeatDist (val): "
            f"{val_bd['imitation']:.4f}/{val_bd['output_dist']:.4f}/{val_bd['feature_dist']:.4f} | "
            f"LR: {lr:.2e}"
        )

        _log_csv(csv_path, _STUDENT_CSV_FIELDS, {
            "epoch":                   epoch + 1,
            "train_loss":              round(avg_train, 6),
            "val_loss":                round(avg_val,   6),
            "train_steering_mae":      round(tr_m["steering_mae"],          6),
            "val_steering_mae":        round(val_m["steering_mae"],         6),
            "val_steering_dir_acc":    round(val_m["steering_dir_acc"],     6),
            "val_brake_acc":           round(val_m["brake_acc"],            6),
            "val_active_turn_mae":     round(pred_m["active_turn_mae"],     6),
            "val_jitter_ratio":        round(pred_m["jitter_ratio"],        6),
            "val_steer_95th_pctl_err": round(pred_m["steer_95th_pctl_err"], 6),
            "loss_imitation":          round(val_bd["imitation"],           6),
            "loss_output_dist":        round(val_bd["output_dist"],         6),
            "loss_feature_dist":       round(val_bd["feature_dist"],        6),
            "lr":                      lr,
        })

        scheduler.step()
        save_checkpoint(student, optimizer, scheduler, epoch, avg_val, model_path)

        if avg_val < best_val_loss and epoch >= fully_masked_epochs + curriculum_epochs:
            best_val_loss = avg_val
            save_checkpoint(student, optimizer, scheduler, epoch, avg_val, best_path)
            print(f"*** Best student saved → {best_path}  (Val Loss: {best_val_loss:.4f}) ***")

        if epoch > fully_masked_epochs + curriculum_epochs:
            if avg_val < es_best_val:
                es_best_val      = avg_val
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Early stopping: no student val improvement for {patience} epochs.")
                    break

    return best_val_loss
