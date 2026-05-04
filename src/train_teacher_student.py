"""
train_teacher_student.py – Knowledge-distillation pipeline entry point.

Requires precomputed DPT depth data (--pred_dir).

Phase 1 – Train the teacher on clean lane masks:
    python src/train_teacher_student.py --mode train_teacher \\
        --pred_dir data/processed/dpt_pred \\
        --teacher_path models/teacher.pth --epochs 30

Phase 2 – Train the student (DrivingPolicyNet) with curriculum + distillation:
    python src/train_teacher_student.py --mode train_student \\
        --pred_dir data/processed/dpt_pred \\
        --teacher_path models/teacher_best.pth \\
        --student_path models/student.pth \\
        --epochs 70 --curriculum_epochs 40 --fully_masked_epochs 8 \\
        --lambda_output 1.0 --lambda_feature 0.1

The student uses the identical 4-ch depth+RGB curriculum pipeline as
train_offline_policy.py.  Setting --lambda_output 0 --lambda_feature 0
recovers the exact baseline behaviour.
"""

import argparse
import math

import torch
import torch.optim as optim

from models import DrivingPolicyNet
from policy.distillation import DistillationLoss, train_teacher_loop, train_student_loop
from policy.trainer import build_loaders
from utils.checkpoints import load_checkpoint


def _curriculum_lr_lambda(fully_masked_epochs: int, total_epochs: int):
    """Same LR schedule used in train_offline_policy.py."""
    def lr_lambda(epoch: int) -> float:
        if epoch < fully_masked_epochs:
            return 1.0
        base_factor = 0.1
        min_factor  = 0.01
        T_max        = total_epochs - fully_masked_epochs
        if T_max <= 0:
            return base_factor
        current_step = epoch - fully_masked_epochs
        return min_factor + 0.5 * (base_factor - min_factor) * (1 + math.cos(math.pi * current_step / T_max))
    return lr_lambda


def _cosine_lr_lambda(epochs: int):
    def lr_lambda(epoch: int) -> float:
        return 0.01 + 0.5 * (1.0 - 0.01) * (1 + math.cos(math.pi * epoch / max(epochs, 1)))
    return lr_lambda


# ============================================================
# PHASE 1: TEACHER
# ============================================================

def train_teacher(args):
    print("--- Phase 1: Training Teacher on Lane Masks ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    train_loader, val_loader = build_loaders(
        use_precomputed=True, pred_dir=args.pred_dir,
        data_dir=args.data_dir, batch_size=args.batch_size,
    )

    teacher   = DrivingPolicyNet(in_channels=2, image_size=args.image_size).to(device)
    optimizer = optim.AdamW(teacher.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_cosine_lr_lambda(args.epochs))

    train_teacher_loop(
        teacher, device, train_loader, val_loader,
        optimizer, scheduler,
        epochs=args.epochs,
        model_path=args.teacher_path,
        image_size=args.image_size,
        pixel_noise_frac=args.pixel_noise_frac,
        patience=args.patience,
    )


# ============================================================
# PHASE 2: STUDENT (curriculum + distillation)
# ============================================================

def train_student(args):
    print("--- Phase 2: Training Student (DrivingPolicyNet) with Distillation ---")
    if not args.teacher_path:
        raise ValueError("--teacher_path is required for train_student mode")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")
    print(f"λ_output={args.lambda_output}  λ_feature={args.lambda_feature}")

    train_loader, val_loader = build_loaders(
        use_precomputed=True, pred_dir=args.pred_dir,
        data_dir=args.data_dir, batch_size=args.batch_size,
    )

    teacher = DrivingPolicyNet(in_channels=2, image_size=args.image_size).to(device)
    load_checkpoint(args.teacher_path, teacher, device=device)
    print(f"[INFO] Teacher loaded from {args.teacher_path}")

    student   = DrivingPolicyNet(image_size=args.image_size).to(device)
    optimizer = optim.AdamW(student.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_curriculum_lr_lambda(args.fully_masked_epochs, args.epochs),
    )

    start_epoch = 0
    best_val    = float("inf")
    if args.resume_from:
        start_epoch, best_val = load_checkpoint(
            args.resume_from, student, optimizer=optimizer, scheduler=scheduler, device=device,
        )
        print(f"[INFO] Resuming student from {args.resume_from} (epoch {start_epoch})")

    dist_loss = DistillationLoss(
        lambda_output=args.lambda_output,
        lambda_feature=args.lambda_feature,
    )

    train_student_loop(
        student, teacher, device, train_loader, val_loader,
        optimizer, scheduler,
        epochs=args.epochs,
        start_epoch=start_epoch,
        best_val_loss=best_val,
        model_path=args.student_path,
        dist_loss=dist_loss,
        curriculum_epochs=args.curriculum_epochs,
        fully_masked_epochs=args.fully_masked_epochs,
        image_size=args.image_size,
        lane_mask_prob=args.lane_mask_prob,
        pixel_noise_frac=args.pixel_noise_frac,
        patience=args.patience,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teacher-Student Knowledge Distillation")
    parser.add_argument("--mode",          type=str,   required=True, choices=["train_teacher", "train_student"])
    parser.add_argument("--pred_dir",      type=str,   default="data/processed/dpt_pred",
                        help="Root of precomputed DPT dataset (must contain train/ and val/)")
    parser.add_argument("--data_dir",      type=str,   default="dataset")
    parser.add_argument("--teacher_path",  type=str,   default="models/teacher.pth")
    parser.add_argument("--student_path",  type=str,   default="models/student.pth")
    parser.add_argument("--epochs",        type=int,   default=70)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--image_size",    type=int,   default=84)
    # distillation weights
    parser.add_argument("--lambda_output", type=float, default=1.0,
                        help="Weight for output distillation loss (student vs teacher actions)")
    parser.add_argument("--lambda_feature",type=float, default=0.1,
                        help="Weight for feature distillation loss (student vs teacher bottleneck)")
    # curriculum (student only, matches train_offline_policy.py defaults)
    parser.add_argument("--curriculum_epochs",   type=int,   default=40)
    parser.add_argument("--fully_masked_epochs", type=int,   default=8)
    parser.add_argument("--lane_mask_prob",      type=float, default=0)
    # misc
    parser.add_argument("--pixel_noise_frac", type=float, default=0.0)
    parser.add_argument("--patience",         type=int,   default=8)
    parser.add_argument("--resume_from",      type=str,   default=None,
                        help="Resume student training from an existing checkpoint")
    args = parser.parse_args()

    if args.mode == "train_teacher":
        train_teacher(args)
    elif args.mode == "train_student":
        train_student(args)
