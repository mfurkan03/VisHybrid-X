"""
train_test_policy.py – train, fine-tune, and offline-test the DrivingPolicyNet.

Usage
-----
# Train from scratch (precomputed depth)
python src/train_test_policy.py --mode train \
    --pred_dir data/processed/dpt_pred \
    --model_path models/policy_model.pth

# Fine-tune a saved checkpoint
python src/train_test_policy.py --mode finetune \
    --finetune_from models/policy_model_best.pth \
    --model_path models/policy_finetuned.pth \
    --pred_dir data/processed/dpt_pred \
    --epochs 10 --lr 2e-5 \
    --freeze_backbone

# Offline test
python src/train_test_policy.py --mode test \
    --model_path models/policy_model_best.pth \
    --pred_dir data/processed/dpt_pred

# Simulation test (separate script)
# python src/test_simulation_policy.py --model_path models/policy_model_best.pth
"""

import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import DepthEstimationModel, build_policy
from policy.datasets import PrecomputedDepthDataset, MetaDriveRGBDataset
from policy.losses import custom_driving_loss, compute_offline_metrics
from policy.trainer import build_loaders, train_loop, extract_features_frozen
from utils.checkpoints import (
    load_checkpoint, freeze_backbone, print_trainable_params
)


# ============================================================
# 1. TRAIN FROM SCRATCH
# ============================================================
def train_policy(
    epochs:     int   = 20,
    batch_size: int   = 64,
    model_path: str   = "models/policy_model.pth",
    dpt_path:   str   = None,
    data_dir:   str   = "data/raw",
    lr:         float = 1e-4,
    pred_dir:   str   = None,
    curriculum_epochs: int = 10,
    fully_masked_epochs: int = 3,
    image_size: int = None,
    arch: str = "simple",
):
    print("--- Phase 2: Training Driving Policy (from scratch) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_precomputed = pred_dir is not None and os.path.isdir(os.path.join(pred_dir, "train"))
    depth_estimator = None if use_precomputed else DepthEstimationModel(finetuned_path=dpt_path)

    print(f"[INFO] {'Using PRECOMPUTED DPT from: ' + pred_dir if use_precomputed else 'Live DPT inference.'}")

    train_loader, val_loader = build_loaders(use_precomputed, pred_dir, data_dir, batch_size, depth_estimator)

    policy_model = build_policy(arch, image_size).to(device)
    optimizer    = optim.AdamW(policy_model.parameters(), lr=lr)
    scheduler    = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-fully_masked_epochs, eta_min=lr*10**-2) # 3 epochs less steps for scheduler

    train_loop(
        policy_model, device, train_loader, val_loader,
        optimizer, scheduler,
        epochs=epochs, start_epoch=0, best_val_loss=float("inf"),
        model_path=model_path, use_precomputed=use_precomputed,
        depth_estimator=depth_estimator, tag="Train",
        curriculum_epochs=curriculum_epochs,
        fully_masked_epochs=fully_masked_epochs,
        image_size=image_size,
    )


# ============================================================
# 2. FINE-TUNE
# ============================================================
def finetune_policy(
    finetune_from:   str,
    epochs:          int   = 10,
    batch_size:      int   = 32,
    model_path:      str   = "models/policy_finetuned.pth",
    dpt_path:        str   = None,
    data_dir:        str   = "data/raw",
    lr:              float = 2e-5,
    pred_dir:        str   = None,
    freeze_bb:       bool  = False,
    reset_optimizer: bool  = False,
    resume:          bool  = False,
    curriculum_epochs: int = 10,
    fully_masked_epochs: int = 3,
    image_size: int = None,
    arch: str = "simple",
):
    """
    Fine-tune (or resume) a previously saved policy model.

    freeze_bb       : freeze CNN/backbone layers; only train the head
    reset_optimizer : ignore saved optimizer/scheduler (good for new datasets)
    resume          : restore optimizer & scheduler to continue seamlessly
    """
    print("--- Fine-tuning Driving Policy ---")
    print(f"    Source : {finetune_from}  |  Output : {model_path}")
    print(f"    Epochs : {epochs}  |  LR : {lr}  |  Freeze backbone : {freeze_bb}")
    print(f"    Resume optimizer : {resume and not reset_optimizer}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_precomputed = pred_dir is not None and os.path.isdir(os.path.join(pred_dir, "train"))
    depth_estimator = None if use_precomputed else DepthEstimationModel(finetuned_path=dpt_path)

    train_loader, val_loader = build_loaders(use_precomputed, pred_dir, data_dir, batch_size, depth_estimator)

    policy_model = build_policy(arch, image_size).to(device)
    if freeze_bb:
        freeze_backbone(policy_model)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, policy_model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 1e-2)

    restore_opt              = resume and not reset_optimizer
    start_epoch, best_val   = load_checkpoint(
        finetune_from, policy_model,
        optimizer = optimizer if restore_opt else None,
        scheduler = scheduler if restore_opt else None,
        device    = device,
    )

    if not restore_opt:
        start_epoch = 0
        best_val    = float("inf")
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        print(f"[INFO] Optimizer reset. Fine-tuning from epoch 0, LR={lr}.")

    print_trainable_params(policy_model)

    train_loop(
        policy_model, device, train_loader, val_loader,
        optimizer, scheduler,
        epochs=epochs, start_epoch=start_epoch, best_val_loss=best_val,
        model_path=model_path, use_precomputed=use_precomputed,
        depth_estimator=depth_estimator, tag="Finetune",
        curriculum_epochs=curriculum_epochs,
        fully_masked_epochs=fully_masked_epochs,
        image_size=image_size,
    )


# ============================================================
# 3. TEST
# ============================================================
def test_policy(
    model_path: str,
    dpt_path:   str,
    data_dir:   str,
    pred_dir:   str = None,
    image_size: int = None,
    arch: str = "simple",
):
    print("--- Phase 3: Offline Testing Driving Policy ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_model = build_policy(arch, image_size).to(device)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict):
        key = "model" if "model" in ckpt else ("policy" if "policy" in ckpt else None)
        policy_model.load_state_dict(ckpt[key] if key else ckpt)
    else:
        policy_model.load_state_dict(ckpt)
    policy_model.eval()

    print("\n=> Offline Evaluation on Test Split...")

    use_precomputed = pred_dir is not None and os.path.isdir(os.path.join(pred_dir, "test"))

    if use_precomputed:
        from policy.trainer import apply_lane_mask
        test_ds = PrecomputedDepthDataset(pred_dir=pred_dir, split="test")
        def collate_fn(batch):
            depths, rgbs, actions, egos = zip(*batch)
            return (
                torch.tensor(np.stack(depths), dtype=torch.float32),
                np.stack(rgbs),
                np.stack(actions),
                np.stack(egos),
            )
    else:
        depth_estimator = DepthEstimationModel(finetuned_path=dpt_path)
        test_ds = MetaDriveRGBDataset(data_dir=data_dir, split="test")
        def collate_fn(batch):
            rgbs, actions, egos = zip(*batch)
            return np.stack(rgbs), np.stack(actions), np.stack(egos)

    if len(test_ds) == 0:
        print("[WARN] No test samples found.")
        return

    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loss   = 0.0
    test_pred, test_true = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if use_precomputed:
                depth_t, rgb_np, actions_np, ego_np = batch
                depth_t   = depth_t.to(device)
                actions_t = torch.tensor(actions_np, dtype=torch.float32, device=device)
                ego_t     = torch.tensor(ego_np,     dtype=torch.float32, device=device)
                combined  = apply_lane_mask(depth_t, rgb_np, device)
            else:
                rgb_np, actions_np, ego_np = batch
                actions_t   = torch.tensor(actions_np, dtype=torch.float32, device=device)
                combined, _ = extract_features_frozen(rgb_np, depth_estimator, device, image_size=image_size)
                ego_t       = torch.tensor(ego_np, dtype=torch.float32, device=device)

            pred       = policy_model(combined, ego_t)
            test_loss += custom_driving_loss(pred, actions_t).item()
            test_pred.append(pred.cpu().numpy())
            test_true.append(actions_np)

    test_pred_np  = np.concatenate(test_pred, axis=0)
    test_true_np  = np.concatenate(test_true, axis=0)
    test_metrics  = compute_offline_metrics(test_pred_np, test_true_np)
    avg_test_loss = test_loss / len(test_loader)

    print(f"\n=== OFFLINE SUMMARY ===")
    print(f"Test Loss:        {avg_test_loss:.4f}")
    print(f"Steer MSE:        {test_metrics['steering_mse']:.4f}")
    print(f"Accel MSE:        {test_metrics['accel_mse']:.4f}")
    print(f"Steer MAE:        {test_metrics['steering_mae']:.4f}")
    print(f"Accel MAE:        {test_metrics['accel_mae']:.4f}")
    print(f"Steer Dir Acc:    {test_metrics['steering_dir_acc']*100:.1f}%")
    print(f"Accel Dir Acc:    {test_metrics['direction_acc']*100:.1f}%")
    print(f"Braking Acc:      {test_metrics['brake_acc']*100:.1f}%")
    print(f"Steering Corr:    {test_metrics['steering_corr']:.4f}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       type=str, required=True,
                        choices=["train", "finetune", "test", "all"])
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--data_dir",   type=str,   default="dataset")
    parser.add_argument("--dpt_path",   type=str,   default="models/dpt_finetuned.pth")
    parser.add_argument("--model_path", type=str,   default="models/policy_model.pth")
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--pred_dir",   type=str,   default=None)
    # fine-tune args
    parser.add_argument("--finetune_from",  type=str, default=None)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--reset_optimizer", action="store_true")
    parser.add_argument("--resume",          action="store_true")
    parser.add_argument("--curriculum_epochs", type=int, default=10)
    parser.add_argument("--fully_masked_epochs", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--arch", type=str, default="simple", choices=["simple", "impala"])
    args = parser.parse_args()

    if args.mode in ("train", "all"):
        train_policy(args.epochs, 32, args.model_path,
                     args.dpt_path, args.data_dir, args.lr, pred_dir=args.pred_dir,
                     curriculum_epochs=args.curriculum_epochs,
                     fully_masked_epochs=args.fully_masked_epochs,
                     image_size=args.image_size,
                     arch=args.arch)

    if args.mode == "finetune":
        if args.finetune_from is None:
            parser.error("--finetune_from is required when --mode finetune")
        finetune_policy(
            finetune_from   = args.finetune_from,
            epochs          = args.epochs,
            batch_size      = 32,
            model_path      = args.model_path,
            dpt_path        = args.dpt_path,
            data_dir        = args.data_dir,
            lr              = args.lr if args.lr != 1e-4 else 1e-6,
            pred_dir        = args.pred_dir,
            freeze_bb       = args.freeze_backbone,
            reset_optimizer = args.reset_optimizer,
            resume          = args.resume,
            curriculum_epochs=args.curriculum_epochs,
            fully_masked_epochs=args.fully_masked_epochs,
            image_size=args.image_size,
            arch=args.arch,
        )

    if args.mode in ("test", "all"):
        test_policy(args.model_path, args.dpt_path, args.data_dir,
                    pred_dir=args.pred_dir, image_size=args.image_size, arch=args.arch)