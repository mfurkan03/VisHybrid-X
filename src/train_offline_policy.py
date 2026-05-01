"""
train_offline_policy.py – Train, fine-tune, and test the DrivingPolicyNet strictly offline.
No MetaDrive or CV2 imports, safe for Colab.
"""

import argparse
import os
import math

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import DepthEstimationModel, DrivingPolicyNet, DrivingPolicyNet2
from policy.datasets import PrecomputedDepthDataset, MetaDriveRGBDataset
from policy.losses import custom_driving_loss, compute_offline_metrics,compute_predictive_metrics
from policy.trainer import (
    build_loaders, train_loop, extract_features_frozen, 
    _collate_precomputed, _collate_rgb, apply_lane_mask
)
from utils.checkpoints import (
    load_checkpoint, freeze_backbone, print_trainable_params
)

def get_curriculum_lr_lambda(fully_masked_epochs, total_epochs):
    """
    Keeps LR at 1.0x for fully_masked_epochs, immediately drops by 10^-1,
    then applies cosine decay to 10^-2 over the remaining epochs.
    """
    def lr_lambda(epoch):
        if epoch < fully_masked_epochs:
            return 1.0
        else:
            base_factor = 0.1  # Immediate 10^-1 drop
            min_factor = 0.01  # Ends at 10^-2
            T_max = total_epochs - fully_masked_epochs
            if T_max <= 0:
                return base_factor
            current_step = epoch - fully_masked_epochs
            # Cosine decay from base_factor down to min_factor
            return min_factor + 0.5 * (base_factor - min_factor) * (1 + math.cos(math.pi * current_step / T_max))
    return lr_lambda


# ============================================================
# 1. TRAIN FROM SCRATCH
# ============================================================
def train_policy(
    epochs:     int   = 20,
    batch_size: int   = 32,
    model_path: str   = "models/policy_model.pth",
    dpt_path:   str   = None,
    data_dir:   str   = "data/raw",
    lr:         float = 1e-4,
    pred_dir:   str   = None,
    curriculum_epochs: int = 7,
    fully_masked_epochs: int = 3,
    image_size: int = None,
    policy:     str   = "standard",
):
    print("--- Phase 2: Training Driving Policy (from scratch) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")
    use_precomputed = pred_dir is not None and os.path.isdir(os.path.join(pred_dir, "train"))
    depth_estimator = None if use_precomputed else DepthEstimationModel(finetuned_path=dpt_path)

    print(f"[INFO] {'Using PRECOMPUTED DPT from: ' + pred_dir if use_precomputed else 'Live DPT inference.'}")

    train_loader, val_loader = build_loaders(use_precomputed, pred_dir, data_dir, batch_size)

    model_cls    = DrivingPolicyNet2 if policy == "deep" else DrivingPolicyNet
    policy_model = model_cls(image_size=image_size).to(device)
    print(f"[INFO] Policy network: {model_cls.__name__}")
    
    optimizer = optim.AdamW(policy_model.parameters(), lr=lr)
    
    # Applied Custom Scheduler
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=get_curriculum_lr_lambda(fully_masked_epochs, epochs)
    )

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
    policy:     str   = "standard",
):
    print("--- Fine-tuning Driving Policy ---")
    print(f"    Source : {finetune_from}  |  Output : {model_path}")
    print(f"    Epochs : {epochs}  |  LR : {lr}  |  Freeze backbone : {freeze_bb}")
    print(f"    Resume optimizer : {resume and not reset_optimizer}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_precomputed = pred_dir is not None and os.path.isdir(os.path.join(pred_dir, "train"))
    depth_estimator = None if use_precomputed else DepthEstimationModel(finetuned_path=dpt_path)

    train_loader, val_loader = build_loaders(use_precomputed, pred_dir, data_dir, batch_size)

    model_cls    = DrivingPolicyNet2 if policy == "deep" else DrivingPolicyNet
    policy_model = model_cls(image_size=image_size).to(device)
    print(f"[INFO] Policy network: {model_cls.__name__}")
    if freeze_bb:
        freeze_backbone(policy_model)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, policy_model.parameters()), lr=lr)
    
    # Applied Custom Scheduler
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=get_curriculum_lr_lambda(fully_masked_epochs, epochs)
    )

    restore_opt = resume and not reset_optimizer
    start_epoch, best_val = load_checkpoint(
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
# 3. TEST OFFLINE
# ============================================================
def test_offline_policy(
    model_path:   str,
    dpt_path:     str,
    data_dir:     str,
    pred_dir:     str  = None,
    image_size:   int  = None,
    policy:       str  = "standard",
):
    print("--- Phase 3: Offline Testing Driving Policy ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cls    = DrivingPolicyNet2 if policy == "deep" else DrivingPolicyNet
    policy_model = model_cls(image_size=image_size).to(device)
    print(f"[INFO] Policy network: {model_cls.__name__}")
        
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict):
        key = "model" if "model" in ckpt else ("policy" if "policy" in ckpt else None)
        policy_model.load_state_dict(ckpt[key] if key else ckpt)
    else:
        policy_model.load_state_dict(ckpt)
    policy_model.eval()

    depth_estimator = None

    print("\n=> Offline Evaluation on Test Split...")
    use_precomputed = pred_dir is not None and os.path.isdir(os.path.join(pred_dir, "test"))

    if use_precomputed:
        test_ds    = PrecomputedDepthDataset(pred_dir=pred_dir, split="test")
        collate_fn = _collate_precomputed
    else:
        depth_estimator = DepthEstimationModel(finetuned_path=dpt_path)
        test_ds    = MetaDriveRGBDataset(data_dir=data_dir, split="test")
        collate_fn = _collate_rgb

    if len(test_ds) > 0:
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

        test_pred_np = np.concatenate(test_pred, axis=0)
        test_true_np = np.concatenate(test_true, axis=0)
        test_metrics = compute_offline_metrics(test_pred_np, test_true_np)
        predictive_metrics = compute_predictive_metrics(test_pred_np, test_true_np) # NEW
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
        print(f"\n=== SIMULATION PREDICTION METRICS ===")
        print(f"95th Pctl Error:  {predictive_metrics['steer_95th_pctl_err']:.4f}")
        print(f"Active Turn MAE:  {predictive_metrics['active_turn_mae']:.4f}")
        print(f"Jitter Ratio:     {predictive_metrics['jitter_ratio']:.2f}x")
        print(f"Out of Bounds %:  {predictive_metrics['out_of_bounds_rate']*100:.2f}%")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       type=str, required=True, choices=["train", "finetune", "test"])
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
    parser.add_argument("--curriculum_epochs", type=int, default=20)
    parser.add_argument("--fully_masked_epochs", type=int, default=3)
    parser.add_argument("--image_size",   type=int, default=84)
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--policy",       type=str, default="standard", choices=["standard", "deep"])
    args = parser.parse_args()

    if args.mode == "train":
        train_policy(args.epochs, args.batch_size, args.model_path,
                     args.dpt_path, args.data_dir, args.lr, pred_dir=args.pred_dir,
                     curriculum_epochs=args.curriculum_epochs,
                     fully_masked_epochs=args.fully_masked_epochs,
                     image_size=args.image_size,
                     policy=args.policy)

    elif args.mode == "finetune":
        if args.finetune_from is None:
            parser.error("--finetune_from is required when --mode finetune")
        finetune_policy(
            finetune_from   = args.finetune_from,
            epochs          = args.epochs,
            batch_size      = args.batch_size,
            model_path      = args.model_path,
            dpt_path        = args.dpt_path,
            data_dir        = args.data_dir,
            lr              = args.lr,
            pred_dir        = args.pred_dir,
            freeze_bb       = args.freeze_backbone,
            reset_optimizer = args.reset_optimizer,
            resume          = args.resume,
            curriculum_epochs=args.curriculum_epochs,
            fully_masked_epochs=args.fully_masked_epochs,
            image_size=args.image_size,
            policy=args.policy,
        )

    elif args.mode == "test":
        test_offline_policy(args.model_path, args.dpt_path, args.data_dir,
                            pred_dir=args.pred_dir, image_size=args.image_size, policy=args.policy)