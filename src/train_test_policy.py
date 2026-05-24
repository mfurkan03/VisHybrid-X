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
import math
import os

BENCHMARK_SEEDS = [0, 1, 2, 3, 4]


def _seeded_path(model_path: str, seed: int) -> str:
    base, ext = os.path.splitext(model_path)
    return f"{base}_seed_{seed}{ext}"

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import DepthEstimationModel, build_policy


def _curriculum_lr_lambda(fully_masked_epochs: int, total_epochs: int, eta_min_ratio: float = 5e-2):
    """LR schedule: hold at 1.0 during fully-masked phase, drop ×0.1 at curriculum start, cosine anneal after."""
    drop = 0.1
    T = max(total_epochs - fully_masked_epochs, 1)
    def lr_lambda(epoch: int) -> float:
        if epoch < fully_masked_epochs:
            return 1.0
        t = epoch - fully_masked_epochs
        cosine = eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1 + math.cos(math.pi * t / T))
        return drop * cosine
    return lr_lambda

from policy.datasets import PrecomputedDepthDataset, MetaDriveRGBDataset
from policy.losses import (custom_driving_loss_beta, compute_offline_metrics,
                           compute_predictive_metrics, compute_heading_metrics)
from policy.trainer import build_loaders, train_loop, extract_features_frozen
from utils.checkpoints import (
    load_checkpoint, freeze_backbone, print_trainable_params
)
from utils.seed import seed_everything


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
    always_lane_masked: bool = False,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    seed: int = 42,
    prob_pixel_noise: float = 1.0,
    prob_hflip: float = 0.5,
    prob_grayscale: float = 0.1,
):
    print("--- Phase 2: Training Driving Policy (from scratch) ---")
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_precomputed = pred_dir is not None and os.path.isdir(os.path.join(pred_dir, "train"))
    depth_estimator = None if use_precomputed else DepthEstimationModel(finetuned_path=dpt_path)

    print(f"[INFO] {'Using PRECOMPUTED DPT from: ' + pred_dir if use_precomputed else 'Live DPT inference.'}")
    train_loader, val_loader = build_loaders(use_precomputed, pred_dir, data_dir, batch_size, depth_estimator, seed=seed)

    policy_model = build_policy(arch, image_size).to(device)
    optimizer    = optim.AdamW(policy_model.parameters(), lr=lr)
    scheduler    = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_curriculum_lr_lambda(fully_masked_epochs, epochs))

    train_loop(
        policy_model, device, train_loader, val_loader,
        optimizer, scheduler,
        epochs=epochs, start_epoch=0, best_val_loss=float("inf"),
        model_path=model_path, use_precomputed=use_precomputed,
        depth_estimator=depth_estimator, tag="Train",
        curriculum_epochs=curriculum_epochs,
        fully_masked_epochs=fully_masked_epochs,
        image_size=image_size,
        always_lane_masked=always_lane_masked,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        prob_pixel_noise=prob_pixel_noise,
        prob_hflip=prob_hflip,
        prob_grayscale=prob_grayscale,
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
    always_lane_masked: bool = False,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    seed: int = 42,
    prob_pixel_noise: float = 1.0,
    prob_hflip: float = 0.5,
    prob_grayscale: float = 0.1,
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

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_precomputed = pred_dir is not None and os.path.isdir(os.path.join(pred_dir, "train"))
    depth_estimator = None if use_precomputed else DepthEstimationModel(finetuned_path=dpt_path)

    train_loader, val_loader = build_loaders(use_precomputed, pred_dir, data_dir, batch_size, depth_estimator, seed=seed)

    policy_model = build_policy(arch, image_size).to(device)
    if freeze_bb:
        freeze_backbone(policy_model)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, policy_model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_curriculum_lr_lambda(fully_masked_epochs, epochs))

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
        always_lane_masked=always_lane_masked,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        prob_pixel_noise=prob_pixel_noise,
        prob_hflip=prob_hflip,
        prob_grayscale=prob_grayscale,
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
    always_lane_masked: bool = False,
    seed: int = 42,
):
    print("--- Phase 3: Offline Testing Driving Policy ---")
    seed_everything(seed)
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
            depths, rgbs, actions, egos, ego_fulls = zip(*batch)
            return (
                torch.tensor(np.stack(depths), dtype=torch.float32),
                np.stack(rgbs),
                np.stack(actions),
                np.stack(egos),
                np.stack(ego_fulls),
            )
    else:
        depth_estimator = DepthEstimationModel(finetuned_path=dpt_path)
        test_ds = MetaDriveRGBDataset(data_dir=data_dir, split="test")
        def collate_fn(batch):
            rgbs, actions, egos = zip(*batch)
            n = len(actions)
            return (np.stack(rgbs), np.stack(actions), np.stack(egos),
                    np.zeros((n, 5), dtype=np.float32))

    if len(test_ds) == 0:
        print("[WARN] No test samples found.")
        return

    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loss   = 0.0
    test_pred, test_true, test_ego, test_ego_full = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if use_precomputed:
                depth_t, rgb_np, actions_np, ego_np, ego_full_np = batch
                depth_t   = depth_t.to(device)
                actions_t = torch.tensor(actions_np, dtype=torch.float32, device=device)
                ego_t     = torch.tensor(ego_np,     dtype=torch.float32, device=device)
                combined  = apply_lane_mask(depth_t, rgb_np, device, image_size=image_size,
                                            always_lane_masked=always_lane_masked)
            else:
                rgb_np, actions_np, ego_np, ego_full_np = batch
                actions_t   = torch.tensor(actions_np, dtype=torch.float32, device=device)
                combined, _ = extract_features_frozen(rgb_np, depth_estimator, device, image_size=image_size,
                                                      always_lane_masked=always_lane_masked)
                ego_t       = torch.tensor(ego_np, dtype=torch.float32, device=device)

            pred_alpha, pred_beta = policy_model(combined, ego_t)
            actions_01 = (actions_t.clamp(-1.0, 1.0) + 1.0) / 2.0
            test_loss += custom_driving_loss_beta(pred_alpha, pred_beta, actions_01).item()
            mean_01    = pred_alpha / (pred_alpha + pred_beta)
            pred_mean  = (mean_01 * 2.0 - 1.0).cpu().numpy()
            test_pred.append(pred_mean)
            test_true.append(actions_np)
            test_ego.append(ego_np)
            test_ego_full.append(ego_full_np)

    test_pred_np     = np.concatenate(test_pred,     axis=0)
    test_true_np     = np.concatenate(test_true,     axis=0)
    test_ego_np      = np.concatenate(test_ego,      axis=0)
    test_ego_full_np = np.concatenate(test_ego_full, axis=0)

    test_m   = compute_offline_metrics(test_pred_np, test_true_np)
    test_pm  = compute_predictive_metrics(test_pred_np, test_true_np, test_ego_np)
    test_hm  = compute_heading_metrics(test_pred_np, test_true_np, test_ego_full_np)
    avg_test_loss = test_loss / len(test_loader)

    print(f"\n=== OFFLINE SUMMARY ===")
    print(f"Test Loss:              {avg_test_loss:.4f}")
    print(f"Steer MSE:              {test_m['steering_mse']:.4f}")
    print(f"Accel MSE:              {test_m['accel_mse']:.4f}")
    print(f"Steer MAE:              {test_m['steering_mae']:.4f}")
    print(f"Accel MAE:              {test_m['accel_mae']:.4f}")
    print(f"Steer Dir Acc:          {test_m['steering_dir_acc']*100:.1f}%")
    print(f"Accel Dir Acc:          {test_m['direction_acc']*100:.1f}%")
    print(f"Braking Acc:            {test_m['brake_acc']*100:.1f}%")
    print(f"Steering Corr:          {test_m['steering_corr']:.4f}")
    print(f"\n--- Predictive Metrics ---")
    print(f"P95 Steer Error:        {test_pm['steer_p95_error']:.4f}")
    print(f"Active Turn MAE:        {test_pm['active_turn_mae']:.4f}")
    print(f"Critical Turn MAE:      {test_pm['critical_turn_mae']:.4f}")
    print(f"Jitter Ratio:           {test_pm['jitter_ratio']:.3f}  (1.0=expert)")
    print(f"Out-of-Bounds Rate:     {test_pm['out_of_bounds_rate']*100:.1f}%")
    print(f"Speed-Weighted MAE:     {test_pm['speed_weighted_steer_mae']:.4f}")
    print(f"Pre-Brake Anticipation: {test_pm['pre_brake_anticipation']*100:.1f}%")
    if test_hm:
        print(f"\n--- Heading Metrics ---")
        print(f"Heading Dir Acc:        {test_hm['heading_dir_acc']*100:.1f}%")
        print(f"Heading Delta MAE:      {test_hm['heading_delta_mae']:.4f}")
        print(f"Window Div Mean:        {test_hm['window_heading_div_mean']:.4f}")
        print(f"Window Div P95:         {test_hm['window_heading_div_p95']:.4f}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       type=str, required=True,
                        choices=["train", "finetune", "test", "all"])
    parser.add_argument("--epochs",     type=int,   default=70)
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
    parser.add_argument("--curriculum_epochs", type=int, default=30)
    parser.add_argument("--fully_masked_epochs", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--arch", type=str, default="simple", choices=["simple", "impala", "impala_v2"])
    parser.add_argument("--always_lane_masked", action="store_true",
                        help="Force alpha=0 (fully lane-masked) for every batch, skipping curriculum")
    parser.add_argument("--early_stopping_patience", type=int, default=8,
                        help="Stop if val loss does not improve for this many epochs (0=disabled)")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0,
                        help="Minimum improvement in val loss to count as progress")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed for reproducibility")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run training over 5 seeds (0-4); checkpoints saved with _seed_x suffix")
    parser.add_argument("--aug_pixel_noise", type=float, default=1.0,
                        help="Probability [0-1] of adding Gaussian noise to 10%% of RGB pixels per sample (0=off)")
    parser.add_argument("--aug_hflip", type=float, default=0.5,
                        help="Probability [0-1] of horizontal flip per sample; negates steer and heading_delta (0=off)")
    parser.add_argument("--aug_grayscale", type=float, default=0.1,
                        help="Probability [0-1] of converting RGB to grayscale per sample (0=off)")
    args = parser.parse_args()

    seeds = BENCHMARK_SEEDS if args.benchmark else [args.seed]
    if args.benchmark:
        print(f"[BENCHMARK] Running {len(seeds)} seeds: {seeds}")

    if args.mode in ("train", "all"):
        for seed in seeds:
            mp = _seeded_path(args.model_path, seed) if args.benchmark else args.model_path
            if args.benchmark:
                print(f"\n[BENCHMARK] === Seed {seed} — checkpoint: {mp} ===")
            train_policy(args.epochs, 32, mp,
                         args.dpt_path, args.data_dir, args.lr, pred_dir=args.pred_dir,
                         curriculum_epochs=args.curriculum_epochs,
                         fully_masked_epochs=args.fully_masked_epochs,
                         image_size=args.image_size,
                         arch=args.arch,
                         always_lane_masked=args.always_lane_masked,
                         early_stopping_patience=args.early_stopping_patience,
                         early_stopping_min_delta=args.early_stopping_min_delta,
                         seed=seed,
                         prob_pixel_noise=args.aug_pixel_noise,
                         prob_hflip=args.aug_hflip,
                         prob_grayscale=args.aug_grayscale)

    if args.mode == "finetune":
        if args.finetune_from is None:
            parser.error("--finetune_from is required when --mode finetune")
        for seed in seeds:
            mp = _seeded_path(args.model_path, seed) if args.benchmark else args.model_path
            if args.benchmark:
                print(f"\n[BENCHMARK] === Seed {seed} — checkpoint: {mp} ===")
            finetune_policy(
                finetune_from   = args.finetune_from,
                epochs          = args.epochs,
                batch_size      = 32,
                model_path      = mp,
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
                always_lane_masked=args.always_lane_masked,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                seed=seed,
                prob_pixel_noise=args.aug_pixel_noise,
                prob_hflip=args.aug_hflip,
                prob_grayscale=args.aug_grayscale,
            )

    if args.mode in ("test", "all"):
        test_policy(args.model_path, args.dpt_path, args.data_dir,
                    pred_dir=args.pred_dir,
                    image_size=args.image_size, arch=args.arch,
                    always_lane_masked=args.always_lane_masked, seed=args.seed)