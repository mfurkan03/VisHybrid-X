""""
python src/train_dpt.py --mode train --epochs 5 --data_dir dataset --model_path models/dpt_finetuned.pth

# Step 2 – precompute & cache predictions ONCE
python src/train_dpt.py --mode precompute \
    --model_path models/dpt_finetuned.pth \
    --data_dir dataset \
    --out_dir data/processed/dpt_pred

# Step 3 – train policy at full speed (no DPT inference per step)
python src/train_test_policy.py --mode train \
    --pred_dir data/processed/dpt_pred \
    --model_path models/policy_model.pth

# Step 4 – fine-tune a saved policy model (resume / transfer)
python src/train_test_policy.py --mode finetune \
    --finetune_from models/policy_model_best.pth \
    --model_path models/policy_finetuned.pth \
    --pred_dir data/processed/dpt_pred \
    --epochs 10 \
    --lr 2e-5 \
    --freeze_backbone          # optional: freeze CNN layers, train only head
"""

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera

import cv2
import time
import sys
from pathlib import Path
from tqdm import tqdm
import math

from models import DepthEstimationModel, DrivingPolicyNet, extract_ego_state, EGO_DIM


# ==========================================
# 1. HELPERS
# ==========================================
def get_lane_mask_visual(rgb_image, threshold_value=180):
    if rgb_image.max() <= 1.0:
        img_uint8 = (rgb_image * 255.0).astype(np.uint8)
    else:
        img_uint8 = rgb_image.astype(np.uint8)
    h, w = img_uint8.shape[:2]
    roi_img = img_uint8.copy()
    roi_img[0:int(h * 0.55), :] = 0
    gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return mask


def extract_features_frozen(rgb_batch, depth_estimator, device):
    rescaled_img = cv2.resize(rgb_batch[0], (196, 196), interpolation=cv2.INTER_LINEAR)
    new_batch = np.expand_dims(rescaled_img, axis=0)
    with torch.no_grad():
        depth_tensors = depth_estimator.predict_batch_with_grad(new_batch)

    for i in range(depth_tensors.shape[0]):
        d = depth_tensors[i, 0]
        d_min, d_max = d.min(), d.max()
        depth_tensors[i, 0] = 1.0 - (d - d_min) / (d_max - d_min + 1e-6)

    lane_list = []
    for rgb in rgb_batch:
        mask = get_lane_mask_visual(rgb)
        lane_norm = (cv2.resize(mask, (84, 84)) / 255.0).astype(np.float32)
        lane_list.append(lane_norm)
    lane_tensor = torch.tensor(np.stack(lane_list), device=device).unsqueeze(1)

    combined = torch.cat([depth_tensors, lane_tensor], dim=1)
    ego_zeros = torch.zeros(combined.shape[0], EGO_DIM, device=device)
    return combined, ego_zeros


# ==========================================
# 2. CHECKPOINT UTILS
# ==========================================
def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path, extra=None):
    """Save a full training checkpoint (model + optimizer + scheduler state)."""
    payload = {
        "epoch":      epoch,
        "val_loss":   val_loss,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Load checkpoint into model (and optionally optimizer / scheduler).

    Handles three checkpoint formats:
      1. Full checkpoint dict with 'model' key  (saved by save_checkpoint)
      2. Legacy format with 'policy' key
      3. Raw state-dict (weights only)

    Returns:
        start_epoch (int), best_val_loss (float)
    """
    ckpt = torch.load(path, map_location=device)

    # ── Determine weights ──────────────────────────────────────────────────────
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            if optimizer  is not None and "optimizer"  in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler  is not None and "scheduler"  in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch    = ckpt.get("epoch",    0) + 1
            best_val_loss  = ckpt.get("val_loss", float("inf"))
            print(f"[INFO] Resumed full checkpoint from epoch {ckpt.get('epoch', '?')} "
                  f"(val_loss={best_val_loss:.4f})")
        elif "policy" in ckpt:
            # legacy format written by older versions of this script
            model.load_state_dict(ckpt["policy"])
            start_epoch   = 0
            best_val_loss = float("inf")
            print("[INFO] Loaded legacy 'policy' checkpoint (weights only).")
        else:
            # assume it IS the state dict
            model.load_state_dict(ckpt)
            start_epoch   = 0
            best_val_loss = float("inf")
            print("[INFO] Loaded raw state-dict checkpoint.")
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    return start_epoch, best_val_loss


def freeze_backbone(model: nn.Module):
    """
    Freeze every parameter whose name starts with 'backbone' or 'cnn'.
    Adjust the prefix to match the actual attribute names in DrivingPolicyNet.
    """
    frozen = 0
    for name, param in model.named_parameters():
        if name.startswith(("backbone", "cnn", "encoder", "feature")):
            param.requires_grad = False
            frozen += 1
    if frozen:
        print(f"[INFO] Frozen {frozen} backbone parameter tensors.")
    else:
        print("[WARNING] --freeze_backbone was set but no parameters matched "
              "prefixes ('backbone', 'cnn', 'encoder', 'feature'). "
              "All parameters will be trained.")


def print_trainable_params(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")


# ==========================================
# 3. DATASETS
# ==========================================
class MetaDriveRGBDataset(Dataset):
    """
    Loads raw RGB frames + ego_state from .npz files.
    Falls back gracefully if ego_state is missing (older data without it).
    """

    def __init__(self, data_dir, split="train"):
        split_dir  = os.path.join(data_dir, split)
        self.files = glob.glob(os.path.join(split_dir, "*.npz"))
        self.rgb_frames = []
        self.actions    = []
        self.ego_states = []

        for f in self.files:
            try:
                data = np.load(f, allow_pickle=True)
            except Exception:
                continue

            rgb_keys = [k for k in data.files if k.endswith('_rgb')]
            if not rgb_keys or 'action' not in data.files:
                continue

            n = len(data['action'])

            if 'ego_state' in data.files:
                ego = data['ego_state']
            else:
                ego = np.zeros((n, EGO_DIM), dtype=np.float32)

            self.rgb_frames.extend(data[rgb_keys[0]])
            self.actions.extend(data['action'])
            self.ego_states.extend(ego)

        print(f"[INFO] PolicyDataset-RGB ({split}): {len(self.actions)} samples from {split_dir}.")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (
            self.rgb_frames[idx],
            np.array(self.actions[idx],    dtype=np.float32),
            np.array(self.ego_states[idx], dtype=np.float32),
        )


class PrecomputedDepthDataset(Dataset):
    """
    Fast dataset using pre-cached DPT depth + lane + ego_state.

    Expected file layout:
        <pred_dir>/<split>/episode_N.npz
            depth_pred : float32  (N, 1, 84, 84)
            lane_mask  : float32  (N, 1, 84, 84)
            action     : float32  (N, 2)
            ego_state  : float32  (N, EGO_DIM)   ← new; zeros if missing
    """

    def __init__(self, pred_dir: str, split: str = "train"):
        split_dir  = os.path.join(pred_dir, split)
        self.files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))

        self.depth_frames: list = []
        self.lane_frames:  list = []
        self.actions:      list = []
        self.ego_states:   list = []

        for f in self.files:
            try:
                data = np.load(f, allow_pickle=True)
            except Exception as e:
                print(f"[WARNING] Could not load {f}: {e}")
                continue

            if 'depth_pred' not in data.files or 'lane_mask' not in data.files or 'action' not in data.files:
                print(f"[WARNING] Missing keys in {f}, skipping.")
                continue

            depths  = data['depth_pred']
            lanes   = data['lane_mask']
            actions = data['action']
            n       = min(len(depths), len(lanes), len(actions))

            if 'ego_state' in data.files:
                ego = data['ego_state'][:n]
            else:
                ego = np.zeros((n, EGO_DIM), dtype=np.float32)

            self.depth_frames.extend(depths[:n])
            self.lane_frames.extend(lanes[:n])
            self.actions.extend(actions[:n])
            self.ego_states.extend(ego)

        print(
            f"[INFO] PrecomputedDepthDataset ({split}): "
            f"{len(self.actions)} samples from {split_dir}."
        )

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        depth    = self.depth_frames[idx]
        lane     = self.lane_frames[idx]
        combined = np.concatenate([depth, lane], axis=0)
        action   = np.array(self.actions[idx],    dtype=np.float32)
        ego      = np.array(self.ego_states[idx], dtype=np.float32)
        return combined, action, ego


# ==========================================
# 4. LOSS & METRICS
# ==========================================
def custom_driving_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse            = (pred - target) ** 2
    brake_mask     = (target[:, 1] < 0.0).float()
    brake_mult     = 3.0
    penalty_weight = 1.0 + brake_mask * brake_mult
    mse_weighted = mse.clone()
    mse_weighted[:, 1] = mse[:, 1] * penalty_weight
    return mse_weighted.mean()


def compute_offline_metrics(pred_actions, true_actions):
    steering_mse  = float(np.mean((pred_actions[:, 0] - true_actions[:, 0]) ** 2))
    accel_mse     = float(np.mean((pred_actions[:, 1] - true_actions[:, 1]) ** 2))
    corr, _       = pearsonr(pred_actions[:, 0], true_actions[:, 0])
    direction_acc = float(np.mean(np.sign(pred_actions[:, 1]) == np.sign(true_actions[:, 1])))
    return {"steering_mse": steering_mse, "accel_mse": accel_mse,
            "steering_corr": float(corr), "direction_acc": direction_acc}


# ==========================================
# 5. CORE TRAIN LOOP  (shared by train & finetune)
# ==========================================
def _build_loaders(use_precomputed, pred_dir, data_dir, batch_size, depth_estimator):
    """Return (train_loader, val_loader, use_precomputed)."""

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


def _run_epoch(policy_model, loader, optimizer, device,
               use_precomputed, depth_estimator, is_train, desc):
    """Run one training or validation epoch. Returns (avg_loss, preds, trues)."""
    policy_model.train() if is_train else policy_model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []

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


def _train_loop(
    policy_model,
    device,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    epochs:         int,
    start_epoch:    int,
    best_val_loss:  float,
    model_path:     str,
    use_precomputed: bool,
    depth_estimator,
    tag:            str = "Train",
):
    """
    Shared epoch loop used by both `train_policy` and `finetune_policy`.
    Saves best checkpoint and last checkpoint after every epoch.
    """
    file_root, file_ext = os.path.splitext(model_path)
    best_path = f"{file_root}_best{file_ext}"

    for epoch in range(start_epoch, start_epoch + epochs):
        # ── Training ──────────────────────────────────────────────────────────
        avg_train, tr_pred, tr_true = _run_epoch(
            policy_model, train_loader, optimizer, device,
            use_precomputed, depth_estimator, is_train=True,
            desc=f"[{tag}] Epoch {epoch+1}/{start_epoch+epochs} [Train]",
        )

        # ── Validation ────────────────────────────────────────────────────────
        avg_val, val_pred, val_true = _run_epoch(
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

        # ── Checkpointing ─────────────────────────────────────────────────────
        save_checkpoint(policy_model, optimizer, scheduler, epoch, avg_val, model_path)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_checkpoint(policy_model, optimizer, scheduler, epoch, avg_val, best_path)
            print(f"*** Best model saved → {best_path}  (Val Loss: {best_val_loss:.4f}) ***")

    return best_val_loss


# ==========================================
# 6. TRAIN POLICY  (fresh training)
# ==========================================
def train_policy(
    epochs:     int   = 20,
    batch_size: int   = 64,
    model_path: str   = "policy_model.pth",
    dpt_path:   str   = None,
    data_dir:   str   = "data/raw",
    lr:         float = 1e-4,
    pred_dir:   str   = None,
):
    print("--- Phase 2: Training Driving Policy (from scratch) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_precomputed = (
        pred_dir is not None
        and os.path.isdir(os.path.join(pred_dir, "train"))
    )
    depth_estimator = None if use_precomputed else DepthEstimationModel(finetuned_path=dpt_path)

    if use_precomputed:
        print(f"[INFO] Using PRECOMPUTED DPT predictions from: {pred_dir}")
    else:
        print("[INFO] No precomputed predictions – using live DPT inference.")

    train_loader, val_loader = _build_loaders(
        use_precomputed, pred_dir, data_dir, batch_size, depth_estimator
    )

    policy_model = DrivingPolicyNet().to(device)
    optimizer    = optim.AdamW(policy_model.parameters(), lr=lr)
    scheduler    = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    _train_loop(
        policy_model, device, train_loader, val_loader,
        optimizer, scheduler,
        epochs=epochs, start_epoch=0, best_val_loss=float("inf"),
        model_path=model_path, use_precomputed=use_precomputed,
        depth_estimator=depth_estimator, tag="Train",
    )


# ==========================================
# 7. FINE-TUNE POLICY  (resume / transfer)
# ==========================================
def finetune_policy(
    finetune_from:   str,
    epochs:          int   = 10,
    batch_size:      int   = 32,
    model_path:      str   = "models/policy_finetuned.pth",
    dpt_path:        str   = None,
    data_dir:        str   = "data/raw",
    lr:              float = 2e-5,
    pred_dir:        str   = None,
    freeze_bb: bool  = False,
    reset_optimizer: bool  = False,
    resume:          bool  = False,
):
    """
    Fine-tune (or resume) a previously saved policy model.

    Args:
        finetune_from:   Path to the checkpoint to load weights from.
        epochs:          Additional epochs to train.
        lr:              Learning rate (default lower than training).
        freeze_bb: If True, freeze CNN/backbone layers; only head is trained.
        reset_optimizer: If True, ignore the saved optimizer/scheduler state and
                         start fresh (useful for transfer to a new dataset).
        resume:          If True, also restore optimizer & scheduler state so
                         training continues seamlessly from where it left off.
    """
    print("--- Fine-tuning Driving Policy ---")
    print(f"    Source checkpoint : {finetune_from}")
    print(f"    Output checkpoint : {model_path}")
    print(f"    Epochs            : {epochs}")
    print(f"    LR                : {lr}")
    print(f"    Freeze backbone   : {freeze_bb}")
    print(f"    Resume optimizer  : {resume and not reset_optimizer}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_precomputed = (
        pred_dir is not None
        and os.path.isdir(os.path.join(pred_dir, "train"))
    )
    depth_estimator = None if use_precomputed else DepthEstimationModel(finetuned_path=dpt_path)

    train_loader, val_loader = _build_loaders(
        use_precomputed, pred_dir, data_dir, batch_size, depth_estimator
    )

    # ── Build model ───────────────────────────────────────────────────────────
    policy_model = DrivingPolicyNet().to(device)

    if freeze_bb:
        freeze_backbone(policy_model)

    # ── Build optimizer & scheduler BEFORE loading so we can optionally restore
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, policy_model.parameters()), lr=lr
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 1e-2
    )

    # ── Load checkpoint ───────────────────────────────────────────────────────
    restore_opt = resume and not reset_optimizer
    start_epoch, best_val_loss = load_checkpoint(
        finetune_from, policy_model,
        optimizer  = optimizer  if restore_opt else None,
        scheduler  = scheduler  if restore_opt else None,
        device     = device,
    )

    # When NOT resuming we always restart from epoch 0
    if not restore_opt:
        start_epoch   = 0
        best_val_loss = float("inf")
        # Override LR in case checkpoint had a different one
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        print(f"[INFO] Optimizer reset. Starting fine-tune from epoch 0 with LR={lr}.")

    print_trainable_params(policy_model)

    _train_loop(
        policy_model, device, train_loader, val_loader,
        optimizer, scheduler,
        epochs=epochs, start_epoch=start_epoch, best_val_loss=best_val_loss,
        model_path=model_path, use_precomputed=use_precomputed,
        depth_estimator=depth_estimator, tag="Finetune",
    )


# ==========================================
# 8. TESTING LOOP
# ==========================================
def test_policy(model_path, dpt_path, data_dir, num_episodes,
                pred_dir=None, test_mode="all"):
    print("--- Phase 3: Testing Driving Policy ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_model = DrivingPolicyNet().to(device)
    checkpoint   = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        key = "model" if "model" in checkpoint else ("policy" if "policy" in checkpoint else None)
        if key:
            policy_model.load_state_dict(checkpoint[key])
        else:
            policy_model.load_state_dict(checkpoint)
    else:
        policy_model.load_state_dict(checkpoint)
    policy_model.eval()

    depth_estimator = None

    # ---- Offline test ----
    if test_mode in ("offline", "all"):
        print("\n=> Running Offline Evaluation on Test Split...")

        use_precomputed = (
            pred_dir is not None
            and os.path.isdir(os.path.join(pred_dir, "test"))
        )

        if use_precomputed:
            test_ds = PrecomputedDepthDataset(pred_dir=pred_dir, split="test")

            def collate_fn(batch):
                combined, actions, egos = zip(*batch)
                return (
                    torch.tensor(np.stack(combined), dtype=torch.float32),
                    np.stack(actions),
                    np.stack(egos),
                )
        else:
            depth_estimator = DepthEstimationModel(finetuned_path=dpt_path)
            test_ds = MetaDriveRGBDataset(data_dir=data_dir, split="test")

            def collate_fn(batch):
                rgbs, actions, egos = zip(*batch)
                return np.stack(rgbs), np.stack(actions), np.stack(egos)

        if len(test_ds) > 0:
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
            test_loss   = 0.0
            test_pred, test_true = [], []

            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Testing"):
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

                    pred = policy_model(combined, ego_t)
                    test_loss += custom_driving_loss(pred, actions_t).item()
                    test_pred.append(pred.cpu().numpy())
                    test_true.append(actions_np)

            avg_test = test_loss / len(test_loader)
            test_m   = compute_offline_metrics(np.concatenate(test_pred), np.concatenate(test_true))

            print("\n=== OFFLINE TEST RESULTS ===")
            print(f"Test Loss    : {avg_test:.4f}")
            print(f"Steering MSE : {test_m['steering_mse']:.4f}")
            print(f"Accel MSE    : {test_m['accel_mse']:.4f}")
            print(f"Direction Acc: {test_m['direction_acc']:.3f}")
        else:
            print("[WARNING] No test data found. Skipping offline evaluation.")

    # ---- Online simulation test ----
    if test_mode in ("simulation", "all"):
        print("\n=> Running Online Evaluation (Simulation)...")

        if depth_estimator is None:
            depth_estimator = DepthEstimationModel(finetuned_path=dpt_path)

        config = {
            "use_render":        True,
            "image_observation": True,
            "sensors":           {"rgb": (RGBCamera, 200, 200)},
            "vehicle_config":    {"image_source": "rgb"},
            "show_interface":    False,
            "image_on_cuda":     False,
            "start_seed":        316181,
        }
        env = MetaDriveEnv(config)
        success_flags, route_completions = [], []

        for ep in range(num_episodes):
            obs, info = env.reset()
            done       = False
            step_count = 0
            anlik_fps  = 0.0
            last_time  = time.time()
            last_steer = 0.0

            while not done:
                step_count += 1
                rgb_img = env.engine.get_sensor("rgb").perceive(env.agent)

                combined_tensor, _ = extract_features_frozen(
                    rgb_img[np.newaxis], depth_estimator, device
                )

                ego_reading = extract_ego_state(env.agent, last_steer=last_steer)
                ego_t = torch.tensor(
                    ego_reading.ego_model, dtype=torch.float32, device=device
                ).unsqueeze(0)

                with torch.no_grad():
                    pred_action = policy_model(combined_tensor, ego_t).cpu().numpy()[0]

                last_steer = float(pred_action[0])

                if step_count % 1 == 0:
                    depth_uint8   = (combined_tensor[0, 0].cpu().numpy() * 255).astype(np.uint8)
                    lane_uint8    = (combined_tensor[0, 1].cpu().numpy() * 255).astype(np.uint8)
                    depth_color   = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
                    lane_color    = cv2.cvtColor(lane_uint8, cv2.COLOR_GRAY2BGR)
                    vis_size      = (400, 400)
                    depth_resized = cv2.resize(depth_color, vis_size, interpolation=cv2.INTER_LINEAR)
                    lane_resized  = cv2.resize(lane_color,  vis_size, interpolation=cv2.INTER_NEAREST)

                    hud = np.zeros((40, 800, 3), dtype=np.uint8)
                    cv2.putText(
                        hud,
                        f"spd:{ego_reading.total_speed:+.2f}  fwd:{ego_reading.forward_speed:+.2f}  "
                        f"lat:{ego_reading.lateral_speed:+.2f}  hdg:{ego_reading.heading_delta:+.2f}  "
                        f"str:{ego_reading.last_steer:+.2f}  "
                        f"->  steer:{pred_action[0]:+.2f}  throt:{pred_action[1]:+.2f}",
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1,
                    )
                    dashboard = np.hstack((depth_resized, lane_resized))
                    dashboard = np.vstack((hud, dashboard))
                    cv2.imshow("Depth | Lane  (with Ego HUD)", dashboard)
                    cv2.waitKey(1)

                for _ in range(1):
                    obs, reward, terminated, truncated, info = env.step(pred_action)
                    done = terminated or truncated
                    if done:
                        break

                cur_time  = time.time()
                elapsed   = cur_time - last_time
                last_time = cur_time
                if elapsed > 0:
                    anlik_fps = 0.9 * anlik_fps + 0.1 * (1.0 / elapsed)
                print(
                    f"EP: {ep+1} | Steer: {pred_action[0]:.2f} | "
                    f"Throttle {pred_action[1]:.2f} | "
                    f"Spd: {ego_reading.total_speed:+.2f} | FPS: {anlik_fps:.1f}",
                    end="\r",
                )

            success_flags.append(bool(info.get("arrive_dest", False)))
            route_completions.append(info.get("route_completion", 0.0))
            print(f"\nEpisode {ep+1} done. Success: {success_flags[-1]}")

        print(
            f"\n=== ONLINE SUMMARY ===\n"
            f"Success: {np.mean(success_flags)*100:.1f}%  "
            f"Route: {np.mean(route_completions)*100:.1f}%"
        )
        env.close()
        cv2.destroyAllWindows()


# ==========================================
# 9. MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["train", "finetune", "test", "all"],
        help="'finetune' loads --finetune_from and continues training.",
    )
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--episodes",   type=int,   default=1)
    parser.add_argument("--data_dir",   type=str,   default="dataset")
    parser.add_argument("--dpt_path",   type=str,   default="models/dpt_finetuned.pth")
    parser.add_argument("--model_path", type=str,   default="models/policy_model.pth",
                        help="Path to save the (fine-tuned) model.")
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument(
        "--pred_dir", type=str, default=None,
        help="Path to precomputed DPT predictions (train_dpt.py --mode precompute).",
    )
    parser.add_argument(
        "--test_mode", type=str, default="all",
        choices=["offline", "simulation", "all"],
    )

    # ── Fine-tune specific args ────────────────────────────────────────────────
    parser.add_argument(
        "--finetune_from", type=str, default=None,
        help="Checkpoint to load for fine-tuning (required when --mode finetune).",
    )
    parser.add_argument(
        "--freeze_backbone", action="store_true",
        help="Freeze CNN/backbone layers during fine-tuning (only train head).",
    )
    parser.add_argument(
        "--reset_optimizer", action="store_true",
        help="Ignore saved optimizer/scheduler state; start fresh (good for new datasets).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Restore optimizer & scheduler state to continue seamlessly from checkpoint.",
    )

    args = parser.parse_args()

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.mode in ("train", "all"):
        train_policy(
            args.epochs, 32, args.model_path,
            args.dpt_path, args.data_dir, args.lr,
            pred_dir=args.pred_dir,
        )

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
            lr              = args.lr if args.lr != 1e-4 else 1e-6,  # sensible ft default
            pred_dir        = args.pred_dir,
            freeze_bb       = args.freeze_backbone,
            reset_optimizer = args.reset_optimizer,
            resume          = args.resume,
        )

    if args.mode in ("test", "all"):
        test_policy(
            args.model_path, args.dpt_path, args.data_dir,
            args.episodes, pred_dir=args.pred_dir,
            test_mode=args.test_mode,
        )