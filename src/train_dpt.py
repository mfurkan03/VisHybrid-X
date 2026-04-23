import os
import glob
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import sys
from pathlib import Path
from tqdm import tqdm

target_folder = Path(__file__).resolve().parent.parent / 'Depth-Anything-V2'
sys.path.append(str(target_folder))
from depth_anything_v2.dpt import DepthAnythingV2

# ==========================================
# 1. DEPTH MODEL
# ==========================================
class DepthEstimationModel:
    def __init__(self, encoder='vits', trainable=True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.trainable = trainable

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        self.model = DepthAnythingV2(**model_configs[encoder])
        ckpt_path = f'{target_folder}/checkpoints/depth_anything_v2_{encoder}.pth'
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
            print(f"[INFO] DepthAnythingV2 loaded: {ckpt_path}")
        else:
            print(f"[WARNING] Checkpoint not found: {ckpt_path}")

        self.model = self.model.to(self.device)
        self.use_fp16 = (self.device == 'cuda')

        if self.trainable:
            self.set_train_mode()
        else:
            self.set_eval_mode()

    def set_train_mode(self):
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True

    def set_eval_mode(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def predict_batch(self, rgb_images: np.ndarray) -> torch.Tensor:
        results = []
        for rgb in rgb_images:
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            depth_raw = self.model.infer_image(rgb_bgr)
            d_tensor = torch.from_numpy(depth_raw).to(self.device).unsqueeze(0).unsqueeze(0)

            import torch.nn.functional as F
            depth_resized = F.interpolate(d_tensor, size=(84, 84), mode='bilinear', align_corners=False)
            d_min, d_max = depth_resized.min(), depth_resized.max()
            depth_norm = (depth_resized - d_min) / (d_max - d_min + 1e-6)
            results.append(depth_norm.float())

        return torch.cat(results, dim=0)

    def predict_batch_with_grad(self, rgb_images: np.ndarray) -> torch.Tensor:
        results = []
        import torch.nn.functional as F
        for rgb in rgb_images:
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)

            h, w = rgb.shape[:2]
            new_h = int(np.round(h / 14.0)) * 14
            new_w = int(np.round(w / 14.0)) * 14
            rgb_resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            rgb_bgr = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR)

            img = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            img = (img - mean) / std

            if self.use_fp16:
                with torch.amp.autocast('cuda'):
                    depth = self.model(img)
            else:
                depth = self.model(img)

            if depth.dim() == 3: depth = depth.unsqueeze(1)
            elif depth.dim() == 2: depth = depth.unsqueeze(0).unsqueeze(0)

            depth_resized = F.interpolate(depth, size=(84, 84), mode='bilinear', align_corners=False)
            d_min = depth_resized.amin(dim=(-2, -1), keepdim=True)
            d_max = depth_resized.amax(dim=(-2, -1), keepdim=True)
            depth_norm = (depth_resized - d_min) / (d_max - d_min + 1e-6)
            results.append(depth_norm.float())

        return torch.cat(results, dim=0)


# ==========================================
# 2. DATASET & LOSS
# ==========================================
class MetaDriveDepthDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        split_dir = os.path.join(data_dir, split)
        self.files      = glob.glob(os.path.join(split_dir, "*.npz"))
        self.rgb_frames = []
        self.gt_depths  = []

        for f in self.files:
            try: data = np.load(f, allow_pickle=True)
            except Exception: continue

            rgb_keys      = [k for k in data.files if k.endswith('_rgb')]
            combined_keys = [k for k in data.files if k.endswith('_combined')]
            if not rgb_keys or not combined_keys: continue

            self.rgb_frames.extend(data[rgb_keys[0]])
            self.gt_depths.extend(data[combined_keys[0]][:, 0:1, :, :])

        print(f"[INFO] DepthDataset ({split}): {len(self.gt_depths)} samples loaded from {split_dir}.")

    def __len__(self): return len(self.gt_depths)
    def __getitem__(self, idx):
        return self.rgb_frames[idx], np.array(self.gt_depths[idx], dtype=np.float32)


def silog_loss(pred: torch.Tensor, target: torch.Tensor, variance_focus: float = 0.85) -> torch.Tensor:
    eps = 1e-6
    if pred.shape != target.shape:
        import torch.nn.functional as F
        target = F.interpolate(target, size=pred.shape[-2:], mode='bilinear', align_corners=False)

    pred   = pred.clamp(min=eps)
    target = target.clamp(min=eps)

    d = torch.log(pred) - torch.log(target)
    loss = d.pow(2).mean() - variance_focus * d.mean().pow(2)
    return loss


# ==========================================
# 3. DPT TRAINING LOOP
# ==========================================
def train_dpt(epochs=20, batch_size=64, model_path="dpt_finetuned.pth", data_dir="dataset", lr=1e-5):
    print("--- Phase 1: Training Depth Model (DPT) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    depth_estimator = DepthEstimationModel(trainable=True)

    train_ds = MetaDriveDepthDataset(data_dir=data_dir, split="train")
    val_ds   = MetaDriveDepthDataset(data_dir=data_dir, split="val")

    def collate_fn(batch):
        rgbs, depths = zip(*batch)
        return np.stack(rgbs), torch.tensor(np.stack(depths), dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = optim.AdamW(depth_estimator.model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
    import math
    min_val_loss = math.inf

    for epoch in range(epochs):
        depth_estimator.set_train_mode()
        train_loss = 0.0

        for rgb_np, gt_depth in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            gt_depth = gt_depth.to(device)
            optimizer.zero_grad()
            pred_depth = depth_estimator.predict_batch_with_grad(rgb_np)
            loss = silog_loss(pred_depth, gt_depth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(depth_estimator.model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        depth_estimator.set_eval_mode()
        val_loss = 0.0
        with torch.no_grad():
            for rgb_np, gt_depth in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                gt_depth = gt_depth.to(device)
                pred_depth = depth_estimator.predict_batch(rgb_np)
                val_loss += silog_loss(pred_depth, gt_depth).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)

        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6e}")
        scheduler.step()

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            torch.save(depth_estimator.model.state_dict(), model_path)
            print(f"*** Saved new best DPT checkpoint: {model_path} (Val Loss: {min_val_loss:.4f}) ***")


# ==========================================
# 4. PRECOMPUTE DPT PREDICTIONS
# ==========================================
def precompute_dpt_predictions(
    dpt_path:    str,
    data_dir:    str  = "dataset",
    out_dir:     str  = "data/processed/dpt_pred",
    batch_size:  int  = 32,
    splits:      tuple = ("train", "val"),
):
    """
    Run the (optionally fine-tuned) DPT model over all RGB frames in the
    requested splits and cache the resulting depth + lane-mask tensors to
    disk.  The policy trainer can then load these directly, skipping
    per-step inference completely.

    Output layout
    --------------
    <out_dir>/
      train/
        episode_0.npz   # keys: depth_pred [N,1,84,84], lane_mask [N,1,84,84], action [N,2]
        episode_1.npz
        ...
      val/
        episode_0.npz
        ...

    Each .npz mirrors the episode-level structure of the raw dataset so
    the policy DataLoader can index them the same way.
    """
    import torch.nn.functional as F

    print("--- Precomputing DPT Predictions ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load depth model (frozen) ----
    from depth_anything_v2.dpt import DepthAnythingV2
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    }
    dpt_model = DepthAnythingV2(**model_configs['vits'])
    if dpt_path and os.path.exists(dpt_path):
        dpt_model.load_state_dict(torch.load(dpt_path, map_location='cpu'))
        print(f"[INFO] Fine-tuned DPT loaded from: {dpt_path}")
    else:
        # Fall back to base checkpoint
        base_ckpt = str(target_folder / 'checkpoints' / 'depth_anything_v2_vits.pth')
        if os.path.exists(base_ckpt):
            dpt_model.load_state_dict(torch.load(base_ckpt, map_location='cpu'))
            print(f"[INFO] Base DPT checkpoint loaded from: {base_ckpt}")
        else:
            print("[WARNING] No DPT checkpoint found – using random weights.")
    dpt_model = dpt_model.to(device).eval()
    for p in dpt_model.parameters():
        p.requires_grad = False
    use_fp16 = (device.type == 'cuda')

    # ---- Helper: run DPT on a numpy RGB image ----
    def _infer_depth(rgb_np: np.ndarray) -> np.ndarray:
        """Returns a float32 array of shape (1, 84, 84), values in [0, 1]."""
        if rgb_np.dtype != np.uint8:
            rgb_np = (rgb_np * 255).astype(np.uint8) if rgb_np.max() <= 1.0 else rgb_np.astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

        if use_fp16:
            with torch.amp.autocast('cuda'):
                depth_raw = dpt_model.infer_image(rgb_bgr)
        else:
            depth_raw = dpt_model.infer_image(rgb_bgr)

        d = torch.from_numpy(depth_raw).to(device).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        d = F.interpolate(d, size=(84, 84), mode='bilinear', align_corners=False)
        d_min, d_max = d.min(), d.max()
        d = (d - d_min) / (d_max - d_min + 1e-6)
        return d.squeeze(0).cpu().numpy().astype(np.float32)  # (1, 84, 84)

    # ---- Helper: compute lane mask from RGB ----
    def _lane_mask(rgb_np: np.ndarray, threshold: int = 180) -> np.ndarray:
        """Returns a float32 array of shape (1, 84, 84), values in {0, 1}."""
        if rgb_np.max() <= 1.0:
            img_u8 = (rgb_np * 255).astype(np.uint8)
        else:
            img_u8 = rgb_np.astype(np.uint8)
        h, w = img_u8.shape[:2]
        roi = img_u8.copy()
        roi[: int(h * 0.55), :] = 0
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        resized = cv2.resize(mask, (84, 84), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
        return resized[np.newaxis]  # (1, 84, 84)

    # ---- Process each split ----
    for split in splits:
        split_in_dir  = os.path.join(data_dir, split)
        split_out_dir = os.path.join(out_dir, split)
        os.makedirs(split_out_dir, exist_ok=True)

        episode_files = sorted(glob.glob(os.path.join(split_in_dir, "*.npz")))
        if not episode_files:
            print(f"[WARNING] No .npz files found in {split_in_dir}. Skipping split '{split}'.")
            continue

        print(f"\n[{split.upper()}] Processing {len(episode_files)} episode file(s) -> {split_out_dir}")

        for ep_path in episode_files:
            ep_name = os.path.basename(ep_path)
            out_path = os.path.join(split_out_dir, ep_name)

            if os.path.exists(out_path):
                print(f"  [SKIP] {ep_name} already exists.")
                continue

            try:
                data = np.load(ep_path, allow_pickle=True)
            except Exception as e:
                print(f"  [ERROR] Could not load {ep_path}: {e}")
                continue

            rgb_keys = [k for k in data.files if k.endswith('_rgb')]
            if not rgb_keys:
                print(f"  [SKIP] No RGB key found in {ep_name}.")
                continue

            rgb_frames = data[rgb_keys[0]]   # (N, H, W, 3) uint8
            actions    = data['action'] if 'action' in data.files else None
            N          = len(rgb_frames)

            depth_preds = np.empty((N, 1, 84, 84), dtype=np.float32)
            lane_masks  = np.empty((N, 1, 84, 84), dtype=np.float32)

            # Run inference in mini-batches for tqdm granularity
            for start in tqdm(range(0, N, batch_size), desc=f"  {ep_name}", leave=False):
                end   = min(start + batch_size, N)
                batch = rgb_frames[start:end]
                for i, rgb in enumerate(batch):
                    depth_preds[start + i] = _infer_depth(rgb)
                    lane_masks[start + i]  = _lane_mask(rgb)

            save_dict = {
                "depth_pred": depth_preds,   # (N, 1, 84, 84) float32
                "lane_mask":  lane_masks,    # (N, 1, 84, 84) float32
            }
            if actions is not None:
                save_dict["action"] = actions

            np.savez_compressed(out_path, **save_dict)
            print(f"  Saved {ep_name}  [{N} frames]  ->  {out_path}")

    print(f"\n[INFO] Precomputation complete. Cached predictions are in: {out_dir}")


# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       type=str, default="train",
                        choices=["train", "precompute"],
                        help="'train' = fine-tune DPT; 'precompute' = cache DPT predictions for policy training")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--data_dir",   type=str,   default="dataset")
    parser.add_argument("--model_path", type=str,   default="models/dpt_finetuned.pth")
    parser.add_argument("--lr",         type=float, default=5e-5)
    # Precompute-only args
    parser.add_argument("--out_dir",    type=str,   default="data/processed/dpt_pred",
                        help="Where to write cached DPT predictions (used with --mode precompute)")
    parser.add_argument("--splits",     type=str,   nargs="+", default=["train", "val"],
                        help="Which dataset splits to precompute (default: train val)")
    args = parser.parse_args()

    if args.mode == "train":
        train_dpt(args.epochs, args.batch_size, args.model_path, args.data_dir, args.lr)

    elif args.mode == "precompute":
        precompute_dpt_predictions(
            dpt_path   = args.model_path,
            data_dir   = args.data_dir,
            out_dir    = args.out_dir,
            batch_size = args.batch_size,
            splits     = tuple(args.splits),
        )