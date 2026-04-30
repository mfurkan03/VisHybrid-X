"""
policy/datasets.py – Dataset classes for policy training.
"""

import glob
import os

import numpy as np
from torch.utils.data import Dataset

from models import EGO_DIM


class MetaDriveRGBDataset(Dataset):
    """
    Loads raw RGB frames + ego_state from .npz files.
    Falls back to zeros if ego_state is missing (older data).
    """

    def __init__(self, data_dir: str, split: str = "train"):
        split_dir  = os.path.join(data_dir, split)
        files      = glob.glob(os.path.join(split_dir, "*.npz"))
        self.rgb_frames: list = []
        self.actions:    list = []
        self.ego_states: list = []

        for f in files:
            try:
                data = np.load(f, allow_pickle=True)
            except Exception:
                continue
            rgb_keys = [k for k in data.files if k.endswith("_rgb")]
            if not rgb_keys or "action" not in data.files:
                continue
            n   = len(data["action"])
            ego = data["ego_state"] if "ego_state" in data.files else np.zeros((n, EGO_DIM), dtype=np.float32)
            self.rgb_frames.extend(data[rgb_keys[0]])
            self.actions.extend(data["action"])
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
    Fast dataset using pre-cached DPT depth + raw RGB + ego_state.
    Lane masks are computed at training time (in trainer.py) rather than
    stored on disk, so the dataset returns the raw RGB frames needed for that.

    Expected file layout:
        <pred_dir>/<split>/episode_N.npz
            depth_pred : float32  (N, 1, 84, 84)
            rgb        : uint8    (N, 84, 84, 3)
            action     : float32  (N, 2)
            ego_state  : float32  (N, EGO_DIM)  ← zeros if missing

    Note: older files that contain a `lane_mask` key instead of `rgb` are
    skipped with a warning — re-run train_dpt.py --mode precompute to
    regenerate them.
    """

    def __init__(self, pred_dir: str, split: str = "train"):
        split_dir         = os.path.join(pred_dir, split)
        files             = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
        self.depth_frames: list = []
        self.rgb_frames:   list = []
        self.actions:      list = []
        self.ego_states:   list = []

        skipped = 0
        for f in files:
            try:
                data = np.load(f, allow_pickle=True)
            except Exception as e:
                print(f"[WARNING] Could not load {f}: {e}")
                continue

            # Guard: reject old files that have lane_mask but no rgb
            if "rgb" not in data.files:
                print(f"[WARNING] {os.path.basename(f)} has no 'rgb' key "
                      f"(old precomputed format). Skipping — re-run precompute.")
                skipped += 1
                continue

            if not {"depth_pred", "action"}.issubset(data.files):
                print(f"[WARNING] Missing required keys in {f}, skipping.")
                continue

            depths  = data["depth_pred"]               # (N, 1, 84, 84)
            rgbs    = data["rgb"]                      # (N, 84, 84, 3)
            actions = data["action"]
            n       = min(len(depths), len(rgbs), len(actions))
            ego     = (data["ego_state"][:n] if "ego_state" in data.files
                       else np.zeros((n, EGO_DIM), dtype=np.float32))

            self.depth_frames.extend(depths[:n])
            self.rgb_frames.extend(rgbs[:n])
            self.actions.extend(actions[:n])
            self.ego_states.extend(ego)

        if skipped:
            print(f"[WARNING] Skipped {skipped} file(s) with old format. "
                  f"Delete data/processed/dpt_pred and re-run: "
                  f"python src/train_dpt.py --mode precompute")

        print(f"[INFO] PrecomputedDepthDataset ({split}): "
              f"{len(self.actions)} samples from {split_dir}.")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (
            self.depth_frames[idx],                        # float32 (1, 84, 84)
            self.rgb_frames[idx],                          # uint8   (84, 84, 3)
            np.array(self.actions[idx],    dtype=np.float32),
            np.array(self.ego_states[idx], dtype=np.float32),
        )