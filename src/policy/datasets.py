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
    Fast dataset using pre-cached DPT depth + lane + ego_state.

    Expected file layout:
        <pred_dir>/<split>/episode_N.npz
            depth_pred : float32  (N, 1, 84, 84)
            lane_mask  : float32  (N, 1, 84, 84)
            action     : float32  (N, 2)
            ego_state  : float32  (N, EGO_DIM)   ← zeros if missing
    """

    def __init__(self, pred_dir: str, split: str = "train"):
        split_dir          = os.path.join(pred_dir, split)
        files              = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
        self.depth_frames: list = []
        self.lane_frames:  list = []
        self.actions:      list = []
        self.ego_states:   list = []

        for f in files:
            try:
                data = np.load(f, allow_pickle=True)
            except Exception as e:
                print(f"[WARNING] Could not load {f}: {e}")
                continue
            if not {"depth_pred", "lane_mask", "action"}.issubset(data.files):
                print(f"[WARNING] Missing keys in {f}, skipping.")
                continue
            depths  = data["depth_pred"]
            lanes   = data["lane_mask"]
            actions = data["action"]
            n       = min(len(depths), len(lanes), len(actions))
            ego     = data["ego_state"][:n] if "ego_state" in data.files else np.zeros((n, EGO_DIM), dtype=np.float32)
            self.depth_frames.extend(depths[:n])
            self.lane_frames.extend(lanes[:n])
            self.actions.extend(actions[:n])
            self.ego_states.extend(ego)

        print(f"[INFO] PrecomputedDepthDataset ({split}): "
              f"{len(self.actions)} samples from {split_dir}.")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        combined = np.concatenate([self.depth_frames[idx], self.lane_frames[idx]], axis=0)
        return (
            combined,
            np.array(self.actions[idx],    dtype=np.float32),
            np.array(self.ego_states[idx], dtype=np.float32),
        )