"""
policy/datasets.py – Dataset classes for policy training.
"""

import glob
import os

import numpy as np
from torch.utils.data import Dataset

from models import EGO_DIM, EGO_MOTION_DIM, NAVI_DIM


def _load_navi(data, nav_dir: str, split: str, episode_basename: str, n_frames: int) -> np.ndarray:
    """
    Return (n_frames, NAVI_DIM) navigation array.

    Priority:
    1. `navi_state` key in the file itself (new datasets)
    2. Separate nav_dir folder matched by filename (legacy datasets)
    3. Zeros (= forward command) for old data with no nav at all
    """
    if "navi_state" in data.files:
        navi = data["navi_state"].astype(np.float32)
        m = min(len(navi), n_frames)
        out = np.zeros((n_frames, NAVI_DIM), dtype=np.float32)
        out[:m] = navi[:m, :NAVI_DIM]
        return out
    if nav_dir:
        nav_path = os.path.join(nav_dir, split, episode_basename)
        if os.path.exists(nav_path):
            try:
                navi = np.load(nav_path, allow_pickle=True)["navi_state"].astype(np.float32)
                m = min(len(navi), n_frames)
                out = np.zeros((n_frames, NAVI_DIM), dtype=np.float32)
                out[:m] = navi[:m, :NAVI_DIM]
                return out
            except Exception as e:
                print(f"[WARNING] Could not read nav file {nav_path}: {e}")
    return np.zeros((n_frames, NAVI_DIM), dtype=np.float32)


class MetaDriveRGBDataset(Dataset):
    """
    Loads raw RGB frames + ego_state (motion) + navigation from .npz files.
    Falls back to zeros if ego_state / navigation is missing (older data).

    The model ego vector returned by __getitem__ is the 5-dim
    [total_speed, last_steer, heading_delta, navi_left, navi_right].
    """

    def __init__(self, data_dir: str, split: str = "train", nav_dir: str = None):
        split_dir  = os.path.join(data_dir, split)
        files      = glob.glob(os.path.join(split_dir, "*.npz"))
        self.rgb_frames: list = []
        self.actions:    list = []
        self.ego_states: list = []   # 5-dim model ego (motion ++ nav)

        for f in files:
            try:
                data = np.load(f, allow_pickle=True)
            except Exception:
                continue
            rgb_keys = [k for k in data.files if k.endswith("_rgb")]
            if not rgb_keys or "action" not in data.files:
                continue
            n   = len(data["action"])
            ego = (data["ego_state"][:, :EGO_MOTION_DIM] if "ego_state" in data.files
                   else np.zeros((n, EGO_MOTION_DIM), dtype=np.float32))
            navi = _load_navi(data, nav_dir, split, os.path.basename(f), n)
            ego5 = np.concatenate([ego[:n], navi[:n]], axis=1)   # (n, EGO_DIM)
            self.rgb_frames.extend(data[rgb_keys[0]])
            self.actions.extend(data["action"])
            self.ego_states.extend(ego5)

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
            depth_pred     : float32  (N, 1, 196, 196)
            rgb            : uint8    (N, H, W, 3)
            action         : float32  (N, 2)
            ego_state_full : float32  (N, 5)   ← motion state; zeros if missing
            navi_state     : float32  (N, 2)   ← [navi_left, navi_right]; zeros if missing

    nav_dir is a legacy fallback for old datasets where navi_state was stored
    in a separate folder. New datasets have navi_state inline in each file.

    Note: older files that contain a `lane_mask` key instead of `rgb` are
    skipped with a warning — re-run train_dpt.py --mode precompute to
    regenerate them.
    """

    def __init__(self, pred_dir: str, split: str = "train", nav_dir: str = None):
        split_dir         = os.path.join(pred_dir, split)
        files             = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
        self.depth_frames:    list = []
        self.rgb_frames:      list = []
        self.actions:         list = []
        self.navi_states:     list = []   # (NAVI_DIM,) per frame
        self.ego_full_states: list = []

        skipped = 0
        n_nav_loaded = 0
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

            depths  = data["depth_pred"]
            rgbs    = data["rgb"]
            actions = data["action"]
            n       = min(len(depths), len(rgbs), len(actions))
            ego_full = (data["ego_state_full"][:n] if "ego_state_full" in data.files
                        else np.zeros((n, 5), dtype=np.float32))
            navi = _load_navi(data, nav_dir, split, os.path.basename(f), n)
            if "navi_state" in data.files or (nav_dir and os.path.exists(os.path.join(nav_dir, split, os.path.basename(f)))):
                n_nav_loaded += 1

            self.depth_frames.extend(depths[:n])
            self.rgb_frames.extend(rgbs[:n])
            self.actions.extend(actions[:n])
            self.navi_states.extend(navi[:n])
            self.ego_full_states.extend(ego_full)

        if skipped:
            print(f"[WARNING] Skipped {skipped} file(s) with old format. "
                  f"Delete data/processed/dpt_pred and re-run: "
                  f"python src/train_dpt.py --mode precompute")

        nav_note = (f"+nav ({n_nav_loaded} episodes)" if nav_dir else "nav=zeros")
        print(f"[INFO] PrecomputedDepthDataset ({split}): "
              f"{len(self.actions)} samples from {split_dir} [{nav_note}].")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        ego_full = np.array(self.ego_full_states[idx], dtype=np.float32)
        navi     = np.array(self.navi_states[idx],     dtype=np.float32)   # (NAVI_DIM,)
        # Model ego (EGO_DIM=5): [total_speed, last_steer, heading_delta] ++ [navi_left, navi_right]
        ego_model = np.concatenate([ego_full[[0, 1, 4]], navi]).astype(np.float32)
        return (
            self.depth_frames[idx],
            self.rgb_frames[idx],
            np.array(self.actions[idx], dtype=np.float32),
            ego_model,            # EGO_DIM=5 model input
            ego_full,             # full 5-dim physical state kept for heading metrics
        )