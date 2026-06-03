"""
policy/datasets.py – Dataset classes for policy training.
"""

import glob
import os

import numpy as np
from torch.utils.data import Dataset

from models import EGO_DIM, EGO_MOTION_DIM, NAVI_DIM


# Optional nav early-warning: extend each turn command backward in time by this
# many frames so the model is told about an upcoming junction turn EARLIER and
# has room to set up, instead of only learning the command once it is already in
# the intersection (the reported "it decides the turn too late").  OFF (0) by
# default — enable per-run with e.g.  NAV_WARMUP_FRAMES=4 python train_...  .
# Smearing happens per-episode inside _load_navi, so it never bleeds one
# episode's turn into another.  The expert's *steer* during these added frames
# is still ≈0 (it has not started turning yet), so the model learns "command is
# on, but stay straight until the junction visual arrives" — it does not turn
# early; it just commits to the correct direction sooner.
NAV_WARMUP_FRAMES = int(os.environ.get("NAV_WARMUP_FRAMES", "0"))

# Brake anticipation: make GENUINE braking start this many frames EARLIER so the
# policy learns to decelerate sooner instead of braking at the last second.  OFF
# (0) by default; enable per-run with e.g. BRAKE_ANTICIPATION_FRAMES=2 .  Applied
# per-episode; throttle (accel) channel only, steering untouched.  At
# decision_repeat=1/save_every_n=20 a frame ≈0.2 s, so k=2 ≈ 0.4 s of margin.
#
# IMPORTANT: only future frames that are *real* brakes (accel < the threshold
# below) are pulled forward.  A naive forward-min over ALL frames dragged the
# whole throttle target negative — this expert coasts/feathers, dipping below 0
# ~30% of the time, so the min of the next k frames is almost always lower than
# the current frame and the car ends up braking everywhere (it stopped on open
# road).  Gating on a real-brake threshold keeps cruising throttle intact.
BRAKE_ANTICIPATION_FRAMES = int(os.environ.get("BRAKE_ANTICIPATION_FRAMES", "0"))
BRAKE_ANTICIPATION_THRESH = float(os.environ.get("BRAKE_ANTICIPATION_THRESH", "-0.1"))


def _smear_back(col: np.ndarray, k: int) -> np.ndarray:
    """Extend every active (==1) run in a 1-D flag array backward by k frames."""
    if k <= 0:
        return col
    out = col.copy()
    for _ in range(k):
        out[:-1] = np.maximum(out[:-1], out[1:])   # pull each active flag one frame earlier
    return out


def _anticipate_brake(actions: np.ndarray, k: int,
                      brake_thresh: float = BRAKE_ANTICIPATION_THRESH) -> np.ndarray:
    """Start GENUINE braking up to k frames earlier, WITHOUT lowering cruising throttle.

    For each frame t, look ahead k frames; only frames that are real brakes
    (accel < brake_thresh) are eligible to be pulled forward, and accel[t] is
    lowered to that brake value. Frames with no real brake ahead are left
    untouched — essential, because this expert coasts/feathers (throttle < 0
    ~30% of the time) and a naive forward-min would drag the whole target
    negative and make the car brake everywhere. Per-episode; steering untouched."""
    if k <= 0 or len(actions) == 0:
        return actions
    out   = actions.astype(np.float32, copy=True)
    accel = actions[:, 1].astype(np.float32)
    n     = len(accel)
    future_brake = np.full(n, np.inf, dtype=np.float32)   # min real-brake within the window
    for i in range(1, k + 1):
        cand = accel[i:]
        real = np.where(cand < brake_thresh, cand, np.inf)            # genuine brakes only
        future_brake[:n - i] = np.minimum(future_brake[:n - i], real)
    out[:, 1] = np.minimum(accel, future_brake)            # inf where no real brake → accel unchanged
    return out


def _load_navi(data, n_frames: int) -> np.ndarray:
    """Return (n_frames, NAVI_DIM) from the file's navi_state key, or zeros.
    Files store 2 cols [navi_left, navi_right]; navi_forward is derived as 1-left-right.
    If NAV_WARMUP_FRAMES>0, the left/right commands are smeared backward in time first."""
    out = np.zeros((n_frames, NAVI_DIM), dtype=np.float32)
    if "navi_state" in data.files:
        navi  = data["navi_state"].astype(np.float32)   # (N, 2) on disk
        m     = min(len(navi), n_frames)
        left  = _smear_back(navi[:m, 0].copy(), NAV_WARMUP_FRAMES)
        right = _smear_back(navi[:m, 1].copy(), NAV_WARMUP_FRAMES)
        out[:m, 0] = left                              # navi_left
        out[:m, 1] = right                             # navi_right
        if NAVI_DIM >= 3:
            # forward = whatever is left over; clip in case a smear overlaps.
            out[:m, 2] = np.clip(1.0 - left - right, 0.0, 1.0)  # navi_forward
    else:
        # No nav data: treat all frames as forward (safe default)
        if NAVI_DIM >= 3:
            out[:, 2] = 1.0
    return out


class MetaDriveRGBDataset(Dataset):
    """
    Loads raw RGB frames + ego_state (motion) + navigation from .npz files.
    Falls back to zeros if ego_state / navigation is missing (older data).

    The model ego vector returned by __getitem__ is the 5-dim
    [total_speed, last_steer, heading_delta, navi_left, navi_right].
    """

    def __init__(self, data_dir: str, split: str = "train"):
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
            navi = _load_navi(data, n)
            ego5 = np.concatenate([ego[:n], navi[:n]], axis=1)   # (n, EGO_DIM)
            acts = _anticipate_brake(np.asarray(data["action"], dtype=np.float32),
                                     BRAKE_ANTICIPATION_FRAMES)
            self.rgb_frames.extend(data[rgb_keys[0]])
            self.actions.extend(acts)
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

    Note: older files that contain a `lane_mask` key instead of `rgb` are
    skipped with a warning — re-run train_dpt.py --mode precompute to
    regenerate them.
    """

    def __init__(self, pred_dir: str, split: str = "train"):
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
            navi = _load_navi(data, n)
            if "navi_state" in data.files:
                n_nav_loaded += 1

            acts = _anticipate_brake(np.asarray(actions[:n], dtype=np.float32),
                                     BRAKE_ANTICIPATION_FRAMES)
            self.depth_frames.extend(depths[:n])
            self.rgb_frames.extend(rgbs[:n])
            self.actions.extend(acts)
            self.navi_states.extend(navi[:n])
            self.ego_full_states.extend(ego_full)

        if skipped:
            print(f"[WARNING] Skipped {skipped} file(s) with old format. "
                  f"Delete data/processed/dpt_pred and re-run: "
                  f"python src/train_dpt.py --mode precompute")

        print(f"[INFO] PrecomputedDepthDataset ({split}): "
              f"{len(self.actions)} samples from {split_dir} "
              f"[{n_nav_loaded}/{len(files) - skipped} episodes with nav].")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        ego_full = np.array(self.ego_full_states[idx], dtype=np.float32)
        navi     = np.array(self.navi_states[idx],     dtype=np.float32)   # (NAVI_DIM,)
        # Model ego (EGO_DIM=6): [total_speed, last_steer, heading_delta] ++ [navi_left, navi_right, navi_forward]
        ego_model = np.concatenate([ego_full[[0, 1, 4]], navi]).astype(np.float32)
        return (
            self.depth_frames[idx],
            self.rgb_frames[idx],
            np.array(self.actions[idx], dtype=np.float32),
            ego_model,            # EGO_DIM=5 model input
            ego_full,             # full 5-dim physical state kept for heading metrics
        )