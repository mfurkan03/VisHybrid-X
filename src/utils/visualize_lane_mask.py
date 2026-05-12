"""
Quick visual check of the lane mask on a few sample frames.
Uses apply_lane_mask / get_lane_mask_visual from trainer.py directly.
Columns: original | white mask | yellow mask | colored mask | masked RGB (α=0)
The colored mask shows white-detected regions as white and yellow-detected regions
as yellow so it's clear the training input preserves color distinction.
Run from repo root: python src/utils/visualize_lane_mask.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from policy.trainer import apply_lane_mask, get_lane_mask_visual

NPZ      = "data/raw/mixed/train/42_0.0_3_4000_0.70_0.npz"
N_FRAMES = 6

data    = np.load(NPZ)
rgb_key = next(k for k in data.keys() if k.endswith("rgb"))
rgb_all = data[rgb_key]   # (T, H, W, 3) uint8

device  = torch.device("cpu")
indices = np.linspace(0, len(rgb_all) - 1, N_FRAMES, dtype=int)

COLS = ["original", "white mask", "yellow mask", "colored mask", "masked RGB (α=0)"]
fig, axes = plt.subplots(N_FRAMES, len(COLS), figsize=(4 * len(COLS), N_FRAMES * 3))
fig.suptitle("Lane mask inspection", fontsize=14)

for col, title in enumerate(COLS):
    axes[0, col].set_title(title, fontsize=10)

def _roi(img_uint8):
    h = img_uint8.shape[0]
    roi = img_uint8.copy()
    roi[0:int(h * 0.55), :] = 0
    return roi

for row, idx in enumerate(indices):
    frame  = rgb_all[idx]    # (H, W, 3) uint8
    rgb_np = frame[None]     # (1, H, W, 3) uint8

    H, W    = frame.shape[:2]
    depth_t = torch.zeros(1, 1, H, W, device=device)

    roi  = _roi(frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    _, white  = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    hsv       = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    yellow    = cv2.inRange(hsv, np.array([15, 80, 165]), np.array([35, 255, 255]))

    # Colored mask: white-detected → white, yellow-detected → yellow.
    # Yellow overwrites white so overlapping pixels show as yellow.
    colored = np.zeros((*frame.shape[:2], 3), dtype=np.uint8)
    colored[white  > 0] = [255, 255, 255]
    colored[yellow > 0] = [255, 220,   0]

    out_0 = apply_lane_mask(depth_t, rgb_np, device, always_lane_masked=True)
    rgb_0 = (out_0[0, 1:].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    for col, (img, kw) in enumerate([
        (frame,   {}),
        (white,   {"cmap": "gray"}),
        (yellow,  {"cmap": "gray"}),
        (colored, {}),
        (rgb_0,   {}),
    ]):
        axes[row, col].imshow(img, **kw)
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(f"frame {idx}", fontsize=8)

plt.tight_layout()
out = "lane_mask_check.png"
plt.savefig(out, dpi=120)
print(f"Saved → {out}")
plt.show()
