"""
Quick visual check of the lane mask on a few sample frames.
Columns: original | white mask | yellow mask | combined mask | masked RGB (alpha=0)
Run from repo root: python src/utils/visualize_lane_mask.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import cv2
import numpy as np
import matplotlib.pyplot as plt

NPZ      = "data/raw/mixed/train/42_0.0_3_4000_0.70_0.npz"
N_FRAMES = 6

data    = np.load(NPZ)
rgb_key = next(k for k in data.keys() if k.endswith("rgb"))
rgb_all = data[rgb_key]   # (T, H, W, 3) uint8

def _roi(img_uint8):
    h = img_uint8.shape[0]
    roi = img_uint8.copy()
    roi[0:int(h * 0.55), :] = 0
    return roi

def white_mask(frame, threshold=180):
    roi = _roi(frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    _, m = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return m

def yellow_mask(frame):
    roi = _roi(frame)
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv, np.array([15, 80, 165]), np.array([35, 255, 255]))

indices = np.linspace(0, len(rgb_all) - 1, N_FRAMES, dtype=int)

COLS = ["original", "white mask", "yellow mask", "combined mask", "masked RGB (α=0)"]
fig, axes = plt.subplots(N_FRAMES, len(COLS), figsize=(4 * len(COLS), N_FRAMES * 3))
fig.suptitle("Lane mask inspection", fontsize=14)

for col, title in enumerate(COLS):
    axes[0, col].set_title(title, fontsize=10)

for row, idx in enumerate(indices):
    frame  = rgb_all[idx]
    wm     = white_mask(frame)
    ym     = yellow_mask(frame)
    cm     = cv2.bitwise_or(wm, ym)
    masked = (frame * (cm[:, :, None] / 255.0)).astype("uint8")

    for col, (img, kw) in enumerate([
        (frame,  {}),
        (wm,     {"cmap": "gray"}),
        (ym,     {"cmap": "gray"}),
        (cm,     {"cmap": "gray"}),
        (masked, {}),
    ]):
        axes[row, col].imshow(img, **kw)
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(f"frame {idx}", fontsize=8)

plt.tight_layout()
out = "lane_mask_check.png"
plt.savefig(out, dpi=120)
print(f"Saved → {out}")
plt.show()
