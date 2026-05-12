"""
Interactive HSV tuner for yellow lane detection.
Drag sliders to adjust the HSV range; the mask updates live.
When satisfied, the final values are printed — paste them into trainer.py.

Run from repo root: python src/utils/tune_yellow_mask.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

NPZ      = "data/raw/mixed/train/42_0.0_3_4000_0.70_0.npz"
FRAME_IDX = 50   # change to a frame that shows both yellow lanes AND green hills

data    = np.load(NPZ)
rgb_key = next(k for k in data.keys() if k.endswith("rgb"))
rgb_all = data[rgb_key]

FRAME_IDX = min(FRAME_IDX, len(rgb_all) - 1)
frame = rgb_all[FRAME_IDX]   # (H, W, 3) uint8

h = frame.shape[0]
roi = frame.copy()
roi[0:int(h * 0.55), :] = 0

# Initial HSV bounds  (OpenCV H: 0-180, S/V: 0-255)
INIT = dict(h_lo=15, h_hi=35, s_lo=80, s_hi=255, v_lo=165, v_hi=255)

def compute_mask(h_lo, h_hi, s_lo, s_hi, v_lo, v_hi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv,
                       np.array([h_lo, s_lo, v_lo]),
                       np.array([h_hi, s_hi, v_hi]))

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
plt.subplots_adjust(bottom=0.42)
axes[0].set_title("original (ROI)");  im_orig   = axes[0].imshow(roi)
axes[1].set_title("yellow mask");     im_mask   = axes[1].imshow(compute_mask(**INIT), cmap="gray", vmin=0, vmax=255)
axes[2].set_title("overlay");
overlay_init = roi.copy(); overlay_init[compute_mask(**INIT) > 0] = [255, 0, 0]
im_over = axes[2].imshow(overlay_init)
for ax in axes: ax.axis("off")

# --- Sliders ---
slider_specs = [
    ("H low",  0, 180, INIT["h_lo"],  0.06),
    ("H high", 0, 180, INIT["h_hi"],  0.12),
    ("S low",  0, 255, INIT["s_lo"],  0.18),
    ("S high", 0, 255, INIT["s_hi"],  0.24),
    ("V low",  0, 255, INIT["v_lo"],  0.30),
    ("V high", 0, 255, INIT["v_hi"],  0.36),
]
sliders = {}
for label, vmin, vmax, vinit, bottom in slider_specs:
    ax_sl = plt.axes([0.15, bottom, 0.70, 0.04])
    sliders[label] = Slider(ax_sl, label, vmin, vmax, valinit=vinit, valstep=1)

def update(_):
    vals = {k: int(v.val) for k, v in sliders.items()}
    mask = compute_mask(vals["H low"], vals["H high"],
                        vals["S low"], vals["S high"],
                        vals["V low"], vals["V high"])
    overlay = roi.copy(); overlay[mask > 0] = [255, 0, 0]
    im_mask.set_data(mask)
    im_over.set_data(overlay)
    fig.canvas.draw_idle()

for s in sliders.values():
    s.on_changed(update)

def on_close(_):
    vals = {k: int(v.val) for k, v in sliders.items()}
    print("\n--- Final HSV bounds (paste into trainer.py) ---")
    print(f"yellow_mask = cv2.inRange(hsv,")
    print(f"              np.array([{vals['H low']}, {vals['S low']}, {vals['V low']}]),")
    print(f"              np.array([{vals['H high']}, {vals['S high']}, {vals['V high']}]))")

fig.canvas.mpl_connect("close_event", on_close)
plt.suptitle("Tune yellow HSV range — close window to print final values", fontsize=11)
plt.show()
