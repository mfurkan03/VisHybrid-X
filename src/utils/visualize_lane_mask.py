"""
utils/visualize_lane_mask.py

Two modes:

  --mode poster       Static multi-frame grid saved to PNG.
                      Columns: original | white mask | yellow mask |
                               colored mask | masked RGB (α=0)

  --mode interactive  (default) Matplotlib window with a live alpha slider.
                      Shows 8 panels: original, mask, model-input RGB,
                      depth, overlay, non-lane, lane-only, stats.

Usage (from repo root):
    # interactive slider on a specific file / frame
    python src/utils/visualize_lane_mask.py \
        --npz data/processed/dpt_pred/train/episode_0.npz --frame 10

    # static poster of N evenly-spaced frames
    python src/utils/visualize_lane_mask.py --mode poster \
        --npz data/raw/mixed/train/42_0.0_3_4000_0.70_0.npz \
        --n_frames 6 --out visualizations/lane_mask_poster.png
"""

import argparse
import glob
import pathlib
import sys

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from policy.trainer import apply_lane_mask, get_lane_mask_visual


# ── shared helpers ─────────────────────────────────────────────────────────────

def _load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    rgb_keys   = [k for k in data.files if k.endswith("_rgb") or k == "rgb"]
    depth_keys = [k for k in data.files if "depth" in k]
    if not rgb_keys:
        raise ValueError(f"No RGB key in {npz_path}. Keys: {list(data.files)}")
    rgb_all   = data[rgb_keys[0]]
    depth_all = data[depth_keys[0]] if depth_keys else None
    return rgb_all, depth_all


def _auto_find_npz():
    roots = [
        "data/processed/dpt_pred",
        "data/raw",
        "dataset",
    ]
    for root in roots:
        hits = glob.glob(f"{root}/**/*.npz", recursive=True)
        if hits:
            return hits[0]
    return None


def _depth_colormap(depth_hw: np.ndarray | None, h: int, w: int) -> np.ndarray:
    if depth_hw is None:
        tile = np.full((h, w, 3), 40, dtype=np.uint8)
        cv2.putText(tile, "no depth", (10, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
        return tile
    import matplotlib.cm as cm
    d = depth_hw.squeeze()
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    rgb = (cm.inferno(d)[:, :, :3] * 255).astype(np.uint8)
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    return rgb


def _blend(rgb_hwc: np.ndarray, alpha: float, threshold: int) -> np.ndarray:
    """Return blended RGB (H, W, 3) float32 [0,1] matching apply_lane_mask."""
    img_f = rgb_hwc.astype(np.float32) / 255.0
    mask  = get_lane_mask_visual(rgb_hwc, threshold_value=threshold)
    m3    = (mask.astype(np.float32) / 255.0)[:, :, None]
    return img_f * m3 + img_f * (1.0 - m3) * alpha


def _maybe_resize(img: np.ndarray, size: int | None) -> np.ndarray:
    if size and img.shape[:2] != (size, size):
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img


# ── poster mode ───────────────────────────────────────────────────────────────

def _roi(img_uint8: np.ndarray) -> np.ndarray:
    h   = img_uint8.shape[0]
    roi = img_uint8.copy()
    roi[0:int(h * 0.55), :] = 0
    return roi


def run_poster(args):
    rgb_all, _ = _load_npz(args.npz)
    n       = min(args.n_frames, len(rgb_all))
    indices = np.linspace(0, len(rgb_all) - 1, n, dtype=int)
    device  = torch.device("cpu")

    cols = ["original", "white mask", "yellow mask", "colored mask", "masked RGB (α=0)"]
    fig, axes = plt.subplots(n, len(cols), figsize=(5 * len(cols), n * 4))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Lane mask inspection", fontsize=14)
    for col, title in enumerate(cols):
        axes[0, col].set_title(title, fontsize=10)

    for row, idx in enumerate(indices):
        frame  = rgb_all[idx]
        H, W   = frame.shape[:2]
        rgb_np = frame[None]
        depth_t = torch.zeros(1, 1, H, W, device=device)

        roi    = _roi(frame)
        gray   = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        _, white  = cv2.threshold(gray, args.threshold, 255, cv2.THRESH_BINARY)
        hsv       = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        yellow    = cv2.inRange(hsv, np.array([15, 80, 165]), np.array([35, 255, 255]))

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

    plt.tight_layout(pad=1.5)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved → {args.out}")
    plt.show()


# ── interactive mode ──────────────────────────────────────────────────────────

def run_interactive(args):
    from matplotlib.widgets import Slider

    rgb_all, depth_all = _load_npz(args.npz)
    frame_idx  = min(args.frame, len(rgb_all) - 1)
    rgb_hwc    = rgb_all[frame_idx]
    if rgb_hwc.max() <= 1.0:
        rgb_hwc = (rgb_hwc * 255).astype(np.uint8)
    else:
        rgb_hwc = rgb_hwc.astype(np.uint8)

    depth_hw = None
    if depth_all is not None:
        d = depth_all[frame_idx]
        depth_hw = (d[0] if d.ndim == 3 else d).astype(np.float32)

    H, W    = rgb_hwc.shape[:2]
    sz      = args.image_size
    mask_raw = get_lane_mask_visual(rgb_hwc, threshold_value=args.threshold)
    lane_frac = (mask_raw > 127).mean()

    def resize(img): return _maybe_resize(img, sz)

    # precompute static panels
    orig_d   = resize(rgb_hwc)
    mask_3ch = cv2.cvtColor(mask_raw, cv2.COLOR_GRAY2RGB)
    mask_d   = resize(mask_3ch)
    depth_d  = resize(_depth_colormap(depth_hw, H, W))

    lane_px = (resize(mask_raw) > 127) if sz else (mask_raw > 127)

    # ── figure ────────────────────────────────────────────────────────────────
    BG = "#1a1a2e"
    fig = plt.figure(figsize=(15, 7), facecolor=BG)
    fig.canvas.manager.set_window_title("Lane Mask Visualizer")

    gs   = gridspec.GridSpec(2, 4, figure=fig,
                             top=0.92, bottom=0.14,
                             left=0.03, right=0.97,
                             hspace=0.35, wspace=0.08)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]

    lbl  = dict(color="white", fontsize=9, pad=4)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")

    im_orig    = axes[0].imshow(orig_d)
    im_mask    = axes[1].imshow(mask_d)
    im_blend   = axes[2].imshow(orig_d)
    im_depth   = axes[3].imshow(depth_d)
    im_overlay = axes[4].imshow(orig_d)
    im_nonlane = axes[5].imshow(orig_d)
    im_lane    = axes[6].imshow(orig_d)

    axes[0].set_title("Original RGB",         **lbl)
    axes[1].set_title("Lane Mask",             **lbl)
    axes[2].set_title("Model Input (RGB ch.)", **lbl)
    axes[3].set_title("Depth Channel",         **lbl)
    axes[4].set_title("Mask Overlay",          **lbl)
    axes[5].set_title("Non-Lane Region",       **lbl)
    axes[6].set_title("Lane Region Only",      **lbl)
    axes[7].set_title("Alpha Info",            **lbl)
    axes[7].set_facecolor(BG)

    info_text = axes[7].text(
        0.5, 0.5, "", transform=axes[7].transAxes,
        ha="center", va="center", color="white",
        fontsize=10, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#2a2a4a", edgecolor="#6666aa"),
    )

    # slider
    ax_sl  = fig.add_axes([0.15, 0.05, 0.70, 0.03], facecolor="#2a2a4a")
    slider = Slider(ax_sl, "alpha  ", 0.0, 1.0, valinit=0.0, color="#5555cc")
    slider.label.set_color("white")
    slider.valtext.set_color("white")

    fig.text(0.15, 0.02, "α=0  →  only lane markings visible",
             color="#aaaacc", fontsize=8, ha="left")
    fig.text(0.85, 0.02, "α=1  →  full RGB",
             color="#aaaacc", fontsize=8, ha="right")
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Lane Mask Inspector  |  {pathlib.Path(args.npz).name}  frame {frame_idx}",
        color="white", fontsize=11,
    )

    def update(val):
        alpha     = slider.val
        blended_f = _blend(rgb_hwc, alpha, args.threshold)   # (H,W,3) [0,1]
        blended_u8 = (blended_f * 255).astype(np.uint8)

        im_blend.set_data(resize(blended_u8))

        m3       = (mask_raw.astype(np.float32) / 255.0)[:, :, None]
        nonlane  = resize((blended_f * (1.0 - m3)).clip(0, 1))
        lane_only = resize((blended_f * m3).clip(0, 1))
        im_nonlane.set_data((nonlane  * 255).astype(np.uint8) if nonlane.max() <= 1.0 else nonlane)
        im_lane.set_data((lane_only * 255).astype(np.uint8) if lane_only.max() <= 1.0 else lane_only)

        # overlay: green tint on lane pixels, dims with alpha
        ov      = orig_d.copy().astype(np.float32)
        tint    = int(60 * (1.0 - alpha))
        ov[lane_px, 1] = np.clip(ov[lane_px, 1] + tint, 0, 255)
        im_overlay.set_data(ov.astype(np.uint8))

        phase = ("fully masked (α=0)" if alpha < 0.01 else
                 "full RGB (α=1)"     if alpha > 0.99 else
                 "curriculum blend")
        axes[2].set_title(f"Model Input — {phase}", **lbl)

        info_text.set_text(
            f"alpha = {alpha:.3f}\n\n"
            f"lane coverage  : {lane_frac*100:.1f}%\n"
            f"non-lane supp. : {(1-alpha)*(1-lane_frac)*100:.1f}%\n"
            f"mean brightness: {blended_f.mean():.3f}\n\n"
            f"threshold : {args.threshold}\n"
            f"frame     : {frame_idx}"
        )
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0.0)
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",       default="interactive",
                    choices=["interactive", "poster"])
    ap.add_argument("--npz",        default=None)
    ap.add_argument("--frame",      type=int, default=0,
                    help="frame index (interactive mode)")
    ap.add_argument("--n_frames",   type=int, default=6,
                    help="number of frames to show (poster mode)")
    ap.add_argument("--threshold",  type=int, default=180,
                    help="brightness threshold for lane detection")
    ap.add_argument("--image_size", type=int, default=None,
                    help="resize output to this square (interactive mode)")
    ap.add_argument("--out",        default="visualizations/lane_mask_poster.png",
                    help="output path (poster mode)")
    args = ap.parse_args()

    if args.npz is None:
        args.npz = _auto_find_npz()
        if args.npz is None:
            print("[ERROR] No .npz file found. Pass --npz <path> explicitly.")
            sys.exit(1)
        print(f"[INFO] Auto-selected: {args.npz}")

    if args.mode == "poster":
        run_poster(args)
    else:
        run_interactive(args)


if __name__ == "__main__":
    main()
