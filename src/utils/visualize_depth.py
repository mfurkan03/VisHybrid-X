"""
Side-by-side visualisation of GT depth vs DPT predictions.

Two modes:
  --mode poster       Static multi-frame grid saved to PNG.
                      Columns: RGB | GT Depth | Predicted Depth | Abs Error

  --mode interactive  (default) Matplotlib window with a frame slider.
                      Shows all 4 panels and live error stats.

Supported input combos
----------------------
A) Raw npz  +  DPT model  (runs inference on the fly)
       --npz dataset/train/ep0.npz --model_path models/dpt_finetuned.pth

B) Raw npz  +  precomputed npz  (no GPU needed)
       --npz dataset/train/ep0.npz --pred_npz data/processed/dpt_pred/train/ep0.npz

C) Precomputed npz only  (no GT — GT panel shown as "N/A")
       --npz data/processed/dpt_pred/train/ep0.npz

Usage (from repo root):
    python src/utils/visualize_depth.py \\
        --npz dataset/train/episode_0.npz \\
        --model_path models/dpt_finetuned.pth \\
        --n_frames 6 --out visualizations/depth_vis.png
"""

import argparse
import glob
import pathlib
import sys

import cv2
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


# ── helpers ────────────────────────────────────────────────────────────────────

def _depth_colormap(depth_hw: np.ndarray, vmin: float | None = None,
                    vmax: float | None = None) -> np.ndarray:
    """(H, W) float → (H, W, 3) uint8 using inferno colormap."""
    d = depth_hw.squeeze().astype(np.float32)
    lo = d.min() if vmin is None else vmin
    hi = d.max() if vmax is None else vmax
    d = (d - lo) / (hi - lo + 1e-8)
    d = d.clip(0, 1)
    return (cm.inferno(d)[:, :, :3] * 255).astype(np.uint8)


def _error_colormap(abs_err: np.ndarray) -> np.ndarray:
    """(H, W) abs error → (H, W, 3) uint8 using hot colormap."""
    d = abs_err.squeeze().astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return (cm.hot(d)[:, :, :3] * 255).astype(np.uint8)


def _load_raw_npz(path: str):
    """Load a raw dataset npz. Returns (rgb_all, gt_depth_all or None)."""
    data = np.load(path, allow_pickle=True)
    rgb_keys   = [k for k in data.files if k.endswith("_rgb") or k == "rgb"]
    depth_keys = [k for k in data.files if k.endswith("_depth") and "pred" not in k]
    if not rgb_keys:
        raise ValueError(f"No RGB key found in {path}. Keys: {list(data.files)}")
    rgb_all = data[rgb_keys[0]]
    gt_depth_all = None
    if depth_keys:
        d = data[depth_keys[0]]
        # normalise to (N, H, W)
        if d.ndim == 4:
            d = d[:, 0] if d.shape[1] == 1 else d[..., 0]
        gt_depth_all = d.astype(np.float32)
    return rgb_all, gt_depth_all


def _load_pred_npz(path: str):
    """Load a precomputed npz. Returns depth_pred_all (N, H, W)."""
    data = np.load(path, allow_pickle=True)
    pred_keys = [k for k in data.files if "pred" in k or "depth" in k]
    if not pred_keys:
        raise ValueError(f"No depth key in {path}. Keys: {list(data.files)}")
    d = data[pred_keys[0]].astype(np.float32)
    if d.ndim == 4:
        d = d[:, 0] if d.shape[1] == 1 else d[..., 0]
    return d


def _run_model_inference(rgb_all: np.ndarray, model_path: str) -> np.ndarray:
    """Run DPT model on rgb_all, return (N, H, W) float predictions (inverted depth)."""
    import torch
    from models import DepthEstimationModel

    # DepthEstimationModel picks device automatically; load checkpoint via finetuned_path
    model = DepthEstimationModel(finetuned_path=model_path)
    model.set_eval_mode()

    preds = []
    batch_size = 4
    for start in range(0, len(rgb_all), batch_size):
        batch = rgb_all[start:start + batch_size]
        with torch.no_grad():
            raw = model.predict_batch_with_grad(batch)  # (B, 1, H, W)
        raw_np = raw.squeeze(1).cpu().numpy()           # (B, H, W)
        # invert + normalise per frame, matching env_wrapper convention
        for frame in raw_np:
            lo, hi = frame.min(), frame.max()
            norm   = (frame - lo) / (hi - lo + 1e-8)
            preds.append(1.0 - norm)

    return np.stack(preds, axis=0)


def _auto_find_npz(prefer_raw: bool = True) -> str | None:
    roots_raw  = ["dataset", "data/raw"]
    roots_pred = ["data/processed/dpt_pred"]
    roots      = (roots_raw + roots_pred) if prefer_raw else (roots_pred + roots_raw)
    for root in roots:
        hits = glob.glob(f"{root}/**/*.npz", recursive=True)
        if hits:
            return hits[0]
    return None


def _resize_hw3(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


# ── poster mode ───────────────────────────────────────────────────────────────

def run_poster(rgb_all, gt_all, pred_all, args):
    n       = min(args.n_frames, len(rgb_all))
    indices = np.linspace(0, len(rgb_all) - 1, n, dtype=int)
    has_gt  = gt_all is not None

    cols  = ["RGB", "GT Depth", "Predicted Depth", "Abs Error"] if has_gt \
            else ["RGB", "Predicted Depth"]
    ncols = len(cols)

    fig, axes = plt.subplots(n, ncols, figsize=(5 * ncols, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Depth Prediction Visualisation", fontsize=15, y=1.01)
    for col, title in enumerate(cols):
        axes[0, col].set_title(title, fontsize=12, pad=8)

    for row, i in enumerate(indices):
        rgb  = rgb_all[i].astype(np.uint8) if rgb_all[i].max() > 1 \
               else (rgb_all[i] * 255).astype(np.uint8)
        pred = pred_all[i]

        if has_gt:
            gt        = gt_all[i]
            abs_err   = np.abs(gt - pred)
            mae       = abs_err.mean()
            gt_col    = _depth_colormap(gt)
            pred_col  = _depth_colormap(pred)
            err_col   = _error_colormap(abs_err)
            panels = [(rgb, {}), (gt_col, {}), (pred_col, {}), (err_col, {})]
            axes[row, 1].set_xlabel(f"MAE={mae:.4f}", fontsize=8)
        else:
            pred_col = _depth_colormap(pred)
            panels   = [(rgb, {}), (pred_col, {})]

        for col, (img, kw) in enumerate(panels):
            axes[row, col].imshow(img, **kw)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(f"frame {i}", fontsize=9)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=1.5)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved → {args.out}")


# ── interactive mode ──────────────────────────────────────────────────────────

def run_interactive(rgb_all, gt_all, pred_all, args):
    from matplotlib.widgets import Slider

    has_gt   = gt_all is not None
    n_frames = len(rgb_all)

    BG = "#1a1a2e"
    ncols = 4 if has_gt else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 6), facecolor=BG)
    plt.subplots_adjust(bottom=0.18, top=0.88, left=0.03, right=0.97, wspace=0.06)

    lbl = dict(color="white", fontsize=10, pad=6)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")

    def _get_frame(idx):
        rgb  = rgb_all[idx].astype(np.uint8) if rgb_all[idx].max() > 1 \
               else (rgb_all[idx] * 255).astype(np.uint8)
        pred = pred_all[idx]
        gt   = gt_all[idx] if has_gt else None
        return rgb, gt, pred

    rgb0, gt0, pred0 = _get_frame(0)
    im_rgb  = axes[0].imshow(rgb0)
    axes[0].set_title("RGB", **lbl)

    col = 1
    if has_gt:
        im_gt = axes[col].imshow(_depth_colormap(gt0))
        axes[col].set_title("GT Depth", **lbl)
        col += 1

    im_pred = axes[col].imshow(_depth_colormap(pred0))
    axes[col].set_title("Predicted Depth", **lbl)
    col += 1

    if has_gt:
        im_err = axes[col].imshow(_error_colormap(np.abs(gt0 - pred0)))
        axes[col].set_title("Abs Error", **lbl)

    stats_ax = axes[-1] if not has_gt else None

    fig.suptitle(f"Depth Visualiser  |  {pathlib.Path(args.npz).name}",
                 color="white", fontsize=11)

    ax_sl  = fig.add_axes([0.15, 0.06, 0.70, 0.03], facecolor="#2a2a4a")
    slider = Slider(ax_sl, "frame", 0, n_frames - 1, valinit=0,
                    valstep=1, color="#5555cc")
    slider.label.set_color("white")
    slider.valtext.set_color("white")

    title_text = fig.text(0.5, 0.02, "", ha="center", color="#aaaacc", fontsize=9)

    def update(val):
        idx = int(slider.val)
        rgb, gt, pred = _get_frame(idx)
        im_rgb.set_data(rgb)
        im_pred.set_data(_depth_colormap(pred))
        info = f"frame {idx}/{n_frames-1}  |  pred range [{pred.min():.3f}, {pred.max():.3f}]"
        if has_gt:
            err = np.abs(gt - pred)
            im_gt.set_data(_depth_colormap(gt))
            im_err.set_data(_error_colormap(err))
            info += f"  |  MAE={err.mean():.4f}  RMSE={np.sqrt((err**2).mean()):.4f}"
        title_text.set_text(info)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",       default="interactive",
                    choices=["interactive", "poster"])
    ap.add_argument("--npz",        default=None,
                    help="Raw npz (with *_rgb/*_depth) or precomputed npz")
    ap.add_argument("--pred_npz",   default=None,
                    help="Precomputed npz (depth_pred key) — use instead of model inference")
    ap.add_argument("--model_path", default=None,
                    help="DPT checkpoint to run on the fly (combo A)")
    ap.add_argument("--n_frames",   type=int, default=6,
                    help="Frames to show in poster mode")
    ap.add_argument("--out",        default="visualizations/depth_vis.png",
                    help="Output path for poster mode")
    args = ap.parse_args()

    # ── resolve input file ────────────────────────────────────────────────────
    if args.npz is None:
        args.npz = _auto_find_npz(prefer_raw=True)
        if args.npz is None:
            print("[ERROR] No .npz file found. Pass --npz <path>.")
            sys.exit(1)
        print(f"[INFO] Auto-selected: {args.npz}")

    rgb_all, gt_all = _load_raw_npz(args.npz)
    print(f"[INFO] Loaded {len(rgb_all)} frames from {args.npz}")
    if gt_all is not None:
        print(f"[INFO] GT depth found  shape={gt_all.shape}")
    else:
        print("[INFO] No GT depth in this file.")

    # ── get predictions ───────────────────────────────────────────────────────
    pred_all = None

    if args.pred_npz:
        pred_all = _load_pred_npz(args.pred_npz)
        print(f"[INFO] Loaded predictions from {args.pred_npz}  shape={pred_all.shape}")
    elif args.model_path:
        print(f"[INFO] Running DPT inference on {len(rgb_all)} frames …")
        pred_all = _run_model_inference(rgb_all, args.model_path)
        print(f"[INFO] Inference done  shape={pred_all.shape}")
    else:
        # try the depth_pred key in the same file (precomputed format)
        data = np.load(args.npz, allow_pickle=True)
        pred_keys = [k for k in data.files if "pred" in k]
        if pred_keys:
            d = data[pred_keys[0]].astype(np.float32)
            if d.ndim == 4:
                d = d[:, 0] if d.shape[1] == 1 else d[..., 0]
            pred_all = d
            print(f"[INFO] Found predictions in same file (key='{pred_keys[0]}')  shape={pred_all.shape}")
        else:
            print("[WARNING] No predictions found. Pass --model_path or --pred_npz.")
            print("          Showing RGB + GT depth only.")
            # synthesise dummy predictions so code paths still work
            if gt_all is not None:
                pred_all = gt_all.copy()
            else:
                print("[ERROR] Nothing to visualise.")
                sys.exit(1)

    # ── align lengths ─────────────────────────────────────────────────────────
    n = min(len(rgb_all), len(pred_all))
    if gt_all is not None:
        n = min(n, len(gt_all))
        gt_all = gt_all[:n]
    rgb_all  = rgb_all[:n]
    pred_all = pred_all[:n]

    # ── dispatch ──────────────────────────────────────────────────────────────
    if args.mode == "poster":
        run_poster(rgb_all, gt_all, pred_all, args)
    else:
        run_interactive(rgb_all, gt_all, pred_all, args)


if __name__ == "__main__":
    main()
