"""
Side-by-side visualisation of RGB | Precomputed Depth | Lane Mask.

The precomputed .npz already contains both 'rgb' and 'depth_pred' keys.

Usage (from repo root):
    python src/utils/visualize_pipeline.py \
        --npz data/processed/dpt_pred/train/episode_0.npz \
        --n_frames 6 \
        --out visualizations/pipeline_vis.png
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from policy.trainer import get_lane_mask_visual


def depth_to_rgb(depth_frame: np.ndarray) -> np.ndarray:
    d = depth_frame.squeeze()
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return (cm.inferno(d)[:, :, :3] * 255).astype(np.uint8)


def lane_mask_image(rgb_uint8: np.ndarray) -> np.ndarray:
    return get_lane_mask_visual(rgb_uint8)  # (H, W) uint8 binary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz",      default="data/processed/dpt_pred/train/episode_0.npz")
    parser.add_argument("--n_frames", type=int, default=6)
    parser.add_argument("--out",      default="visualizations/pipeline_vis.png")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    rgb_all   = data["rgb"]        # (T, H, W, 3) uint8
    depth_all = data["depth_pred"] # (T, 1, H, W) float32

    n    = min(args.n_frames, len(rgb_all))
    idx  = np.linspace(0, len(rgb_all) - 1, n, dtype=int)
    cols = ["RGB", "Depth", "Lane Mask"]

    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Pipeline Visualisation", fontsize=16, y=1.01)

    for col, title in enumerate(cols):
        axes[0, col].set_title(title, fontsize=13, pad=8)

    for row, i in enumerate(idx):
        frame = rgb_all[i]
        panels = [
            (frame,                         {}),
            (depth_to_rgb(depth_all[i]),    {}),
            (lane_mask_image(frame),        {"cmap": "gray"}),
        ]
        for col, (img, kw) in enumerate(panels):
            axes[row, col].imshow(img, **kw)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(f"frame {i}", fontsize=9)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=1.5)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
