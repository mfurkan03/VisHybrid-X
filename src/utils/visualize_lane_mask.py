"""
Script to verify lane mask (white + yellow) on dataset frames.
Run from the repo root:
    python src/visualize_lane_mask.py --data_dir data/raw/mixed --n_frames 6

Keys while viewing: any key = next image, Q = quit.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from policy.trainer import _batch_lane_mask


def rgb_np_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 → (1, 3, H, W) float [0,1]."""
    t = torch.from_numpy(rgb).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0)


def apply_mask_alpha0(rgb_tensor: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Apply mask with alpha=0 (fully masked): only keep masked pixels."""
    masked = rgb_tensor * mask
    img = masked[0].permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def white_only_mask(rgb_tensor: torch.Tensor, threshold: float = 180 / 255.0) -> torch.Tensor:
    """Original mask: white pixels only."""
    _, _, H, _ = rgb_tensor.shape
    gray = 0.299 * rgb_tensor[:, 0] + 0.587 * rgb_tensor[:, 1] + 0.114 * rgb_tensor[:, 2]
    gray[:, :int(H * 0.55), :] = 0.0
    return (gray >= threshold).float().unsqueeze(1)


def main(data_dir: str, n_frames: int, split: str):
    split_dir = os.path.join(data_dir, split)
    files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
    if not files:
        print(f"No .npz files found in {split_dir}")
        return

    frames_shown = 0
    for fpath in files:
        if frames_shown >= n_frames:
            break
        data = np.load(fpath, allow_pickle=True)
        rgb_keys = [k for k in data.files if k.endswith("_rgb")]
        if not rgb_keys:
            continue
        rgb_frames = data[rgb_keys[0]]  # (N, H, W, 3) uint8

        # Pick a few evenly-spaced frames from this episode
        indices = np.linspace(0, len(rgb_frames) - 1, min(3, len(rgb_frames)), dtype=int)
        for idx in indices:
            if frames_shown >= n_frames:
                break
            rgb = rgb_frames[idx]  # (H, W, 3) uint8
            t   = rgb_np_to_tensor(rgb)

            white_m    = white_only_mask(t)
            combined_m = _batch_lane_mask(t)

            masked_white    = apply_mask_alpha0(t, white_m)
            masked_combined = apply_mask_alpha0(t, combined_m)

            # Yellow-only pixels (combined minus white)
            yellow_only = ((combined_m - white_m).clamp(0, 1))
            yellow_highlight = rgb.copy()
            yellow_pixels = yellow_only[0, 0].numpy().astype(bool)
            yellow_highlight[yellow_pixels] = [0, 200, 255]  # cyan overlay for visibility

            # Print pixel stats for yellow region
            n_yellow = yellow_pixels.sum()
            n_white  = white_m[0, 0].numpy().astype(bool).sum()
            print(f"[{os.path.basename(fpath)} frame {idx}] "
                  f"white pixels: {n_white}  yellow pixels: {n_yellow}")

            # Build display grid: original | white mask | white+yellow mask | yellow highlight
            h, w = rgb.shape[:2]
            scale = max(1, 400 // h)
            def resize(img):
                return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

            row = np.hstack([
                resize(cv2.cvtColor(rgb,              cv2.COLOR_RGB2BGR)),
                resize(cv2.cvtColor(masked_white,     cv2.COLOR_RGB2BGR)),
                resize(cv2.cvtColor(masked_combined,  cv2.COLOR_RGB2BGR)),
                resize(cv2.cvtColor(yellow_highlight, cv2.COLOR_RGB2BGR)),
            ])

            # Labels
            label_h = 24
            label_row = np.zeros((label_h, row.shape[1], 3), dtype=np.uint8)
            for i, txt in enumerate(["Original", "White only", "White+Yellow", "Yellow pixels"]):
                cv2.putText(label_row, txt,
                            (i * w * scale + 4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            display = np.vstack([label_row, row])
            cv2.imshow("Lane Mask Verification  (any key = next, Q = quit)", display)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                return
            frames_shown += 1

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  type=str, default="data/raw/mixed")
    parser.add_argument("--split",     type=str, default="train")
    parser.add_argument("--n_frames",  type=int, default=6)
    args = parser.parse_args()
    main(args.data_dir, args.n_frames, args.split)
