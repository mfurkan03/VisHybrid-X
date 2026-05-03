"""
visualize_test_predictions.py – inspect model predictions vs ground truth on test samples.

Usage
-----
python src/visualize_test_predictions.py \
    --pred_dir  data/processed/dpt_pred \
    --model_path models/policy_model.pth \
    --image_size 84 \
    --num_samples 20 \
    --output visualizations/test_predictions.png
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from models import DrivingPolicyNet, DepthEstimationModel
from policy.datasets import PrecomputedDepthDataset, MetaDriveRGBDataset
from policy.trainer import apply_lane_mask, extract_features_frozen


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_model(model_path, image_size, device):
    model = DrivingPolicyNet(image_size=image_size).to(device)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict):
        key = "model" if "model" in ckpt else ("policy" if "policy" in ckpt else None)
        model.load_state_dict(ckpt[key] if key else ckpt)
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"[INFO] Loaded DrivingPolicyNet from {model_path}")
    return model


def _action_bar(ax, pred, gt, title=""):
    """Draw a horizontal bar chart comparing predicted vs ground-truth actions."""
    labels = ["Steer", "Accel/Brake"]
    x      = np.arange(len(labels))
    width  = 0.3

    ax.barh(x - width / 2, pred, width, label="Pred",  color="#e07b54", alpha=0.85)
    ax.barh(x + width / 2, gt,   width, label="GT",    color="#5b8db8", alpha=0.85)

    ax.set_xlim(-1.1, 1.1)
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=7, loc="upper right")
    steer_err  = abs(pred[0] - gt[0])
    accel_err  = abs(pred[1] - gt[1])
    ax.set_title(
        f"{title}\nSteer err {steer_err:.3f} | Accel err {accel_err:.3f}",
        fontsize=7,
    )


def _to_uint8(arr):
    """Normalise a float array to [0, 255] uint8."""
    a = arr - arr.min()
    mx = a.max()
    if mx > 0:
        a = a / mx
    return (a * 255).astype(np.uint8)


# ── main visualisation ────────────────────────────────────────────────────────

def visualize(
    pred_dir:    str,
    data_dir:    str,
    model_path:  str,
    dpt_path:    str,
    image_size:  int,
    num_samples: int,
    output_path: str,
    seed:        int = 0,
):
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _load_model(model_path, image_size, device)

    # ── choose data source ──────────────────────────────────────────────────
    use_precomputed = (
        pred_dir is not None
        and os.path.isdir(os.path.join(pred_dir, "test"))
    )

    if use_precomputed:
        print("[INFO] Loading precomputed test split …")
        test_ds = PrecomputedDepthDataset(pred_dir=pred_dir, split="test")
    else:
        print("[INFO] Precomputed test split not found — using raw RGB + live DPT …")
        depth_estimator = DepthEstimationModel(finetuned_path=dpt_path)
        test_ds = MetaDriveRGBDataset(data_dir=data_dir, split="test")

    total = len(test_ds)
    if total == 0:
        print("[ERROR] Test dataset is empty.")
        return

    indices = random.sample(range(total), min(num_samples, total))
    print(f"[INFO] Visualising {len(indices)} samples from {total} total.")

    # ── collect predictions ─────────────────────────────────────────────────
    records = []
    with torch.no_grad():
        for idx in indices:
            item = test_ds[idx]

            if use_precomputed:
                depth_np, rgb_np, action_np, ego_np = item
                depth_t  = torch.tensor(depth_np[None], dtype=torch.float32, device=device)
                rgb_arr  = rgb_np[np.newaxis]           # (1, H, W, 3)
                combined = apply_lane_mask(
                    depth_t, rgb_arr, device, image_size=image_size
                )
                ego_t = torch.tensor(ego_np, dtype=torch.float32, device=device).unsqueeze(0)
                orig_rgb = rgb_np
                depth_vis = depth_t[0, 0].cpu().numpy()
            else:
                rgb_np, action_np, ego_np = item
                rgb_arr  = rgb_np[np.newaxis]
                combined, _ = extract_features_frozen(
                    rgb_arr, depth_estimator, device, image_size=image_size
                )
                ego_t    = torch.tensor(ego_np, dtype=torch.float32, device=device).unsqueeze(0)
                orig_rgb = rgb_np
                depth_vis = combined[0, 0].cpu().numpy()

            pred = model(combined, ego_t).cpu().numpy()[0]
            records.append({
                "idx":       idx,
                "orig_rgb":  orig_rgb,          # (H, W, 3) uint8
                "depth_vis": depth_vis,          # (H, W) float
                "combined":  combined[0].cpu().numpy(),  # (4, h, w) processed
                "pred":      pred,
                "gt":        np.array(action_np, dtype=np.float32),
            })

    # ── build figure ────────────────────────────────────────────────────────
    n   = len(records)
    cols = 4      # orig RGB | depth | model input (blended) | action bar
    fig = plt.figure(figsize=(cols * 3.5, n * 3.0))
    outer = gridspec.GridSpec(n, cols, figure=fig, hspace=0.4, wspace=0.3)

    for row, rec in enumerate(records):
        # Col 0: original RGB
        ax0 = fig.add_subplot(outer[row, 0])
        ax0.imshow(rec["orig_rgb"])
        ax0.axis("off")
        if row == 0:
            ax0.set_title("Original RGB", fontsize=9, fontweight="bold")

        # Col 1: precomputed depth map (colorised)
        ax1 = fig.add_subplot(outer[row, 1])
        depth_show = _to_uint8(rec["depth_vis"])
        ax1.imshow(depth_show, cmap="inferno")
        ax1.axis("off")
        if row == 0:
            ax1.set_title("Depth (precomputed)", fontsize=9, fontweight="bold")

        # Col 2: blended input fed to the model (RGB channels after lane mask)
        ax2 = fig.add_subplot(outer[row, 2])
        blended = rec["combined"][1:4]           # (3, h, w) in [0,1]
        blended = np.transpose(blended, (1, 2, 0))
        blended = np.clip(blended, 0, 1)
        ax2.imshow(blended)
        ax2.axis("off")
        if row == 0:
            ax2.set_title("Model input (blended)", fontsize=9, fontweight="bold")

        # Col 3: prediction vs GT bar
        ax3 = fig.add_subplot(outer[row, 3])
        _action_bar(
            ax3, rec["pred"], rec["gt"],
            title=f"Sample #{rec['idx']}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved → {output_path}")

    # ── print summary stats ─────────────────────────────────────────────────
    preds = np.stack([r["pred"] for r in records])
    gts   = np.stack([r["gt"]   for r in records])
    steer_mae = float(np.mean(np.abs(preds[:, 0] - gts[:, 0])))
    accel_mae = float(np.mean(np.abs(preds[:, 1] - gts[:, 1])))
    steer_dir = float(np.mean(np.sign(preds[:, 0]) == np.sign(gts[:, 0])))
    true_brake = gts[:, 1] < -0.05
    brake_acc  = (
        float(np.mean(preds[true_brake, 1] < -0.05)) if np.any(true_brake) else float("nan")
    )
    print(f"\n── Sample metrics ({len(records)} frames) ─────────────────")
    print(f"  Steer MAE   : {steer_mae:.4f}")
    print(f"  Accel MAE   : {accel_mae:.4f}")
    print(f"  Steer DirAcc: {steer_dir*100:.1f}%")
    print(f"  Brake Acc   : {brake_acc*100:.1f}%" if not np.isnan(brake_acc)
          else "  Brake Acc   : N/A (no braking GT in sample)")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise model predictions on test samples")
    parser.add_argument("--pred_dir",    type=str, default=None,
                        help="Precomputed depth dir. Falls back to live DPT if test split absent.")
    parser.add_argument("--data_dir",    type=str, default="dataset",
                        help="Raw dataset dir (used when pred_dir unavailable).")
    parser.add_argument("--model_path",  type=str, required=True)
    parser.add_argument("--dpt_path",    type=str, default="models/dpt_finetuned.pth",
                        help="DPT checkpoint (only needed when falling back to live inference).")
    parser.add_argument("--image_size",  type=int, default=84)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--output",      type=str, default="visualizations/test_predictions.png")
    parser.add_argument("--seed",        type=int, default=0)
    args = parser.parse_args()

    visualize(
        pred_dir    = args.pred_dir,
        data_dir    = args.data_dir,
        model_path  = args.model_path,
        dpt_path    = args.dpt_path,
        image_size  = args.image_size,
        num_samples = args.num_samples,
        output_path = args.output,
        seed        = args.seed,
    )
