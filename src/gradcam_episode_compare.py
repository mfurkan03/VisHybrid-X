import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.gradcam_demo import (
    DrivingPolicyNet,
    create_depth_estimator,
    load_rgb_and_combined_from_npz,
    run_gradcam_for_rgb,
    save_gradcam_outputs,
    save_rgb,
)


TARGETS = ["steer", "accel"]


def parse_args():
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description="Generate steer/accel Grad-CAM outputs for every frame in one episode."
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--depth_checkpoint_path", default=None)
    parser.add_argument("--recompute_from_rgb", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None)
    return parser.parse_args()


def make_comparison_rgb(rgb_image, target_results):
    tiles = []
    labels = []
    for target in TARGETS:
        labels.append(
            [
                target.upper(),
                f"steer={target_results[target]['pred_np'][0]:.3f}",
                f"accel={target_results[target]['pred_np'][1]:.3f}",
            ]
        )
        tiles.append(target_results[target]["cam_rgb_overlay"])

    pad = 24
    label_h = 100
    tile_h, tile_w = tiles[0].shape[:2]
    canvas_h = tile_h + label_h + pad * 2
    canvas_w = tile_w * len(tiles) + pad * (len(tiles) + 1)
    canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

    for idx, (label_lines, tile) in enumerate(zip(labels, tiles)):
        x0 = pad + idx * (tile_w + pad)
        y0 = label_h + pad
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile
        cv2.putText(canvas, label_lines[0], (x0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, label_lines[1], (x0, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (50, 50, 50), 1, cv2.LINE_AA)
        cv2.putText(canvas, label_lines[2], (x0, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (50, 50, 50), 1, cv2.LINE_AA)

    return canvas


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_root, exist_ok=True)

    model = DrivingPolicyNet().to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    depth_estimator = None
    if args.recompute_from_rgb:
        depth_estimator = create_depth_estimator(device, args.depth_checkpoint_path)

    npz_data = np.load(args.npz_path, allow_pickle=True)
    if "cam_0_rgb" in npz_data.files:
        num_frames = len(npz_data["cam_0_rgb"])
    elif "rgb" in npz_data.files:
        num_frames = len(npz_data["rgb"])
    else:
        rgb_keys = [k for k in npz_data.files if "rgb" in k.lower()]
        if not rgb_keys:
            raise KeyError(f"No RGB key found in {args.npz_path}. Keys: {npz_data.files}")
        num_frames = len(npz_data[rgb_keys[0]])

    if args.max_frames is not None:
        num_frames = min(num_frames, args.max_frames)

    summary_rows = []

    for frame_index in range(num_frames):
        rgb_image, rgb_key, combined_frame, combined_key = load_rgb_and_combined_from_npz(
            args.npz_path, frame_index
        )
        if args.recompute_from_rgb:
            combined_frame = None
            input_source = (
                f"{args.npz_path} [rgb_key={rgb_key}, frame_index={frame_index}, "
                f"recompute_from_rgb=True]"
            )
        else:
            input_source = (
                f"{args.npz_path} [rgb_key={rgb_key}, combined_key={combined_key}, "
                f"frame_index={frame_index}]"
            )

        frame_dir = os.path.join(args.output_root, f"frame_{frame_index:04d}")
        os.makedirs(frame_dir, exist_ok=True)

        target_results = {}
        for target in TARGETS:
            result = run_gradcam_for_rgb(
                model=model,
                rgb_image=rgb_image,
                input_source=input_source,
                target=target,
                device=device,
                combined_frame=combined_frame,
                depth_estimator=depth_estimator,
            )
            target_results[target] = result
            save_gradcam_outputs(
                os.path.join(frame_dir, target),
                args.checkpoint_path,
                result,
            )

        comparison_rgb = make_comparison_rgb(rgb_image, target_results)
        save_rgb(os.path.join(frame_dir, "comparison_rgb.png"), comparison_rgb)

        summary_rows.append(
            {
                "frame_index": frame_index,
                "steer_pred": float(target_results["steer"]["pred_np"][0]),
                "accel_pred": float(target_results["accel"]["pred_np"][1]),
                "frame_dir": frame_dir,
            }
        )

        print(f"Processed frame {frame_index + 1}/{num_frames}: {frame_dir}")

    summary_path = os.path.join(args.output_root, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_index", "steer_pred", "accel_pred", "frame_dir"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved episode outputs to: {args.output_root}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
