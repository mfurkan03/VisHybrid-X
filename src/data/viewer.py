"""
viewer.py – OpenCV camera visualisation with FPS overlay.
"""

import cv2
import numpy as np

from utils.fps import FPSCounter


def show_cameras(
    raw_frames: dict,
    rgb_cam_names: list,
    depth_cam_names: list,
    fps_counter: FPSCounter,
) -> bool:
    """
    Render a grid of RGB + depth thumbnails with an FPS overlay.

    Returns True if the user pressed 'Q' (quit requested).
    """
    THUMB = 280
    rgb_frames   = []
    depth_frames = []

    for cam_name, depth_name in zip(rgb_cam_names, depth_cam_names):
        rgb_raw, depth_raw = raw_frames[cam_name]

        img = rgb_raw
        if hasattr(img, "get"):
            img = img.get()
        img     = np.array(img, dtype=np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_big = cv2.resize(img_bgr, (THUMB, THUMB), interpolation=cv2.INTER_NEAREST)
        angle   = cam_name.split("_")[1]
        cv2.putText(img_big, f"RGB {angle}deg", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        rgb_frames.append(img_big)

        d = depth_raw
        if hasattr(d, "get"):
            d = d.get()
        d = np.array(d, dtype=np.float32)
        if d.ndim == 3:
            d = d[:, :, 0]
        d_norm  = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
        d_big   = cv2.resize(d_color, (THUMB, THUMB), interpolation=cv2.INTER_NEAREST)
        angle   = depth_name.split("_")[1]
        cv2.putText(d_big, f"DEPTH {angle}deg", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        depth_frames.append(d_big)

    rgb_row   = np.hstack(rgb_frames)
    depth_row = np.hstack(depth_frames)
    combined  = np.vstack([rgb_row, depth_row])

    fps_text = (f"FPS: {fps_counter.instant_fps:5.1f}  |  "
                f"Avg: {fps_counter.average_fps:5.1f}  |  "
                f"Steps: {fps_counter.total_steps}")
    cv2.rectangle(combined, (0, 0), (len(fps_text) * 11 + 10, 30), (0, 0, 0), -1)
    cv2.putText(combined, fps_text, (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("MetaDrive Cameras  (Q = Quit)", combined)
    return (cv2.waitKey(1) & 0xFF) == ord("q")