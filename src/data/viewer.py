"""
viewer.py – Poster-quality OpenCV camera visualisation with ego-state overlay.
"""

import cv2
import numpy as np

from utils.fps import FPSCounter


# ── Palette ──────────────────────────────────────────────────────────────────
BG        = (18,  18,  24)
PANEL_BG  = (30,  30,  40)
ACCENT    = (0,   210, 180)
WHITE     = (240, 240, 245)
GRAY      = (120, 120, 135)
RED_ACC   = (60,  120, 255)
BORDER    = (60,  60,  80)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# ── Layout constants (display) ────────────────────────────────────────────────
THUMB    = 520
PAD      = 14
HEADER_H = 56
FOOTER_H = 44
INFO_W   = 220

_poster_saved = False   # module-level flag — save only once per run


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_panel(canvas, x, y, w, h):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), PANEL_BG, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER, 1)


def _label(canvas, text, x, y, *, color=WHITE, scale=0.52, thickness=1, font=FONT):
    cv2.putText(canvas, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _tag(canvas, text, x, y, *, bg=ACCENT, fg=BG, scale=0.45, pad=4):
    (tw, th), baseline = cv2.getTextSize(text, FONT_BOLD, scale, 1)
    cv2.rectangle(canvas, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), bg, -1)
    cv2.putText(canvas, text, (x, y), FONT_BOLD, scale, fg, 1, cv2.LINE_AA)


def _bar(canvas, x, y, w, h, value, *, lo=-1.0, hi=1.0, bar_color=ACCENT, label=""):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), PANEL_BG, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER, 1)
    zero_x = x + int(w * (-lo) / (hi - lo))
    fill_x = x + int(w * (value - lo) / (hi - lo))
    fill_x = max(x, min(x + w, fill_x))
    zero_x = max(x, min(x + w, zero_x))
    lx, rx = (fill_x, zero_x) if fill_x < zero_x else (zero_x, fill_x)
    cv2.rectangle(canvas, (lx, y + 1), (rx, y + h - 1), bar_color, -1)
    if label:
        _label(canvas, label, x + 4, y + h - 4, color=BG if rx - lx > 40 else GRAY, scale=0.38)


# ── Canvas builder (shared by display and poster) ─────────────────────────────

def _build_canvas(
    raw_frames, rgb_cam_names, depth_cam_names, fps_counter, ego_info, thumb
):
    pad      = max(8,  int(thumb * PAD      / THUMB))
    header_h = max(40, int(thumb * HEADER_H / THUMB))
    footer_h = max(32, int(thumb * FOOTER_H / THUMB))
    info_w   = max(160, int(thumb * INFO_W  / THUMB))
    n        = len(rgb_cam_names)

    grid_w  = n * thumb + (n + 1) * pad
    grid_h  = 2 * thumb + 3 * pad
    total_w = grid_w + info_w + pad
    total_h = header_h + grid_h + footer_h

    # scale font sizes with thumb
    s   = thumb / THUMB
    fs  = lambda base: max(0.3, base * s)

    canvas = np.full((total_h, total_w, 3), BG, dtype=np.uint8)

    # header
    cv2.rectangle(canvas, (0, 0), (total_w, header_h), PANEL_BG, -1)
    cv2.line(canvas, (0, header_h - 1), (total_w, header_h - 1), BORDER, 1)
    cv2.putText(canvas, "MetaDrive", (16, int(header_h * 0.42)),
                FONT_BOLD, fs(0.65), ACCENT, max(1, int(s)), cv2.LINE_AA)
    cv2.putText(canvas, "  Expert Data Collection", (16, int(header_h * 0.78)),
                FONT, fs(0.45), GRAY, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Autonomous Driving  |  IL Pipeline",
                (total_w - int(330 * s), int(header_h * 0.65)),
                FONT, fs(0.45), GRAY, 1, cv2.LINE_AA)

    for col, (cam_name, depth_name) in enumerate(zip(rgb_cam_names, depth_cam_names)):
        rgb_raw, depth_raw = raw_frames[cam_name]
        angle = cam_name.split("_")[1] if "_" in cam_name else cam_name

        # RGB
        img = rgb_raw
        if hasattr(img, "get"):
            img = img.get()
        img     = np.array(img, dtype=np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.resize(img_bgr, (thumb, thumb), interpolation=cv2.INTER_AREA)

        rx = pad + col * (thumb + pad)
        ry = header_h + pad
        _draw_panel(canvas, rx - 2, ry - 2, thumb + 4, thumb + 4)
        canvas[ry:ry + thumb, rx:rx + thumb] = img_bgr

        tag_y  = ry + int(22 * s)
        tag_sc = fs(0.45)
        (tw, th_), bl = cv2.getTextSize(f"RGB  {angle} deg", FONT_BOLD, tag_sc, 1)
        p = max(3, int(4 * s))
        cv2.rectangle(canvas, (rx + 8 - p, tag_y - th_ - p),
                      (rx + 8 + tw + p, tag_y + bl + p), ACCENT, -1)
        cv2.putText(canvas, f"RGB  {angle} deg", (rx + 8, tag_y),
                    FONT_BOLD, tag_sc, BG, 1, cv2.LINE_AA)

        # Depth
        d = depth_raw
        if hasattr(d, "get"):
            d = d.get()
        d = np.array(d, dtype=np.float32)
        if d.ndim == 3:
            d = d[:, :, 0]
        d_norm  = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)
        d_color = cv2.resize(d_color, (thumb, thumb), interpolation=cv2.INTER_AREA)

        dy = header_h + pad + thumb + pad
        _draw_panel(canvas, rx - 2, dy - 2, thumb + 4, thumb + 4)
        canvas[dy:dy + thumb, rx:rx + thumb] = d_color

        dtag_y = dy + int(22 * s)
        (tw2, th2), bl2 = cv2.getTextSize(f"DEPTH  {angle} deg", FONT_BOLD, tag_sc, 1)
        cv2.rectangle(canvas, (rx + 8 - p, dtag_y - th2 - p),
                      (rx + 8 + tw2 + p, dtag_y + bl2 + p), RED_ACC, -1)
        cv2.putText(canvas, f"DEPTH  {angle} deg", (rx + 8, dtag_y),
                    FONT_BOLD, tag_sc, WHITE, 1, cv2.LINE_AA)

    # info panel
    ix    = grid_w + pad
    iy    = header_h + pad
    info_h = grid_h
    _draw_panel(canvas, ix, iy, info_w - pad, info_h)

    cv2.putText(canvas, "EGO STATE", (ix + 12, iy + int(24 * s)),
                FONT_BOLD, fs(0.5), ACCENT, max(1, int(s)), cv2.LINE_AA)
    cv2.line(canvas, (ix + 8, iy + int(30 * s)),
             (ix + info_w - pad - 8, iy + int(30 * s)), BORDER, 1)

    ego         = ego_info or {}
    speed       = float(ego.get("speed",         0.0))
    steer       = float(ego.get("steer",         0.0))
    heading     = float(ego.get("heading_delta", 0.0))
    instant_fps = fps_counter.instant_fps
    avg_fps     = fps_counter.average_fps
    steps       = fps_counter.total_steps

    rows = [
        ("Speed",    f"{speed:+.2f} m/s",  speed / 30.0,  ACCENT),
        ("Steering", f"{steer:+.3f}",       steer,         (80, 180, 255)),
        ("Heading",  f"{heading:+.3f} rad", heading / 1.5, (180, 140, 255)),
    ]

    bar_w   = info_w - pad - 24
    row_gap = int(70 * s)
    for i, (name, value_str, norm, col) in enumerate(rows):
        base_y = iy + int(50 * s) + i * row_gap
        cv2.putText(canvas, name, (ix + 12, base_y),
                    FONT, fs(0.44), GRAY, 1, cv2.LINE_AA)
        cv2.putText(canvas, value_str, (ix + 12, base_y + int(20 * s)),
                    FONT_BOLD, fs(0.55), WHITE, max(1, int(s)), cv2.LINE_AA)
        bar_h = max(8, int(14 * s))
        _bar(canvas, ix + 12, base_y + int(28 * s), bar_w, bar_h,
             float(np.clip(norm, -1, 1)), bar_color=col)

    div_y = iy + int(50 * s) + len(rows) * row_gap + int(10 * s)
    cv2.line(canvas, (ix + 8, div_y), (ix + info_w - pad - 8, div_y), BORDER, 1)
    cv2.putText(canvas, "STEPS", (ix + 12, div_y + int(22 * s)),
                FONT, fs(0.44), GRAY, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{steps:,}", (ix + 12, div_y + int(42 * s)),
                FONT_BOLD, fs(0.65), WHITE, max(1, int(s)), cv2.LINE_AA)

    # footer
    fy = header_h + grid_h
    cv2.rectangle(canvas, (0, fy), (total_w, total_h), PANEL_BG, -1)
    cv2.line(canvas, (0, fy), (total_w, fy), BORDER, 1)
    cv2.putText(canvas, f"FPS  {instant_fps:5.1f}",
                (pad, fy + int(28 * s)), FONT_BOLD, fs(0.55), ACCENT, max(1, int(s)), cv2.LINE_AA)
    cv2.putText(canvas, f"avg {avg_fps:5.1f}",
                (pad + int(130 * s), fy + int(28 * s)), FONT, fs(0.5), GRAY, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"steps  {steps:,}",
                (pad + int(290 * s), fy + int(28 * s)), FONT, fs(0.5), GRAY, 1, cv2.LINE_AA)

    return canvas


# ── Public API ────────────────────────────────────────────────────────────────

def show_cameras(
    raw_frames: dict,
    rgb_cam_names: list,
    depth_cam_names: list,
    fps_counter: FPSCounter,
    ego_info: dict | None = None,
    poster_path: str | None = None,
    poster_after_steps: int = 30,
) -> bool:
    """
    Render and display the camera grid.

    If poster_path is set, a high-res PNG is saved automatically once the
    step counter reaches poster_after_steps (default 30), then collection stops.

    Returns True if quit was requested (poster saved, or user pressed Q).
    """
    global _poster_saved

    canvas = _build_canvas(
        raw_frames, rgb_cam_names, depth_cam_names, fps_counter, ego_info, THUMB
    )

    if poster_path and not _poster_saved and fps_counter.total_steps >= poster_after_steps:
        import os
        if not any(poster_path.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".bmp")):
            poster_path = poster_path + ".png"
        os.makedirs(os.path.dirname(os.path.abspath(poster_path)), exist_ok=True)
        poster_canvas = _build_canvas(
            raw_frames, rgb_cam_names, depth_cam_names, fps_counter, ego_info,
            thumb=800,
        )
        cv2.imwrite(poster_path, poster_canvas)
        print(f"\n  [Poster] Saved high-res frame to '{poster_path}'")
        _poster_saved = True
        return True  # stop collection

    cv2.imshow("MetaDrive  |  Expert Data Collection  (Q = quit)", canvas)
    return (cv2.waitKey(1) & 0xFF) == ord("q")
