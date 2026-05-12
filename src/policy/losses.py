"""
policy/losses.py – loss functions and offline evaluation metrics.
"""

import numpy as np
import torch
from scipy.stats import pearsonr


import torch
import torch.nn as nn

def custom_driving_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    smooth_l1    = nn.functional.smooth_l1_loss(pred, target, reduction='none')
    weighted_loss = smooth_l1.clone()
    # proportional weight: 1.0 at steer=0, 2.5 at steer=1.0 — harder turns penalised more
    turn_weight  = 1.0 + 1.5 * target[:, 0].abs()
    weighted_loss[:, 0] = smooth_l1[:, 0] * turn_weight
    # 3x weight on braking events
    brake_weight = 1.0 + (target[:, 1] < -0.1).float() * 1.0
    weighted_loss[:, 1] = smooth_l1[:, 1] * brake_weight  # matches original 2x brake penalty (1+1)
    return weighted_loss.mean()

def compute_predictive_metrics(
    pred_actions: np.ndarray,
    true_actions: np.ndarray,
    ego_states:   np.ndarray,  # (N, 2): [speed, last_steer]
) -> dict:
    """
    Metrics designed to correlate with online driving performance.
    Unlike compute_offline_metrics, these capture critical moments,
    temporal patterns, and speed-aware errors.
    Called only on validation/test data (shuffle=False) for temporal metrics.
    """
    pred_steer = pred_actions[:, 0]
    true_steer = true_actions[:, 0]
    pred_accel = pred_actions[:, 1]
    true_accel = true_actions[:, 1]
    steer_err  = np.abs(pred_steer - true_steer)
    speed      = ego_states[:, 0]

    steer_p95 = float(np.percentile(steer_err, 95))

    turn_mask = np.abs(true_steer) > 0.05
    active_turn_mae = float(np.mean(steer_err[turn_mask])) if np.any(turn_mask) else 0.0

    pred_delta = np.abs(np.diff(pred_steer))
    true_delta = np.abs(np.diff(true_steer))
    jitter_ratio = float(np.mean(pred_delta) / (np.mean(true_delta) + 1e-6))

    steer_max = np.max(np.abs(true_steer))
    out_of_bounds_rate = float(np.mean(np.abs(pred_steer) > steer_max + 0.05))

    speed_weighted_steer_mae = float(np.mean(steer_err * speed) / (np.mean(speed) + 1e-6))

    critical_mask = (np.abs(true_steer) > 0.1) & (speed > 0.3)
    critical_turn_mae = float(np.mean(steer_err[critical_mask])) if np.any(critical_mask) else 0.0

    # Pre-brake anticipation: T steps before each braking onset, fraction already braking
    brake_onset = np.where(
        (true_accel < -0.05) & (np.concatenate([[0.0], true_accel[:-1]]) >= -0.05)
    )[0]
    T = 10
    anticipation_scores = []
    for onset in brake_onset:
        start = max(0, onset - T)
        if start < onset:
            anticipation_scores.append(float(np.mean(pred_accel[start:onset] < -0.02)))
    pre_brake_anticipation = float(np.mean(anticipation_scores)) if anticipation_scores else 0.0

    return {
        "steer_p95_error":          steer_p95,
        "active_turn_mae":          active_turn_mae,
        "jitter_ratio":             jitter_ratio,
        "out_of_bounds_rate":       out_of_bounds_rate,
        "speed_weighted_steer_mae": speed_weighted_steer_mae,
        "critical_turn_mae":        critical_turn_mae,
        "pre_brake_anticipation":   pre_brake_anticipation,
    }


def compute_heading_metrics(
    pred_actions: np.ndarray,
    true_actions: np.ndarray,
    ego_full:     np.ndarray,  # (N, 5): [speed, last_steer, fwd_speed, lat_speed, heading_delta]
    window:       int = 20,
) -> dict:
    """
    Heading-delta metrics for lane-following quality.
    Requires ego_state_full (5-dim) from precomputed files.
    Returns empty dict if ego_full is all-zeros (precompute not re-run yet).
    """
    if ego_full.shape[1] < 5 or not np.any(ego_full[:, 4]):
        return {}

    speed         = ego_full[:, 0]
    true_hdelta   = ego_full[:, 4]
    pred_steer    = pred_actions[:, 0]
    true_steer    = true_actions[:, 0]

    # Calibrate K: heading_delta ≈ K * steer * speed
    valid = (np.abs(true_steer) > 0.1) & (speed > 0.1)
    if np.sum(valid) >= 10:
        ratios = true_hdelta[valid] / (true_steer[valid] * speed[valid])
        K = float(np.median(ratios))
    else:
        K = 1.0

    pred_hdelta = K * pred_steer * speed

    heading_dir_acc  = float(np.mean(np.sign(pred_hdelta) == np.sign(true_hdelta)))
    heading_delta_mae = float(np.mean(np.abs(pred_hdelta - true_hdelta)))

    # Short-window cumulative heading divergence (W frames ≈ 1 second)
    N = len(pred_hdelta)
    window_errors = []
    for start in range(0, N - window + 1, window):
        window_errors.append(abs(float(np.sum(pred_hdelta[start:start + window] - true_hdelta[start:start + window]))))

    if window_errors:
        window_div_mean = float(np.mean(window_errors))
        window_div_p95  = float(np.percentile(window_errors, 95))
    else:
        window_div_mean = 0.0
        window_div_p95  = 0.0

    return {
        "heading_dir_acc":        heading_dir_acc,
        "heading_delta_mae":      heading_delta_mae,
        "window_heading_div_mean": window_div_mean,
        "window_heading_div_p95":  window_div_p95,
    }


def compute_offline_metrics(pred_actions: np.ndarray, true_actions: np.ndarray) -> dict:
    steering_mse  = float(np.mean((pred_actions[:, 0] - true_actions[:, 0]) ** 2))
    accel_mse     = float(np.mean((pred_actions[:, 1] - true_actions[:, 1]) ** 2))
    
    steering_mae  = float(np.mean(np.abs(pred_actions[:, 0] - true_actions[:, 0])))
    accel_mae     = float(np.mean(np.abs(pred_actions[:, 1] - true_actions[:, 1])))

    # Direction accuracies
    steer_dir_acc = float(np.mean(np.sign(pred_actions[:, 0]) == np.sign(true_actions[:, 0])))
    direction_acc = float(np.mean(np.sign(pred_actions[:, 1]) == np.sign(true_actions[:, 1])))

    # Braking accuracy
    true_brake = (true_actions[:, 1] < -0.05)
    if np.any(true_brake):
        brake_acc = float(np.mean(pred_actions[true_brake, 1] < -0.05))
    else:
        brake_acc = 1.0

    # Pearson correlation
    corr, _ = pearsonr(pred_actions[:, 0], true_actions[:, 0]) if len(pred_actions) > 1 else (0.0, 0.0)

    return {
        "steering_mse":  steering_mse,
        "accel_mse":     accel_mse,
        "steering_mae":  steering_mae,
        "accel_mae":     accel_mae,
        "steering_corr": float(corr),
        "steering_dir_acc": steer_dir_acc,
        "direction_acc": direction_acc,
        "brake_acc":     brake_acc,
    }