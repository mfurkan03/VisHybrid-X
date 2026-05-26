"""
policy/losses.py – loss functions and offline evaluation metrics.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.distributions import Beta


def custom_driving_loss_beta(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    target_01: torch.Tensor,
    turn_weight_scale: float = 1.5,
) -> torch.Tensor:
    """
    NLL loss for Beta-distributed policy output.

    alpha, beta : (B, 2) concentration parameters (both > 1)
    target_01   : (B, 2) expert actions mapped to [0, 1]

    Weighted the same way as the original smooth-L1 loss:
      - Turn weight : proportional to |steer| magnitude, scaled by turn_weight_scale
      - 2× weight on braking events (accel_01 < 0.45 ↔ accel < -0.1)

    turn_weight_scale: multiplier on the steer-magnitude penalty term.
      Default 1.5 → max 2.5× on full lock.
      Raise to 3.0 to double turn emphasis (max 4×) when the model under-steers in corners.
    """
    # Clip to (0.01, 0.99) — not just 1e-6.  Expert actions occasionally exceed
    # [-1,1] (MetaDrive raw output); after (a+1)/2 mapping they land outside [0,1]
    # and a 1e-6 clamp silently treats them as boundary samples.  Boundary samples
    # produce NLL gradients ~16× larger than interior ones, collapsing β → 1.0
    # and pinning the mode to exactly 1.0 (float underflow in Softplus).
    t = target_01.clamp(0.01, 0.99)

    dist = Beta(alpha, beta)
    nll  = -dist.log_prob(t)                              # (B, 2), positive

    steer_mag   = (t[:, 0] - 0.5).abs() * 2.0            # |steer| in [0,1]
    turn_weight = 1.0 + turn_weight_scale * steer_mag
    weighted    = nll.clone()
    weighted[:, 0] = nll[:, 0] * turn_weight

    brake_weight = 1.0 + (t[:, 1] < 0.45).float() * 2.0   # 3x on braking events (accel_01 < 0.45 ↔ accel < -0.1)
    weighted[:, 1] = nll[:, 1] * brake_weight

    return weighted.mean()


def speed_steer_coupling_loss(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    steer_threshold: float = 0.3,
) -> torch.Tensor:
    """
    P3: couple the otherwise-independent steer and throttle heads.

    The steer and throttle heads predict from the same features but never see
    each other, so the throttle head keeps accelerating through corners. This
    soft penalty pushes the policy to ease off the throttle when it predicts a
    sharp turn — the behaviour that stops "accelerating through turns" swerving.

    ⚠️  OFF BY DEFAULT (coupling_weight=0.0). This penalty is only valid when
    the demonstrator actually slows for turns. The MetaDrive IDM/PID expert
    MAINTAINS throttle through curves (measured: expert throttle ≈ +0.25 at
    |steer| ∈ [0.3, 0.5]), so the penalty directly fights the imitation target
    and collapses predicted turn-throttle to ~0 — in closed loop the car then
    coasts to a stop at every curve/junction. It is one-directional (only
    penalises acceleration, never rewards it), so even a small weight applies
    steady downward pressure that the weak Beta-NLL throttle signal cannot
    counter. Enable only with an expert that genuinely decelerates for turns.

    Uses the Beta means (= mu) as a differentiable point estimate of each
    action, mapped back to [-1, 1]:
        steer    = 2 * mu_steer    - 1   ∈ [-1, 1]
        throttle = 2 * mu_throttle - 1   ∈ [-1, 1]   (>0 accelerate, <0 brake)

    penalty = mean( relu(|steer| - steer_threshold) * relu(throttle) )
      • only sharp turns (|steer| > threshold) contribute — gentle steering is
        left alone so normal lane-following is not penalised,
      • only positive throttle is penalised — braking *into* a turn is desired,
      • gradients flow to both mu heads, so the coupling is learned jointly.

    Returns a non-negative scalar; weight it in the caller.
    """
    mu       = alpha / (alpha + beta)            # (B, 2) in (0, 1)
    steer    = mu[:, 0] * 2.0 - 1.0              # [-1, 1]
    throttle = mu[:, 1] * 2.0 - 1.0              # [-1, 1]
    sharp     = (steer.abs() - steer_threshold).clamp(min=0.0)   # >0 only on sharp turns
    accel_pos = throttle.clamp(min=0.0)                          # penalise acceleration only
    return (sharp * accel_pos).mean()


def custom_driving_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    smooth_l1    = nn.functional.smooth_l1_loss(pred, target, reduction='none')
    weighted_loss = smooth_l1.clone()
    brake_weight = 1.0 + (target[:, 1] < -0.1).float() * 3.0  # 4x on braking events
    weighted_loss[:, 1] = smooth_l1[:, 1] * brake_weight
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