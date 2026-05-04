"""
policy/losses.py – loss functions and offline evaluation metrics.
"""

import numpy as np
import torch
from scipy.stats import pearsonr


import torch
import torch.nn as nn

def custom_driving_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:


    smooth_l1 = nn.functional.mse_loss(pred, target)
    
    brake_mask = (target[:, 1] < -0.1).float()
    
    penalty = 1.0 + brake_mask * 1.0
    weighted_loss = smooth_l1.clone()
    weighted_loss[:, 1] = smooth_l1[:, 1] * penalty
    return weighted_loss.mean()

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

import numpy as np

def compute_predictive_metrics(pred_np, true_np):
    """
    Predictive metrics to determine if a model will actually drive well in closed-loop simulation.
    Requires that the input arrays are sequential (DataLoader had shuffle=False).
    """
    metrics = {}
    
    steer_pred = pred_np[:, 0]
    steer_true = true_np[:, 0]
    
    steer_errors = np.abs(steer_pred - steer_true)

    # 1. 95th Percentile Error (Catches Catastrophic Mistakes)
    # Average MAE hides the moments your model decides to swerve into a wall. 
    # This looks at the worst 5% of your errors.
    metrics['steer_95th_pctl_err'] = np.percentile(steer_errors, 95)
    
    # 2. Active Turn MAE (Filters out driving straight)
    # How well does the model steer when it ACTUALLY matters?
    # We define a "turn" as any time the expert steers more than 0.05.
    turn_indices = np.abs(steer_true) > 0.05
    if np.sum(turn_indices) > 0:
        metrics['active_turn_mae'] = np.mean(steer_errors[turn_indices])
    else:
        metrics['active_turn_mae'] = 0.0

    # 3. Temporal Jitter / Smoothness (Oscillation Check)
    # Frame-to-frame change in steering. A nervous driver oscillates rapidly.
    # We compare the model's jitter to the expert's jitter.
    pred_jerk = np.abs(np.diff(steer_pred))
    true_jerk = np.abs(np.diff(steer_true))
    
    # +1e-6 to avoid division by zero
    metrics['jitter_ratio'] = np.mean(pred_jerk) / (np.mean(true_jerk) + 1e-6)

    # 4. Action Out-of-Bounds Rate (Confidence Check)
    # How often does the model output steering outside the bounds of the expert's max/min?
    expert_max = np.max(steer_true) + 0.1
    expert_min = np.min(steer_true) - 0.1
    oob_count = np.sum((steer_pred > expert_max) | (steer_pred < expert_min))
    metrics['out_of_bounds_rate'] = oob_count / len(steer_pred)

    return metrics