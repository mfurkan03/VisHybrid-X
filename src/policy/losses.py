"""
policy/losses.py – loss functions and offline evaluation metrics.
"""

import numpy as np
import torch
from scipy.stats import pearsonr


def custom_driving_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Weighted MSE: braking events get 3× penalty on the acceleration channel."""
    mse         = (pred - target) ** 2
    brake_mask  = (target[:, 1] < 0.0).float()
    penalty     = 1.0 + brake_mask * 3.0
    mse_w       = mse.clone()
    mse_w[:, 1] = mse[:, 1] * penalty
    return mse_w.mean()


def compute_offline_metrics(pred_actions: np.ndarray, true_actions: np.ndarray) -> dict:
    steering_mse  = float(np.mean((pred_actions[:, 0] - true_actions[:, 0]) ** 2))
    accel_mse     = float(np.mean((pred_actions[:, 1] - true_actions[:, 1]) ** 2))
    corr, _       = pearsonr(pred_actions[:, 0], true_actions[:, 0])
    direction_acc = float(np.mean(np.sign(pred_actions[:, 1]) == np.sign(true_actions[:, 1])))
    return {
        "steering_mse":  steering_mse,
        "accel_mse":     accel_mse,
        "steering_corr": float(corr),
        "direction_acc": direction_acc,
    }