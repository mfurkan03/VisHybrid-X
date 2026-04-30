"""
policy/losses.py – loss functions and offline evaluation metrics.
"""

import numpy as np
import torch
from scipy.stats import pearsonr


import torch
import torch.nn as nn

def custom_driving_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:


    smooth_l1 = nn.functional.smooth_l1_loss(pred, target, reduction='none', beta=1.0)
    
    brake_mask = (target[:, 1] < -0.1).float()
    
    penalty = 1.0 + brake_mask * 3.0
    weighted_loss = smooth_l1.clone()
    weighted_loss[:, 1] = smooth_l1[:, 1] * penalty
    return weighted_loss.mean()

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