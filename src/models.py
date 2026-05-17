"""
models.py – all neural network definitions and ego-state utilities.

Public API
----------
DepthEstimationModel   – DepthAnythingV2 wrapper (frozen or trainable)
EgoReading             – NamedTuple snapshot from extract_ego_state()
EGO_DIM                – number of dimensions fed to the policy network (2)
extract_ego_state()    – build EgoReading from a MetaDrive agent
DrivingPolicyNet       – two-stream CNN + ego policy network
DrivingPolicyNet2      – deeper variant with BatchNorm fusion head
"""

import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Depth-Anything-V2 path
# ---------------------------------------------------------------------------
target_folder = Path(__file__).resolve().parent.parent / "Depth-Anything-V2"
sys.path.append(str(target_folder))
from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================
# 1. DEPTH ESTIMATION WRAPPER
# ============================================================
class DepthEstimationModel:
    """
    Unified wrapper for DepthAnythingV2 supporting both inference (frozen)
    and fine-tuning (trainable).
    """

    _CONFIGS = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192,  384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384,  768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    def __init__(self, encoder: str = "vits", finetuned_path: str = None, trainable: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model  = DepthAnythingV2(**self._CONFIGS[encoder])

        if finetuned_path and os.path.exists(finetuned_path):
            self.model.load_state_dict(torch.load(finetuned_path, map_location="cpu"))
            print(f"[INFO] Fine-tuned DPT loaded: {finetuned_path}")
        else:
            ckpt_path = f"{target_folder}/checkpoints/depth_anything_v2_{encoder}.pth"
            if os.path.exists(ckpt_path):
                self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                print(f"[INFO] Base DepthAnythingV2 loaded: {ckpt_path}")

        self.model     = self.model.to(self.device)
        self.use_fp16  = (self.device == "cuda")
        self.trainable = trainable

        if self.trainable:
            self.set_train_mode()
        else:
            self.set_eval_mode()

    # ------------------------------------------------------------------
    def set_train_mode(self):
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True

    def set_eval_mode(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    def predict_batch_with_grad(self, rgb_images: np.ndarray) -> torch.Tensor:
        batch_tensors = []
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std  = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        for rgb in rgb_images:
            if rgb.dtype == np.uint8:
                rgb = rgb.astype(np.float32) / 255.0
            rgb = (rgb - mean) / std
            batch_tensors.append(torch.from_numpy(rgb).permute(2, 0, 1).float())

        x         = torch.stack(batch_tensors).to(self.device)
        depth_raw = self.model(x).unsqueeze(1)
        return depth_raw


# ============================================================
# 2. EGO-STATE UTILITIES
# ============================================================

EGO_DIM = 3  # [total_speed, last_steer, heading_delta]


class EgoReading(NamedTuple):
    """
    Full ego-state snapshot captured at every step.

    Sent to the model (ego_model slice)
    ------------------------------------
    total_speed   : wheel-speed proxy, normalised to [0, 1]
    last_steer    : previous steering command, in [-1, 1]

    Collected but NOT sent to the model
    ------------------------------------
    forward_speed : longitudinal velocity component, normalised to [-1, 1]
    lateral_speed : lateral velocity component, normalised to [-1, 1]
    heading_delta : change in heading since last step (rad), normalised to [-1, 1]
    timestamp     : wall-clock time of this reading (float, seconds since epoch)
    """
    total_speed   : float
    last_steer    : float
    forward_speed : float
    lateral_speed : float
    heading_delta : float
    timestamp     : float

    @property
    def ego_model(self) -> np.ndarray:
        """Return the (EGO_DIM,) array actually fed to the policy network."""
        return np.array([self.total_speed, self.last_steer, self.heading_delta], dtype=np.float32)


def extract_ego_state(agent, last_steer: float = 0.0) -> EgoReading:
    """
    Build a full EgoReading snapshot from a MetaDrive agent.

    Model input  →  ego_reading.ego_model  →  [total_speed, last_steer]
    Logged only  →  forward_speed, lateral_speed, heading_delta, timestamp
    """
    timestamp = time.time()

    vel     = agent.velocity
    spd_fwd = float(vel[0])
    spd_lat = float(vel[1])

    total_speed_raw = float(np.sqrt(spd_fwd ** 2 + spd_lat ** 2))

    current_heading = float(agent.heading_theta)
    prev_heading    = getattr(agent, "_prev_heading", current_heading)
    heading_delta   = current_heading - prev_heading
    heading_delta   = (heading_delta + np.pi) % (2 * np.pi) - np.pi
    agent._prev_heading = current_heading

    return EgoReading(
        total_speed   = float(np.clip(total_speed_raw / 30.0, 0.0, 1.0)),
        last_steer    = float(np.clip(last_steer,             -1.0, 1.0)),
        forward_speed = float(np.clip(spd_fwd / 30.0,         -1.0, 1.0)),
        lateral_speed = float(np.clip(spd_lat / 30.0,         -1.0, 1.0)),
        heading_delta = float(np.clip(heading_delta / 0.3,    -1.0, 1.0)),
        timestamp     = timestamp,
    )

# ============================================================
# 3. DRIVING POLICY NETWORKS
# ============================================================
class DrivingPolicyNet(nn.Module):
    """
    Two-stream policy network.

    Visual stream  : CNN on (4, H, W) observation  → 512-d feature
    Ego stream     : MLP on EGO_DIM ego-state vector →  32-d feature
    Fusion head    : Linear(544 → 2)  →  [steer, accel]

    Ego input: [total_speed, last_steer]
    """

    def __init__(self, in_channels: int = 4, out_dim: int = 2, ego_dim: int = EGO_DIM, p: float = 0.2, image_size: int = None):
        super().__init__()

        self.conv1   = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3   = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            dummy_out = self.flatten(self.conv3(self.conv2(self.conv1(dummy))))
            flattened_dim = dummy_out.shape[1]
        
        self.fc_vis  = nn.Linear(flattened_dim, 512)
        self.dropout_vis = nn.Dropout(p) # Görsel feature dropout

        self.ego_fc = nn.Sequential(
            nn.Linear(ego_dim, 64), nn.ReLU(),
            nn.Dropout(p), # Ego state dropout
            nn.Linear(64, 32),      nn.ReLU(),
        )

        merged_dim = 512 + 32
        self.steer_alpha_head    = nn.Sequential(nn.Linear(merged_dim, 1), nn.Softplus())
        self.steer_beta_head     = nn.Sequential(nn.Linear(merged_dim, 1), nn.Softplus())
        self.throttle_alpha_head = nn.Sequential(nn.Linear(merged_dim, 1), nn.Softplus())
        self.throttle_beta_head  = nn.Sequential(nn.Linear(merged_dim, 1), nn.Softplus())
        # Bias beta > alpha initially → Beta mode ≈ 0.40 in [0,1] (accel ≈ -0.20) — mild brake prior.
        nn.init.constant_(self.throttle_beta_head[0].bias, 1.4)

    def forward(self, x: torch.Tensor, ego: torch.Tensor, return_features: bool = False):
        v = F.relu(self.conv1(x))
        v = F.relu(self.conv2(v))
        v = F.relu(self.conv3(v))
        v = self.flatten(v)
        v = F.relu(self.fc_vis(v))
        v = self.dropout_vis(v)
        e = self.ego_fc(ego)
        merged = torch.cat([v, e], dim=1)
        if return_features:
            return merged
        alpha = torch.cat([self.steer_alpha_head(merged), self.throttle_alpha_head(merged)], dim=1) + 2.0
        beta  = torch.cat([self.steer_beta_head(merged),  self.throttle_beta_head(merged)],  dim=1) + 2.0
        return alpha, beta

class _ImpalaResBlock(nn.Module):
    """Pre-activation residual block used inside the IMPALA CNN."""
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def _impala_stage(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        _ImpalaResBlock(out_ch),
        _ImpalaResBlock(out_ch),
    )


class ImpalaNet(nn.Module):
    """
    IMPALA-style residual CNN backbone — the standard choice for visual RL.

    Why this architecture over a plain deep CNN:
    - Residual connections → stable gradients through depth
    - No BatchNorm → safe with batch_size=1 during RL rollouts (BatchNorm
      is undefined at BS=1 and causes train/eval stat drift in RL)
    - MaxPool stages → better spatial information retention than strided convs
    - Pre-activation ReLU in residual blocks → smoother gradient flow

    Visual stream    : 3 IMPALA stages (32→64→64 ch) → 512-d shared feature
    Ego stream       : 2-layer MLP                    →  32-d feature
    Steer heads      : steer_alpha_head / steer_beta_head (544→1 each)
    Throttle heads   : throttle_alpha_head / throttle_beta_head (544→1 each)

    Separate head pairs let steering and throttle specialise independently
    from the shared visual representation.

    RL note: call model.train() during gradient updates and model.eval()
    during rollout collection — Dropout is the only stateful layer.
    """

    def __init__(self, in_channels: int = 4, out_dim: int = 2, ego_dim: int = EGO_DIM, p: float = 0.3, image_size: int = None):
        super().__init__()

        self.cnn = nn.Sequential(
            _impala_stage(in_channels, 32),
            _impala_stage(32, 64),
            _impala_stage(64, 64),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy         = torch.zeros(1, in_channels, image_size, image_size)
            flattened_dim = self.cnn(dummy).shape[1]

        self.vis_proj = nn.Sequential(
            nn.Linear(flattened_dim, 512), nn.ReLU(),
        )

        self.ego_fc = nn.Sequential(
            nn.Linear(ego_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),      nn.ReLU(),
        )

        merged_dim = 512 + 32
        self.steer_alpha_head    = nn.Sequential(nn.Linear(merged_dim, 1), nn.Softplus())
        self.steer_beta_head     = nn.Sequential(nn.Linear(merged_dim, 1), nn.Softplus())
        self.throttle_alpha_head = nn.Sequential(nn.Linear(merged_dim, 1), nn.Softplus())
        self.throttle_beta_head  = nn.Sequential(nn.Linear(merged_dim, 1), nn.Softplus())
        nn.init.constant_(self.throttle_beta_head[0].bias, 1.4)

    def forward(self, x: torch.Tensor, ego: torch.Tensor, return_features: bool = False):
        v = self.vis_proj(self.cnn(x))
        e = self.ego_fc(ego)
        merged = torch.cat([v, e], dim=1)
        if return_features:
            return merged
        alpha = torch.cat([self.steer_alpha_head(merged), self.throttle_alpha_head(merged)], dim=1) + 2.0
        beta  = torch.cat([self.steer_beta_head(merged),  self.throttle_beta_head(merged)],  dim=1) + 2.0
        return alpha, beta


# ============================================================
# SE building blocks (used only by ImpalaNetV2)
# ============================================================

class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x).flatten(1)
        return x * self.fc(s).view(x.size(0), x.size(1), 1, 1)


class _ImpalaResBlockSE(nn.Module):
    """Pre-activation residual block with SE channel attention."""
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.se = _SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.se(self.net(x))


def _impala_stage_se(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        _ImpalaResBlockSE(out_ch),
        _ImpalaResBlockSE(out_ch),
    )


class ImpalaNetV2(nn.Module):
    """
    Stronger IMPALA variant with better RGB handling.  Use --arch impala_v2.

    Improvements over ImpalaNet:
    - Wider stages (48→96→96 vs 32→64→64) — more capacity for RGB textures
    - SE (Squeeze-and-Excitation) in every res-block — channel attention lets
      each stage learn how much to weight depth vs R/G/B
    - Deeper vis_proj (flat→1024→512 vs flat→512) — richer feature compression
    - Larger output heads (256 vs 128 units)
    - ImageNet normalisation of RGB channels (1-3) inside forward(); depth
      channel (0) is already range-normalised upstream and is left as-is

    Same API as ImpalaNet: forward(x, ego) → [steer, accel]
    No BatchNorm — safe at batch_size=1 for RL rollouts.
    """

    _RGB_MEAN = [0.485, 0.456, 0.406]
    _RGB_STD  = [0.229, 0.224, 0.225]

    def __init__(self, in_channels: int = 4, out_dim: int = 2,
                 ego_dim: int = EGO_DIM, p: float = 0.3, image_size: int = None):
        super().__init__()

        self.register_buffer('rgb_mean', torch.tensor(self._RGB_MEAN).view(1, 3, 1, 1))
        self.register_buffer('rgb_std',  torch.tensor(self._RGB_STD).view(1, 3, 1, 1))

        self.cnn = nn.Sequential(
            _impala_stage_se(in_channels, 48),
            _impala_stage_se(48, 96),
            _impala_stage_se(96, 96),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy         = torch.zeros(1, in_channels, image_size, image_size)
            flattened_dim = self.cnn(dummy).shape[1]

        self.vis_proj = nn.Sequential(
            nn.Linear(flattened_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512),           nn.ReLU(),
        )

        self.ego_fc = nn.Sequential(
            nn.Linear(ego_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),      nn.ReLU(),
        )

        merged_dim = 512 + 32
        self.steer_alpha_head    = nn.Sequential(nn.Linear(merged_dim, 1), nn.Softplus())
        self.steer_beta_head     = nn.Sequential(nn.Linear(merged_dim, 1), nn.Softplus())
        self.throttle_alpha_head = nn.Sequential(nn.Linear(merged_dim, 1), nn.Softplus())
        self.throttle_beta_head  = nn.Sequential(nn.Linear(merged_dim, 1), nn.Softplus())
        nn.init.constant_(self.throttle_beta_head[0].bias, 1.4)

    def forward(self, x: torch.Tensor, ego: torch.Tensor, return_features: bool = False):
        x = torch.cat([
            x[:, :1],
            (x[:, 1:] - self.rgb_mean) / self.rgb_std,
        ], dim=1)
        v = self.vis_proj(self.cnn(x))
        e = self.ego_fc(ego)
        merged = torch.cat([v, e], dim=1)
        if return_features:
            return merged
        alpha = torch.cat([self.steer_alpha_head(merged), self.throttle_alpha_head(merged)], dim=1) + 2.0
        beta  = torch.cat([self.steer_beta_head(merged),  self.throttle_beta_head(merged)],  dim=1) + 2.0
        return alpha, beta


def build_policy(arch: str = "simple", image_size: int = None) -> nn.Module:
    """Factory: 'simple' → DrivingPolicyNet, 'impala' → ImpalaNet, 'impala_v2' → ImpalaNetV2."""
    if arch == "impala":
        return ImpalaNet(image_size=image_size)
    if arch == "impala_v2":
        return ImpalaNetV2(image_size=image_size)
    return DrivingPolicyNet(image_size=image_size)