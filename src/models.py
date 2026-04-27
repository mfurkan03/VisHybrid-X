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

# Append the Depth-Anything-V2 path
target_folder = Path(__file__).resolve().parent.parent / 'Depth-Anything-V2'
sys.path.append(str(target_folder))
from depth_anything_v2.dpt import DepthAnythingV2


# ==========================================
# 1. DEPTH ESTIMATION WRAPPER
# ==========================================
class DepthEstimationModel:
    """
    Unified wrapper for DepthAnythingV2 supporting both inference (frozen)
    and fine-tuning (trainable).
    """
    def __init__(self, encoder='vits', finetuned_path=None, trainable=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        self.model = DepthAnythingV2(**model_configs[encoder])

        if finetuned_path and os.path.exists(finetuned_path):
            self.model.load_state_dict(torch.load(finetuned_path, map_location='cpu'))
            print(f"[INFO] Fine-tuned DPT loaded: {finetuned_path}")
        else:
            ckpt_path = f'{target_folder}/checkpoints/depth_anything_v2_{encoder}.pth'
            if os.path.exists(ckpt_path):
                self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
                print(f"[INFO] Base DepthAnythingV2 loaded: {ckpt_path}")

        self.model = self.model.to(self.device)
        self.use_fp16 = (self.device == 'cuda')
        self.trainable = trainable

        if self.trainable:
            self.set_train_mode()
        else:
            self.set_eval_mode()

    def set_train_mode(self):
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True

    def set_eval_mode(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def predict_batch_with_grad(self, rgb_images: np.ndarray) -> torch.Tensor:
        batch_tensors = []
        for rgb in rgb_images:
            if rgb.dtype == np.uint8:
                rgb = rgb.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
            std  = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
            rgb  = (rgb - mean) / std
            t = torch.from_numpy(rgb).permute(2, 0, 1).float()
            batch_tensors.append(t)

        x = torch.stack(batch_tensors).to(self.device)
        depth_raw = self.model(x)  # No autocast here
        depth_raw = depth_raw.unsqueeze(1)
        depth_resized = F.interpolate(depth_raw, size=(84, 84), mode='bilinear', align_corners=False)
        return depth_resized


# ==========================================
# 2. EGO-STATE UTILITIES
# ==========================================

# Number of dimensions fed into the policy network.
# Only total_speed and last_steer are passed to the model; the remaining
# fields are recorded for logging / temporal-difference analysis but are
# NOT sent to the network.
EGO_DIM = 2   # [total_speed, last_steer]


class EgoReading(NamedTuple):
    """
    Full ego-state snapshot captured at every step.

    Fields sent to the model (ego_model slice)
    -------------------------------------------
    total_speed   : scalar wheel-speed-sensor equivalent (m/s), normalised to [-1, 1]
    last_steer    : previous steering command, already in [-1, 1]

    Fields collected but NOT sent to the model
    ------------------------------------------
    forward_speed : longitudinal velocity component (m/s), normalised
    lateral_speed : lateral velocity component (m/s), normalised
    heading_delta : change in heading since last step (rad), normalised
    timestamp     : wall-clock time of this reading (float, seconds since epoch)

    Helper
    ------
    ego_model     : np.ndarray of shape (EGO_DIM,) – the slice passed to the network
    """
    total_speed   : float   # wheel-speed proxy  → sent to model
    last_steer    : float   # steering angle      → sent to model
    forward_speed : float   # logged only
    lateral_speed : float   # logged only
    heading_delta : float   # logged only
    timestamp     : float   # wall-clock seconds since epoch

    @property
    def ego_model(self) -> np.ndarray:
        """Return the (EGO_DIM,) array that is actually fed to the policy network."""
        return np.array([self.total_speed, self.last_steer], dtype=np.float32)


def extract_ego_state(agent, last_steer: float = 0.0) -> EgoReading:
    """
    Build a full ego-state snapshot from the MetaDrive agent.

    What the model sees  (ego_reading.ego_model)
    ---------------------------------------------
    [0] total_speed  – magnitude of the velocity vector, analogous to a
                       wheel-speed sensor reading; clipped & normalised to [0, 1]
    [1] last_steer   – previous steering action, in [-1, 1], analogous to a
                       steering-wheel angle sensor

    What is collected but NOT passed to the model
    ----------------------------------------------
    forward_speed    – longitudinal velocity component, normalised to [-1, 1]
    lateral_speed    – lateral velocity component, normalised to [-1, 1]
    heading_delta    – change in heading (rad) since the previous step,
                       normalised to [-1, 1]
    timestamp        – time.time() at the moment of this call; use the
                       difference between two timestamps to compute dt for
                       frame-rate analysis or temporal derivatives

    All normalised values are clamped to their stated ranges even under
    edge cases (e.g. very high speeds, simulation glitches).
    """
    timestamp = time.time()

    vel     = agent.velocity           # panda3d Vec3 or similar
    spd_fwd = float(vel[0])            # X-axis in MetaDrive body frame
    spd_lat = float(vel[1])            # Y-axis in MetaDrive body frame

    # Total speed: magnitude of the 2-D velocity → wheel-speed-sensor proxy
    total_speed_raw = float(np.sqrt(spd_fwd ** 2 + spd_lat ** 2))

    # Heading delta: wrap-corrected difference from the previous step
    current_heading = float(agent.heading_theta)
    prev_heading    = getattr(agent, '_prev_heading', current_heading)
    heading_delta   = current_heading - prev_heading
    heading_delta   = (heading_delta + np.pi) % (2 * np.pi) - np.pi   # wrap to [-π, π]
    agent._prev_heading = current_heading

    # Normalise
    # - total speed : max ~30 m/s → [0, 1]  (speed is always non-negative)
    # - components  : forward max ~30 m/s, lateral max ~30 m/s → [-1, 1]
    # - heading δ   : max ~0.3 rad/step → [-1, 1]
    total_speed_norm = float(np.clip(total_speed_raw / 30.0, 0.0, 1.0))
    fwd_norm         = float(np.clip(spd_fwd / 30.0,         -1.0, 1.0))
    lat_norm         = float(np.clip(spd_lat / 30.0,         -1.0, 1.0))
    hdelta_norm      = float(np.clip(heading_delta / 0.3,    -1.0, 1.0))
    steer_clipped    = float(np.clip(last_steer,             -1.0, 1.0))

    return EgoReading(
        total_speed   = total_speed_norm,
        last_steer    = steer_clipped,
        forward_speed = fwd_norm,
        lateral_speed = lat_norm,
        heading_delta = hdelta_norm,
        timestamp     = timestamp,
    )


# ==========================================
# 3. DRIVING POLICY NETWORK  (ego-aware)
# ==========================================
class DrivingPolicyNet(nn.Module):
    """
    Two-stream policy network:
      • CNN stream  – processes (2, 84, 84) visual obs          → 512-d feature
      • Ego stream  – small MLP on EGO_DIM ego-state vector     →  32-d feature

    The ego input contains only the signals that a real vehicle's on-board
    sensors would expose:
        [0] total_speed  – wheel-speed-sensor equivalent, normalised to [0, 1]
        [1] last_steer   – steering-wheel angle, normalised to [-1, 1]

    Additional signals (forward/lateral speed components, heading delta,
    timestamps) are collected by extract_ego_state() for logging and
    temporal analysis but are intentionally excluded from the model input.
    """

    def __init__(self, in_channels: int = 2, out_dim: int = 2, ego_dim: int = EGO_DIM):
        super().__init__()

        # --- Visual (CNN) stream ---
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc_vis  = nn.Linear(64 * 7 * 7, 512)

        # --- Ego-state stream ---
        self.ego_fc = nn.Sequential(
            nn.Linear(ego_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # --- Fusion head ---
        self.fc_out = nn.Linear(512 + 32, out_dim)

    def forward(self, x: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : (B, 2, 84, 84)  – stacked depth + lane-mask observation
        ego : (B, EGO_DIM)    – [total_speed, last_steer]
                                 obtain via ego_reading.ego_model for each sample

        Returns
        -------
        (B, 2) predicted [steer, accel]
        """
        # CNN stream
        v = F.relu(self.conv1(x))
        v = F.relu(self.conv2(v))
        v = F.relu(self.conv3(v))
        v = F.relu(self.fc_vis(self.flatten(v)))

        # Ego stream
        e = self.ego_fc(ego)

        # Fuse & predict
        return self.fc_out(torch.cat([v, e], dim=1))
    
class DrivingPolicyNet2(nn.Module):
    def __init__(self, in_channels: int = 2, out_dim: int = 2, ego_dim: int = 2):
        super().__init__()

        # --- Visual (CNN) stream ---
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        
        # Increased capacity in the visual bottleneck
        self.fc_vis = nn.Linear(64 * 7 * 7, 512)
        self.vis_bn = nn.BatchNorm1d(512)

        # --- Ego-state stream ---
        self.ego_fc = nn.Sequential(
            nn.Linear(ego_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64), # Increased width
            nn.ReLU(inplace=True),
        )

        # --- Enhanced Fusion/Decision Head ---
        # We increase depth and add BatchNorm to stabilize the 
        # combined signals from the two different streams.
        self.decision_layer = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            
            nn.Linear(64, out_dim)
        )

    def forward(self, x: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        # CNN stream
        v = F.relu(self.conv1(x))
        v = F.relu(self.conv2(v))
        v = F.relu(self.conv3(v))
        
        v = self.flatten(v)
        v = F.relu(self.vis_bn(self.fc_vis(v)))

        # Ego stream
        e = self.ego_fc(ego)

        # Fuse & predict through the deeper decision head
        combined = torch.cat([v, e], dim=1)
        return self.decision_layer(combined)