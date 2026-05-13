"""
Actor-Critic Ağ Mimarisi
========================
train_test_policy.py'deki DrivingPolicyNet ile uyumlu CNN omurgası.
IL pretrained ağırlıklar yüklenebilir.
"""
import torch
import torch.nn as nn
import numpy as np


class ActorCritic(nn.Module):
    """
    Giriş : (B, 2, 84, 84)  →  [depth, lane_mask]
    Actor : [steering_mean, throttle_mean] + öğrenilebilir log_std
    Critic: state value V(s)
    """

    def __init__(self):
        super().__init__()

        # ── CNN Feature Extractors (DrivingPolicyNet ile aynı mimari) ──
        self.depth_conv = nn.Sequential(
            nn.Conv2d(1, 24, 5, 2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, 2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, 2), nn.ReLU(),
            nn.Conv2d(48, 64, 3, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        )
        self.lane_conv = nn.Sequential(
            nn.Conv2d(1, 24, 5, 2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, 2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, 2), nn.ReLU(),
            nn.Conv2d(48, 64, 3, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        )

        fused_dim = 64 * 3 * 3 * 2   # 1152

        self.shared_fc = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
        )

        # ── Actor Heads ──
        self.steer_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.accel_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        # log_std başlangıç değerini düşürüyoruz ki eylem varyansı (rastgele direksiyon kırmalar) azalsın.
        # torch.zeros(2) std=1.0 demekti. -1.0 ise std=0.36 yapar.
        self.log_std = nn.Parameter(torch.ones(2) * -1.0)

        # ── Critic Head ──
        self.value_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.flatten = nn.Flatten()

        # ── Throttle Bias: Aracı baştan gaza basmaya teşvik et ──
        # Bias'ı 0.1'e çektik ki ajan körü körüne gazı köklemesin.
        nn.init.constant_(self.accel_head[-1].bias, 0.1)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        depth_feat = self.flatten(self.depth_conv(x[:, 0:1]))
        lane_feat  = self.flatten(self.lane_conv(x[:, 1:2]))
        fused = torch.cat([depth_feat, lane_feat], dim=1)
        return self.shared_fc(fused)

    def forward(self, x: torch.Tensor):
        shared = self._features(x)
        steer_mean = self.steer_head(shared)
        accel_mean = self.accel_head(shared)
        action_mean = torch.cat([steer_mean, accel_mean], dim=1)  # (B, 2)
        value = self.value_head(shared).squeeze(-1)               # (B,)
        return action_mean, value

    def get_action_and_value(self, x: torch.Tensor, action=None):
        """PPO için: aksiyon örnekle veya verilen aksiyonun log prob'unu hesapla."""
        action_mean, value = self.forward(x)
        std = self.log_std.exp().expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        _, value = self.forward(x)
        return value

    def load_pretrained_actor(self, path: str, device='cpu'):
        """IL ile eğitilmiş DrivingPolicyNet ağırlıklarını yükler (sadece actor kısmı)."""
        state = torch.load(path, map_location=device, weights_only=True)
        own = self.state_dict()

        loaded = 0
        for key in state:
            if key in own and own[key].shape == state[key].shape:
                own[key] = state[key]
                loaded += 1

        self.load_state_dict(own)
        print(f"[INFO] IL pretrained: {loaded} katman yüklendi ({path})")
