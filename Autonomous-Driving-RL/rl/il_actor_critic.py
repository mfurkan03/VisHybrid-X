"""
IL Actor-Critic Wrapper
=======================
Wraps any IL policy (ImpalaNet, ImpalaNetV2, DrivingPolicyNet) as a PPO
actor-critic by adding a value head and a learnable log_std on top of the
shared 544-d feature vector that the IL backbone already computes.

IL→RL workflow:
    il_model = build_policy("impala", image_size=84)
    policy   = ILActorCritic(il_model).to(device)
    policy.load_from_il_checkpoint("models/policy_model_best.pth", device)
    # Then fine-tune with PPO as normal.
"""
import sys
from pathlib import Path

# Allow importing from the parent IL repo regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
from torch.distributions import Normal


class ILActorCritic(nn.Module):
    """
    PPO actor-critic that uses an IL backbone for feature extraction.

    The IL backbone exposes a 544-d merged feature (512 visual + 32 ego)
    via forward(..., return_features=True).  We add:
      - value_head : Linear(544→128→1)  — critic
      - log_std    : learnable (2,) parameter — action distribution width

    The actor output reuses the IL model's existing steer_head / accel_head
    (ImpalaNet / ImpalaNetV2) or fc_out (DrivingPolicyNet), so IL-learned
    action weights are preserved and fine-tuned during RL.
    """

    MERGED_DIM = 544  # 512 (visual) + 32 (ego) — fixed across all IL archs

    def __init__(self, il_model: nn.Module):
        super().__init__()
        self.il_model = il_model
        self._has_dual_heads = hasattr(il_model, "steer_head") and hasattr(il_model, "accel_head")

        self.value_head = nn.Sequential(
            nn.Linear(self.MERGED_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Start close to deterministic IL behaviour (std ≈ 0.05); let PPO widen it if needed.
        self.log_std = nn.Parameter(torch.ones(2) * -3.0)

    # ------------------------------------------------------------------
    def _get_merged(self, image: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        """Return 544-d shared feature from the IL backbone."""
        return self.il_model(image, ego, return_features=True)

    def _get_actor_output(self, merged: torch.Tensor) -> torch.Tensor:
        """Map 544-d features → (B, 2) action mean."""
        if self._has_dual_heads:
            steer = self.il_model.steer_head(merged)   # (B, 1)
            accel = self.il_model.accel_head(merged)   # (B, 1)
            return torch.cat([steer, accel], dim=1)    # (B, 2)
        else:
            # DrivingPolicyNet uses a single fc_out(544→2)
            return self.il_model.fc_out(merged)        # (B, 2)

    # ------------------------------------------------------------------
    def forward(self, image: torch.Tensor, ego: torch.Tensor):
        """Return (action_mean (B,2), value (B,))."""
        merged = self._get_merged(image, ego)
        action_mean = self._get_actor_output(merged)
        value = self.value_head(merged).squeeze(-1)
        return action_mean, value

    def get_action_and_value(
        self,
        image: torch.Tensor,
        ego: torch.Tensor,
        action: torch.Tensor = None,
    ):
        """
        Sample an action (or evaluate a given one) and return PPO quantities.

        Returns:
            action      (B, 2)  — sampled or provided
            log_prob    (B,)    — sum of log-probs over action dimensions
            entropy     (B,)    — sum of entropies over action dimensions
            value       (B,)    — critic estimate
        """
        action_mean, value = self.forward(image, ego)
        std = self.log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value

    def get_value(self, image: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        """Return critic value (B,) without sampling an action."""
        merged = self._get_merged(image, ego)
        return self.value_head(merged).squeeze(-1)

    # ------------------------------------------------------------------
    def load_from_il_checkpoint(self, path: str, device: str = "cpu"):
        """
        Load IL model weights from an IL training checkpoint into self.il_model.

        The value_head and log_std are NOT in the IL checkpoint, so they keep
        their random initialisation — that's expected and correct for IL→RL.
        strict=False prevents a crash on those missing keys.
        """
        ckpt = torch.load(path, map_location=device)

        if isinstance(ckpt, dict):
            if "model" in ckpt:
                state = ckpt["model"]
                print(f"[IL→RL] Loading full IL checkpoint (epoch={ckpt.get('epoch','?')}, "
                      f"val_loss={ckpt.get('val_loss', float('nan')):.4f}): {path}")
            elif "policy" in ckpt:
                state = ckpt["policy"]
                print(f"[IL→RL] Loading legacy 'policy' checkpoint: {path}")
            else:
                state = ckpt
                print(f"[IL→RL] Loading raw state-dict: {path}")
        else:
            raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

        result = self.il_model.load_state_dict(state, strict=False)

        n_loaded = len(state) - len(result.missing_keys)
        print(f"[IL→RL] Matched {n_loaded}/{len(state)} parameter tensors.")
        if result.missing_keys:
            print(f"[IL→RL] Missing (randomly init'd): {result.missing_keys}")
        if result.unexpected_keys:
            print(f"[IL→RL] Unexpected (ignored):      {result.unexpected_keys}")
        print("[IL→RL] value_head and log_std start from random init — this is correct.\n")
