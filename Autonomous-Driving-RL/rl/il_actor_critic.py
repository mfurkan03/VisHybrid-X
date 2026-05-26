"""
IL Actor-Critic Wrapper
=======================
Wraps any IL policy (ImpalaNet, ImpalaNetV2, DrivingPolicyNet) as a PPO
actor-critic using a MultivariateNormal action distribution.

The IL model outputs action means directly in [-1, 1] via its action_head
(Linear -> Tanh).  ILActorCritic adds:
  - action_var  : plain buffer (not a learned param) -- diagonal of the cov matrix
  - cov_matrix  : plain buffer -- diag(action_var), rebuilt by set_action_std()
  - value_head  : Linear(544 -> 128 -> ReLU -> 1) -- new, random init (RL critic)

Covariance matrix
-----------------
Not a learned parameter -- it is a fixed tensor manually decayed during training
via set_action_std(std) as the policy improves and less exploration is needed.
The matrix is diagonal: cov = diag(std^2, std^2).

IL->RL workflow
--------------
    il_model = build_policy("impala", image_size=84)
    policy   = ILActorCritic(il_model, init_action_std=0.6).to(device)
    policy.load_from_il_checkpoint("models/policy_model_best.pth", device)
    # Fine-tune with PPO. Decay exploration periodically:
    policy.set_action_std(0.4)   # after N updates
    policy.set_action_std(0.2)   # after more updates
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class ILActorCritic(nn.Module):
    """
    PPO actor-critic that uses an IL backbone for feature extraction.

    The IL backbone exposes a 544-d merged feature (512 visual + 32 ego)
    via forward(..., return_features=True).  Its action_head (Tanh output)
    gives the action mean in [-1, 1] and is fine-tuned during RL at backbone_lr.

    ILActorCritic adds:
      action_var  : buffer (2,) -- per-dim variance, set manually via set_action_std()
      cov_matrix  : buffer (2, 2) -- diag(action_var), used to build MultivariateNormal
      value_head  : Linear(544 -> 128 -> ReLU -> 1) -- new, random init (RL critic)
    """

    MERGED_DIM = 544  # 512 (visual) + 32 (ego) -- fixed across all IL archs
    ACTION_DIM = 2

    def __init__(self, il_model: nn.Module, init_action_std: float = 0.6):
        super().__init__()
        self.il_model = il_model

        # Covariance matrix -- plain buffers, not learned parameters.
        # Decay manually with set_action_std() as policy improves.
        action_var = torch.full((self.ACTION_DIM,), init_action_std ** 2)
        self.register_buffer("action_var",  action_var)
        self.register_buffer("cov_matrix",  torch.diag(action_var))

        # Only new head -- not present in IL checkpoints, starts from random init.
        self.value_head = nn.Sequential(
            nn.Linear(self.MERGED_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    # ------------------------------------------------------------------
    def set_action_std(self, new_action_std: float):
        """Decay exploration by updating the covariance matrix in-place."""
        self.action_var.fill_(new_action_std ** 2)
        self.cov_matrix.copy_(torch.diag(self.action_var))
        print(f"[ILActorCritic] action_std -> {new_action_std:.4f}  "
              f"(var={new_action_std**2:.4f})")

    # ------------------------------------------------------------------
    def _get_merged(self, image: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        """Return 544-d shared feature from the IL backbone."""
        return self.il_model(image, ego, return_features=True)

    def _get_dist(self, merged: torch.Tensor) -> MultivariateNormal:
        """Build MultivariateNormal from separate steer/accel heads and fixed cov_matrix."""
        action_mean = torch.cat([
            self.il_model.steer_head(merged),
            self.il_model.accel_head(merged),
        ], dim=-1)  # (B, 2)
        return MultivariateNormal(action_mean, self.cov_matrix)

    # ------------------------------------------------------------------
    def forward(self, image: torch.Tensor, ego: torch.Tensor):
        """Return (dist, value (B,)) -- used internally."""
        merged = self._get_merged(image, ego)
        return self._get_dist(merged), self.value_head(merged).squeeze(-1)

    def get_action_and_value(
        self,
        image: torch.Tensor,
        ego: torch.Tensor,
        action: torch.Tensor = None,
    ):
        """
        Sample an action (or evaluate a given one) and return PPO quantities.

        action -- if provided, must be in [-1, 1] (as stored in the buffer).
                  When None, a fresh sample is drawn and clamped.

        Returns
        -------
        action      (B, 2)  in [-1, 1]
        log_prob    (B,)    -- MultivariateNormal log_prob (joint over both dims)
        entropy     (B,)    -- distribution entropy
        value       (B,)    -- critic estimate
        """
        merged = self._get_merged(image, ego)
        dist   = self._get_dist(merged)
        value  = self.value_head(merged).squeeze(-1)

        if action is None:
            action = dist.sample().clamp(-1.0, 1.0)

        log_prob = dist.log_prob(action)   # (B,) -- MultivariateNormal sums over dims
        entropy  = dist.entropy()          # (B,)
        return action, log_prob, entropy, value

    def get_value(self, image: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        """Return critic value (B,) without sampling an action."""
        merged = self._get_merged(image, ego)
        return self.value_head(merged).squeeze(-1)

    def act_deterministic(self, image: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        """Return the action mean in [-1, 1] -- for deterministic inference."""
        merged = self._get_merged(image, ego)
        return torch.cat([
            self.il_model.steer_head(merged),
            self.il_model.accel_head(merged),
        ], dim=-1)

    # ------------------------------------------------------------------
    def load_from_il_checkpoint(self, path: str, device: str = "cpu"):
        """
        Load IL model weights from an IL training checkpoint into self.il_model.

        The IL checkpoint contains the full il_model state (backbone CNN,
        ego_fc, action_head).  value_head, action_var, and cov_matrix are NOT
        in the IL checkpoint so they keep their init values -- correct for IL->RL.
        strict=False prevents a crash on those missing keys.
        """
        ckpt = torch.load(path, map_location=device)

        if isinstance(ckpt, dict):
            if "model" in ckpt:
                state = ckpt["model"]
                print(f"[IL->RL] Loading full IL checkpoint (epoch={ckpt.get('epoch','?')}, "
                      f"val_loss={ckpt.get('val_loss', float('nan')):.4f}): {path}")
            elif "policy" in ckpt:
                state = ckpt["policy"]
                print(f"[IL->RL] Loading legacy 'policy' checkpoint: {path}")
            else:
                state = ckpt
                print(f"[IL->RL] Loading raw state-dict: {path}")
        else:
            raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

        result = self.il_model.load_state_dict(state, strict=False)

        n_loaded = len(state) - len(result.missing_keys)
        print(f"[IL->RL] Matched {n_loaded}/{len(state)} parameter tensors.")
        if result.missing_keys:
            print(f"[IL->RL] Missing (randomly init'd): {result.missing_keys}")
        if result.unexpected_keys:
            print(f"[IL->RL] Unexpected (ignored):      {result.unexpected_keys}")
        print("[IL->RL] value_head starts from random init -- this is correct.\n")
