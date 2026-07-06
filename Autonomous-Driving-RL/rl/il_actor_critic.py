"""
IL Actor-Critic Wrapper
=======================
Wraps any IL policy (ImpalaNet, ImpalaNetV2, DrivingPolicyNet) as a PPO
actor-critic using a fixed-covariance Gaussian action distribution.

Fixed-covariance rationale
---------------------------
The IL backbone's steer_head/throttle_head make a plain point-estimate
prediction (Tanh-bounded to [-1, 1]), trained with a regression loss in IL —
there is no learned notion of uncertainty. RL exploration noise is added on
top as a fixed (non-learned) Gaussian std, registered as a buffer so it is
never touched by the optimizer. Samples are unbounded (Normal has full
support); a clipped copy is sent to the environment while the raw sample is
what gets stored for exact log-prob recomputation during the PPO update.

IL→RL workflow
--------------
    il_model = build_policy("impala", image_size=84)
    policy   = ILActorCritic(il_model).to(device)
    policy.load_from_il_checkpoint("models/policy_model_best.pth", device)
    # Then fine-tune with PPO as normal.

Shared heads
------------
IL and RL now use the exact same steer_head/throttle_head — there is no
separate RL-only parameterization, so load_from_il_checkpoint needs no
weight conversion: the checkpoint's heads load directly into the RL policy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
from torch.distributions import Normal


class ILActorCritic(nn.Module):
    """
    PPO actor-critic that uses an IL backbone for feature extraction.

    The IL backbone exposes a merged feature (512 visual + ego-encoder out_dim,
    = 560 with the dedicated nav branch; read from il_model.merged_dim) via
    forward(..., return_features=True).  Its steer_head/throttle_head are
    reused directly — they carry trained IL weights and are fine-tuned
    during RL at backbone_lr.

    ILActorCritic adds only:
      value_head : Linear(merged_dim → 128 → ReLU → 1) — new, random init (RL critic)

    Actions are sampled from Normal(mean, std) where mean comes from the
    shared IL heads (already Tanh-bounded to [-1, 1]) and std is a fixed,
    non-learned buffer. The raw (unbounded) sample is stored for exact
    log-prob recomputation; a clipped copy is sent to the environment.
    """

    MERGED_DIM = 544  # legacy fallback only; the real size is read from
                      # il_model.merged_dim (512 visual + ego-encoder out_dim).
                      # With the dedicated nav branch the ego encoder emits 48,
                      # so current backbones report merged_dim = 560.
    STEER_STD    = 0.05  # fixed, non-learned RL exploration std for steer
    THROTTLE_STD = 0.05  # fixed, non-learned RL exploration std for throttle

    def __init__(self, il_model: nn.Module):
        super().__init__()
        self.il_model = il_model

        # Read the backbone's actual fused-feature width.  Older checkpoints
        # without the attribute fall back to 544 (pre-nav-branch architecture).
        merged_dim = getattr(il_model, "merged_dim", self.MERGED_DIM)

        # Only new head — not present in IL checkpoints, starts from random init.
        self.value_head = nn.Sequential(
            nn.Linear(merged_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Fixed action std — buffer, not a parameter: never touched by the optimizer.
        self.register_buffer("action_std", torch.tensor([self.STEER_STD, self.THROTTLE_STD]))

    # ------------------------------------------------------------------
    def _get_merged(self, image: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        """Return the shared feature from the IL backbone."""
        return self.il_model(image, ego, return_features=True)

    def _get_dist(self, merged: torch.Tensor) -> Normal:
        """Build a fixed-covariance Normal distribution from the shared IL heads.

        mean comes directly from steer_head/throttle_head (Tanh-bounded to
        [-1, 1], the same heads IL training fits with a regression loss).
        std is fixed and non-learned (action_std buffer).
        """
        il = self.il_model
        mean = torch.cat([il.steer_head(merged), il.throttle_head(merged)], dim=-1)  # (B, 2)
        std  = self.action_std.expand_as(mean)
        return Normal(mean, std)

    # ------------------------------------------------------------------
    def forward(self, image: torch.Tensor, ego: torch.Tensor):
        """Return (dist, value (B,)) — used internally."""
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

        action — if provided, is the raw (unbounded) sample as stored in the
                 buffer. When None, a fresh sample is drawn from Normal.

        Returns
        -------
        action      (B, 2)  raw, unbounded — sampled or provided (caller clips
                            before sending to the environment)
        log_prob    (B,)    — sum of log-probs over action dimensions
        entropy     (B,)    — sum of entropies over action dimensions (constant
                            w.r.t. all parameters since std is fixed)
        value       (B,)    — critic estimate
        """
        merged = self._get_merged(image, ego)
        dist   = self._get_dist(merged)
        value  = self.value_head(merged).squeeze(-1)

        if action is None:
            action = dist.sample()   # (B, 2), unbounded

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value

    def get_value(self, image: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        """Return critic value (B,) without sampling an action."""
        merged = self._get_merged(image, ego)
        return self.value_head(merged).squeeze(-1)

    def act_deterministic(self, image: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        """Return the Normal mean directly — already Tanh-bounded to [-1, 1]."""
        merged = self._get_merged(image, ego)
        dist   = self._get_dist(merged)
        return dist.mean

    # ------------------------------------------------------------------
    def load_from_il_checkpoint(self, path: str, device: str = "cpu"):
        """
        Load IL model weights from an IL training checkpoint into self.il_model.

        The IL checkpoint contains the full il_model state (backbone CNN,
        ego_fc, steer_head, throttle_head).  value_head is NOT in the IL
        checkpoint, so it keeps its random initialisation — that is correct
        for IL→RL. strict=False prevents a crash on that missing key. No
        weight conversion is needed: IL and RL share the exact same heads.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)

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
        print("[IL→RL] value_head starts from random init — this is correct.\n")
