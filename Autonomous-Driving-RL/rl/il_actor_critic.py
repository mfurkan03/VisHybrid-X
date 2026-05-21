"""
IL Actor-Critic Wrapper
=======================
Wraps any IL policy (ImpalaNet, ImpalaNetV2, DrivingPolicyNet) as a PPO
actor-critic using a Beta action distribution.

Beta distribution rationale
---------------------------
Gaussian (Normal) is unbounded: samples can fall outside [-1, 1], and
clamping corrupts log-prob / reward attribution.  Beta is naturally
supported on (0, 1).  We sample in [0, 1] and map to [-1, 1] only when
stepping the environment; the buffer stores raw [0, 1] samples so
log-prob re-evaluation during the PPO update is exact.

IL→RL workflow
--------------
    il_model = build_policy("impala", image_size=84)
    policy   = ILActorCritic(il_model).to(device)
    policy.load_from_il_checkpoint("models/policy_model_best.pth", device)
    # Then fine-tune with PPO as normal.

Distribution heads
------------------
The IL model owns the Beta distribution heads (steer_alpha_head,
steer_beta_head, throttle_mu_head, throttle_nu_head) and outputs (alpha, beta)
directly.  ILActorCritic delegates distribution computation to those trained
heads via _get_dist(), and only adds a new value_head for the RL critic.
This ensures IL-trained distribution knowledge is preserved at the start of RL.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
from torch.distributions import Beta


class ILActorCritic(nn.Module):
    """
    PPO actor-critic that uses an IL backbone for feature extraction.

    The IL backbone exposes a 544-d merged feature (512 visual + 32 ego)
    via forward(..., return_features=True).  Its distribution heads
    (steer_alpha_head, steer_beta_head, throttle_mu_head, throttle_nu_head)
    are reused directly — they carry trained IL weights and are fine-tuned
    during RL at backbone_lr.

    ILActorCritic adds only:
      value_head : Linear(544 → 128 → ReLU → 1) — new, random init (RL critic)

    Actions are sampled from Beta(α, β) ∈ (0, 1) and scaled to [-1, 1]
    before being sent to the environment.  The buffer stores the raw [0, 1]
    samples so log-prob re-evaluation during the PPO update is exact.

    CONCENTRATION_SCALE multiplies alpha and beta in _get_dist() to reduce
    sampling variance without changing the distribution mean.  Applied
    identically during rollout sampling and PPO log-prob evaluation so the
    policy gradient remains correct.  Throttle needs this more than steer
    because its concentration (= nu) can be as low as 2.0 at IL init,
    yielding std ≈ 0.5 in [-1, 1] — too noisy for consistent forward motion.
    """

    MERGED_DIM = 544  # 512 (visual) + 32 (ego) — fixed across all IL archs
    CONCENTRATION_SCALE = 1.0  # tighter distribution; mean unchanged, std / sqrt(3)

    def __init__(self, il_model: nn.Module):
        super().__init__()
        self.il_model = il_model

        # Only new head — not present in IL checkpoints, starts from random init.
        self.value_head = nn.Sequential(
            nn.Linear(self.MERGED_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    # ------------------------------------------------------------------
    def _get_merged(self, image: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        """Return 544-d shared feature from the IL backbone."""
        return self.il_model(image, ego, return_features=True)

    def _get_dist(self, merged: torch.Tensor) -> Beta:
        """Build Beta distribution by delegating to the IL model's trained heads.

        Steer  : α, β > 2 via steer_alpha_head / steer_beta_head.
        Throttle: mu/nu reparameterization — α = mu*nu, β = (1-mu)*nu.
        """
        il = self.il_model
        steer_alpha = il.steer_alpha_head(merged) + 2.0   # (B, 1), > 2
        steer_beta  = il.steer_beta_head(merged)  + 2.0   # (B, 1), > 2

        mu             = il.throttle_mu_head(merged).clamp(1e-6, 1.0 - 1e-6)  # (B, 1); clamp prevents sigmoid saturation → exact 0/1 → zero concentration
        # TODO: gradually relax max clamp (e.g. to 50.0) after ~100k PPO steps
        nu             = torch.clamp(il.throttle_nu_head(merged) + 2.0, min=2.0, max=10.0)  # (B, 1)
        throttle_alpha = mu * nu
        throttle_beta  = (1.0 - mu) * nu

        alpha = torch.cat([steer_alpha, throttle_alpha], dim=-1) * self.CONCENTRATION_SCALE  # (B, 2)
        beta  = torch.cat([steer_beta,  throttle_beta],  dim=-1) * self.CONCENTRATION_SCALE  # (B, 2)
        # Clamp concentration to [1e-4, 50]: prevents float32 lgamma overflow at high values
        # and ensures Beta log_prob is numerically stable. Values above ~50 are near-deterministic
        # anyway — clamping here doesn't meaningfully restrict policy expressiveness.
        alpha = alpha.clamp(1e-4, 50.0)
        beta  = beta.clamp(1e-4, 50.0)
        return Beta(alpha, beta)

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

        action — if provided, must be in [0, 1] (as stored in the buffer).
                 When None, a fresh sample is drawn from Beta.

        Returns
        -------
        action      (B, 2)  in [0, 1] — sampled or provided
        log_prob    (B,)    — sum of log-probs over action dimensions
        entropy     (B,)    — sum of entropies over action dimensions
        value       (B,)    — critic estimate
        """
        merged = self._get_merged(image, ego)
        dist   = self._get_dist(merged)
        value  = self.value_head(merged).squeeze(-1)

        if action is None:
            action = dist.sample()   # (B, 2) in (0, 1)

        # Clamp stored actions away from boundaries before log_prob
        log_prob = dist.log_prob(action.clamp(1e-6, 1.0 - 1e-6)).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value

    def get_value(self, image: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        """Return critic value (B,) without sampling an action."""
        merged = self._get_merged(image, ego)
        return self.value_head(merged).squeeze(-1)

    def act_deterministic(self, image: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        """
        Return the Beta mean mapped to [-1, 1] — for deterministic inference.

        mean = α / (α + β).  Less extreme than mode; matches the behaviour of
        the old smooth-L1 regression which also learned the conditional mean.
        """
        merged = self._get_merged(image, ego)
        dist   = self._get_dist(merged)
        alpha  = dist.concentration1   # (B, 2)
        beta_  = dist.concentration0   # (B, 2)
        mean   = alpha / (alpha + beta_)
        return mean * 2.0 - 1.0        # (B, 2) in [-1, 1]

    # ------------------------------------------------------------------
    def reset_nu_heads_for_ppo(self):
        """
        Reset distribution heads to maximum-entropy state before PPO fine-tuning.

        Diagnostic on the IL checkpoint revealed two problems:
          1. throttle_nu_head — already fixed: nu reset to ~2.69 (near-minimum
             concentration), giving a broad Beta.
          2. throttle_mu_head — IL training drove the mean to ~0.84 (heavy
             forward-throttle bias).  At nu=2 this gives Beta(1.68, 0.32),
             which is extremely right-skewed with entropy ≈ -8.5.  PPO cannot
             explore braking or coasting from this starting point.

        Fix for mu_head
        ---------------
        Only the LAST Linear layer is reset (weight → N(0, 0.01), bias → 0.0).
        sigmoid(0) = 0.5, so the output centres at 0.5 after reset.
        Earlier layers in the Sequential are left untouched: they may encode
        useful IL feature projections.  (In practice throttle_mu_head is a
        single Linear+Sigmoid, so this resets the whole head anyway.)

        WHY steer_alpha_head / steer_beta_head are NOT reset
        -----------------------------------------------------
        Steer has no separate mu/nu split — alpha_head and beta_head jointly
        encode both mean steering direction AND concentration.  Resetting them
        would erase IL-learned directional behaviour, which must be preserved.

        Expected state after reset (any input batch)
        --------------------------------------------
          mu  ≈ 0.50  (std ~0.05)
          nu  ≈ 2.69  (just above the +2.0 minimum floor)
          Beta(~1, ~1) → entropy ≈ 0.0  (maximum for this parameterisation)
        """
        il = self.il_model
        print("[reset_nu_heads_for_ppo] Resetting distribution heads:")

        # ── nu head ───────────────────────────────────────────────────────────
        for layer_idx, layer in enumerate(il.throttle_nu_head):
            if isinstance(layer, nn.Linear):
                # zeros → linear_output = 0 for any features
                # Softplus(0) = ln(2) ≈ 0.693, so nu = 0.693 + 2.0 ≈ 2.69 (exact)
                nn.init.zeros_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
                print(f"  throttle_nu_head[{layer_idx}] Linear  "
                      f"weight={layer.weight.abs().max().item():.6f}  bias={layer.bias.mean().item():.4f}")

        # ── mu head: ONLY the last Linear layer ───────────────────────────────
        # N(0, small_std) is NOT sufficient — IL backbone features have large
        # L2 norm (||merged|| ≈ 150+, ImpalaNetV2), so even std=0.01 weights
        # produce pre-sigmoid outputs of ±1.5+, pushing mu away from 0.5.
        # zeros guarantees linear_output = 0 for every observation → sigmoid(0) = 0.5 exact.
        last_linear = None
        for layer in il.throttle_mu_head.modules():
            if isinstance(layer, nn.Linear):
                last_linear = layer
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            nn.init.constant_(last_linear.bias, 0.0)
            print(f"  throttle_mu_head (last Linear)  "
                  f"weight={last_linear.weight.abs().max().item():.6f}  bias={last_linear.bias.mean().item():.4f}")

        print("[reset_nu_heads_for_ppo] Done — mu≈0.50  nu≈2.69  entropy≈0.0\n")

    # ------------------------------------------------------------------
    def load_from_il_checkpoint(self, path: str, device: str = "cpu"):
        """
        Load IL model weights from an IL training checkpoint into self.il_model.

        The IL checkpoint contains the full il_model state (backbone CNN,
        ego_fc, steer_alpha_head, steer_beta_head, throttle_mu_head,
        throttle_nu_head).  value_head is NOT in the IL checkpoint, so it
        keeps its random initialisation — that is correct for IL→RL.
        strict=False prevents a crash on that missing key.
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
        print("[IL→RL] value_head starts from random init — this is correct.\n")
