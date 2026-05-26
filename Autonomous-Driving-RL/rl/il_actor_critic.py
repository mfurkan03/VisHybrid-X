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

    The IL backbone exposes a merged feature (512 visual + ego-encoder out_dim,
    = 560 with the dedicated nav branch; read from il_model.merged_dim) via
    forward(..., return_features=True).  Its distribution heads
    (steer_alpha_head, steer_beta_head, throttle_mu_head, throttle_nu_head)
    are reused directly — they carry trained IL weights and are fine-tuned
    during RL at backbone_lr.

    ILActorCritic adds only:
      value_head : Linear(merged_dim → 128 → ReLU → 1) — new, random init (RL critic)

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

    MERGED_DIM = 544  # legacy fallback only; the real size is read from
                      # il_model.merged_dim (512 visual + ego-encoder out_dim).
                      # With the dedicated nav branch the ego encoder emits 48,
                      # so current backbones report merged_dim = 560.
    CONCENTRATION_SCALE = 1.0  # tighter distribution; mean unchanged, std / sqrt(3)

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

    # ------------------------------------------------------------------
    def _get_merged(self, image: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        """Return 544-d shared feature from the IL backbone."""
        return self.il_model(image, ego, return_features=True)

    def _get_dist(self, merged: torch.Tensor) -> Beta:
        """Build Beta distribution from mu/nu heads for both steer and throttle.

        Both actions use mu/nu reparameterization: α = mu*nu, β = (1-mu)*nu.
        This decouples direction (mu) from concentration (nu), allowing
        reset_nu_heads_for_ppo() to maximize entropy without losing IL-learned
        directional knowledge.
        """
        il = self.il_model

        mu_s = il.steer_mu_head(merged).clamp(1e-6, 1.0 - 1e-6)     # (B, 1)
        # TODO: gradually relax max clamp (e.g. to 50.0) after ~100k PPO steps
        nu_s = torch.clamp(il.steer_nu_head(merged) + 2.0, min=2.0, max=10.0)  # (B, 1)

        mu_t = il.throttle_mu_head(merged).clamp(1e-6, 1.0 - 1e-6)   # (B, 1)
        nu_t = torch.clamp(il.throttle_nu_head(merged) + 2.0, min=2.0, max=10.0)  # (B, 1)

        alpha = torch.cat([mu_s * nu_s, mu_t * nu_t], dim=-1) * self.CONCENTRATION_SCALE  # (B, 2)
        beta  = torch.cat([(1.0 - mu_s) * nu_s, (1.0 - mu_t) * nu_t], dim=-1) * self.CONCENTRATION_SCALE  # (B, 2)
        # Clamp concentration to [1e-4, 50]: prevents float32 lgamma overflow at high values
        # and ensures Beta log_prob is numerically stable.
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
        Reset all nu heads to maximum-entropy state before PPO fine-tuning.

        With the new mu/nu parameterization for both steer and throttle, this
        is now clean and symmetric: nu heads control concentration only, mu
        heads control direction only.

        Nu heads (steer_nu_head, throttle_nu_head)
        -------------------------------------------
        Reset to zeros: linear_output = 0 → Softplus(0) = ln(2) ≈ 0.693
        → nu = 2.693 (minimum reachable given the +2.0 floor).
        Gives Beta(mu*2.69, (1-mu)*2.69) ≈ near-uniform, entropy ≈ -0.23.

        Mu heads (steer_mu_head, throttle_mu_head)
        -------------------------------------------
        Also reset to zeros: sigmoid(0) = 0.5 exactly for any input.
        IL-learned directional knowledge is deliberately discarded here —
        it was encoded in high concentration (nu >> 2), not in mu.  With
        the old alpha/beta parameterization this was a problem; with mu/nu
        the directional info lives in mu which PPO will re-learn quickly
        from the backbone features (which ARE preserved).

        Expected state after reset
        --------------------------
          steer  : mu=0.50, nu=2.69 → Beta(1.35, 1.35), entropy ≈ -0.23
          throttle: mu=0.50, nu=2.69 → Beta(1.35, 1.35), entropy ≈ -0.23
          total entropy ≈ -0.46  (vs -8 before)
        """
        il = self.il_model
        print("[reset_nu_heads_for_ppo] Resetting all mu/nu heads to maximum-entropy state:")

        nu_heads = [("steer_nu_head",    il.steer_nu_head),
                    ("throttle_nu_head", il.throttle_nu_head)]
        mu_heads = [("steer_mu_head",    il.steer_mu_head),
                    ("throttle_mu_head", il.throttle_mu_head)]

        for head_name, head in nu_heads + mu_heads:
            last_linear = None
            for layer in head.modules():
                if isinstance(layer, nn.Linear):
                    last_linear = layer
            if last_linear is not None:
                nn.init.zeros_(last_linear.weight)
                nn.init.constant_(last_linear.bias, 0.0)
                print(f"  {head_name:25s}  weight_max={last_linear.weight.abs().max().item():.6f}  bias={last_linear.bias.mean().item():.4f}")

        print("[reset_nu_heads_for_ppo] Done — mu=0.50 exact, nu=2.69 exact, entropy≈-0.46\n")

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
