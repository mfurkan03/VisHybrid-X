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
import math
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
    policy gradient remains correct.

    steer_concentration_scale is a registered buffer (saved in checkpoints).
    It is 1.0 after IL loading and set to _STEER_SCALE_AT_RESET by
    reset_nu_heads_for_ppo().  It only multiplies the steer concentrations,
    leaving throttle untouched.  The ratio α/β is preserved, so steer means
    are identical before and after reset — only concentration (sharpness) drops.
    """

    MERGED_DIM = 544  # 512 (visual) + 32 (ego) — fixed across all IL archs
    CONCENTRATION_SCALE = 1.0

    # softplus(0) + 2.0 = ln(2) + 2.0 ≈ 2.693 — the steer concentration at zero-init.
    # This scale brings it to exactly 1.0 (Beta(1,1) = uniform = maximum entropy).
    _STEER_SCALE_AT_RESET: float = 1.0 / (math.log(2) + 2.0)  # ≈ 0.372

    def __init__(self, il_model: nn.Module):
        super().__init__()
        self.il_model = il_model

        # Saved in state_dict so RL checkpoint resumes use the correct scale.
        self.register_buffer("steer_concentration_scale", torch.tensor(1.0))

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

        steer_alpha = steer_alpha * self.steer_concentration_scale
        steer_beta  = steer_beta  * self.steer_concentration_scale

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

        Throttle
        --------
        IL training drove mu → ~0.84 (heavy forward bias), giving Beta(1.68, 0.32)
        with entropy ≈ -8.5 — PPO cannot explore braking from this start.
        Fix: zero the last Linear in mu_head (sigmoid(0)=0.5) and all of nu_head
        (softplus(0)+2=2.69 → throttle_alpha=throttle_beta≈1.35 → Beta(1.35,1.35)).

        Steer
        -----
        IL training drives steer concentrations to ~700+ (Smooth-L1 loss has no
        entropy penalty — the model becomes extremely confident).  The 50.0 clamp
        in _get_dist() then caps all outputs at 50 regardless of any scale, so a
        scale alone cannot change the entropy.

        Additionally, the IL steer mean spread across observations is near-trivial
        (~0.09 in [0,1]) — the heads learned "steer neutral with extreme confidence"
        rather than meaningful directional variation.  There is little IL directional
        knowledge worth preserving in the heads themselves.

        Fix: zero-init both steer heads so every observation yields the constant
        softplus(0)+2.0 = 2.693.  The steer_concentration_scale (≈0.372) then
        brings 2.693 × 0.372 ≈ 1.0 → Beta(1,1) (maximum entropy).  Unlike
        Option-B random-init (which failed), zero-init produces a deterministic,
        well-behaved constant output; the backbone relearns steer differentiation
        via PPO gradient flow within a few updates.

        After reset
        -----------
          steer    : constant Beta(1,1) for all obs — maximum entropy
          throttle : mu≈0.50  nu≈2.69  Beta(~1.35, ~1.35)
        """
        il = self.il_model
        print("[reset_nu_heads_for_ppo] Resetting distribution heads:")

        # ── throttle nu head ──────────────────────────────────────────────────
        for layer_idx, layer in enumerate(il.throttle_nu_head):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
                print(f"  throttle_nu_head[{layer_idx}] Linear  "
                      f"weight={layer.weight.abs().max().item():.6f}  bias={layer.bias.mean().item():.4f}")

        # ── throttle mu head (last Linear only) ───────────────────────────────
        last_linear = None
        for layer in il.throttle_mu_head.modules():
            if isinstance(layer, nn.Linear):
                last_linear = layer
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            nn.init.constant_(last_linear.bias, 0.0)
            print(f"  throttle_mu_head (last Linear)  "
                  f"weight={last_linear.weight.abs().max().item():.6f}  bias={last_linear.bias.mean().item():.4f}")

        # ── steer heads: zero-init + scale ───────────────────────────────────
        # IL training drives steer concentrations to 700+ (no entropy penalty in
        # Smooth-L1 loss). The 50.0 clamp in _get_dist() then caps all of them at
        # 50 regardless of any scale, so the scale trick alone has no effect.
        # Solution: zero-init both heads so every observation yields the constant
        # softplus(0)+2.0 = 2.693, then scale × 2.693 ≈ 1.0 → Beta(1,1).
        # This is safe (unlike Option B random-init) because zero-init gives a
        # deterministic well-behaved output; the backbone quickly relearns steer
        # differentiation via PPO gradient flow after a few updates.
        for name, head in [("steer_alpha_head", il.steer_alpha_head),
                            ("steer_beta_head",  il.steer_beta_head)]:
            for i, layer in enumerate(head):
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)
                    print(f"  {name}[{i}] Linear  "
                          f"weight={layer.weight.abs().max().item():.6f}  "
                          f"bias={layer.bias.mean().item():.4f}")

        self.steer_concentration_scale.fill_(self._STEER_SCALE_AT_RESET)
        print(f"  steer_concentration_scale → {self.steer_concentration_scale.item():.4f}"
              f"  (2.693 × {self._STEER_SCALE_AT_RESET:.3f} ≈ 1.0 → Beta(1,1))")

        print("[reset_nu_heads_for_ppo] Done\n")

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
