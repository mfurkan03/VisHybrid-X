"""
Steer-entropy diagnostic
========================
Verifies two things after Option A (steer_concentration_scale):

  a) Scale is working — steer entropy increases after reset.
     Also reports whether the scale is enough (entropy close to Beta(1,1) = 0.0).

  b) IL knowledge is preserved — steer means are identical before and after reset
     (max absolute mean difference should be < 1e-5).

Usage
-----
    python diagnose_steer_entropy.py --checkpoint ../models/policy_model_best.pth --arch impala

No MetaDrive required; uses random synthetic observations.
"""
import sys
import math
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.distributions import Beta

from src.models import build_policy, EGO_DIM
from rl.il_actor_critic import ILActorCritic


# ── helpers ───────────────────────────────────────────────────────────────────

def beta_stats(alpha: torch.Tensor, beta_: torch.Tensor):
    """Return per-dim (mean_01, std_01, entropy) averaged over batch."""
    mean   = (alpha / (alpha + beta_)).mean(0)           # (2,)
    var    = (alpha * beta_ / ((alpha + beta_)**2 * (alpha + beta_ + 1))).mean(0)
    std    = var.sqrt()
    entropy = Beta(alpha, beta_).entropy().mean(0)        # (2,)
    return mean, std, entropy


def get_raw_concentrations(policy: ILActorCritic, imgs: torch.Tensor, egos: torch.Tensor):
    """Return (steer_alpha, steer_beta) BEFORE the concentration scale is applied."""
    il = policy.il_model
    merged = policy._get_merged(imgs, egos)
    sa = il.steer_alpha_head(merged) + 2.0   # (B, 1)
    sb = il.steer_beta_head(merged)  + 2.0   # (B, 1)
    return sa, sb


def report_block(tag: str, policy: ILActorCritic,
                 imgs: torch.Tensor, egos: torch.Tensor):
    with torch.no_grad():
        dist = policy._get_dist(policy._get_merged(imgs, egos))
        alpha = dist.concentration1   # (B, 2)  — after all scaling
        beta_ = dist.concentration0

        steer_a = alpha[:, 0:1]
        steer_b = beta_[:, 0:1]
        thr_a   = alpha[:, 1:2]
        thr_b   = beta_[:, 1:2]

    sm, ss, se = beta_stats(steer_a, steer_b)
    tm, ts, te = beta_stats(thr_a,   thr_b)
    total_H    = (se + te).item()

    print(f"\n{'─'*60}")
    print(f"  {tag}")
    print(f"{'─'*60}")
    print(f"  STEER    α={steer_a.mean():.3f}  β={steer_b.mean():.3f}"
          f"  mean={sm.item():.3f}  std={ss.item():.3f}  H={se.item():.4f} nats")
    print(f"  THROTTLE α={thr_a.mean():.3f}  β={thr_b.mean():.3f}"
          f"  mean={tm.item():.3f}  std={ts.item():.3f}  H={te.item():.4f} nats")
    print(f"  Total entropy/step: {total_H:.4f} nats"
          f"  (Beta(1,1) reference: 0.0000)")

    return sm.item(), se.item()   # steer mean, steer entropy


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch",       default="impala",
                        choices=["simple", "impala", "impala_v2"])
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--batch",      type=int, default=256,
                        help="Synthetic batch size (more = stabler stats)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── build policy ──────────────────────────────────────────────────────────
    il_model = build_policy(args.arch, args.image_size)
    policy   = ILActorCritic(il_model).to(device)
    policy.load_from_il_checkpoint(args.checkpoint, device)
    policy.eval()

    # ── synthetic batch (diverse random features) ─────────────────────────────
    # Random images and ego states — the backbone will produce varied merged
    # features, so steer means won't be constant (that variation IS IL knowledge).
    imgs = torch.randn(args.batch, 4, args.image_size, args.image_size,
                       device=device)
    egos = torch.randn(args.batch, EGO_DIM, device=device)

    # ── BEFORE reset ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  BEFORE reset_nu_heads_for_ppo()")
    print("="*60)
    mean_before, H_before = report_block("BEFORE", policy, imgs, egos)

    # Raw concentrations (before scale) — shows what IL trained
    with torch.no_grad():
        sa_raw, sb_raw = get_raw_concentrations(policy, imgs, egos)
    print(f"\n  Raw steer concentrations (IL, before scale):")
    print(f"    α  min={sa_raw.min():.2f}  max={sa_raw.max():.2f}  mean={sa_raw.mean():.2f}")
    print(f"    β  min={sb_raw.min():.2f}  max={sb_raw.max():.2f}  mean={sb_raw.mean():.2f}")
    print(f"    mean range = [{(sa_raw/(sa_raw+sb_raw)).min():.3f}"
          f", {(sa_raw/(sa_raw+sb_raw)).max():.3f}]  "
          f"(spread shows IL directional knowledge)")

    # ── apply reset ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Calling reset_nu_heads_for_ppo()...")
    print("="*60)
    policy.reset_nu_heads_for_ppo()

    # ── AFTER reset ───────────────────────────────────────────────────────────
    mean_after, H_after = report_block("AFTER", policy, imgs, egos)

    # ── verification ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  VERIFICATION")
    print("="*60)

    mean_diff = abs(mean_after - mean_before)
    print(f"\n  (a) Entropy change:")
    print(f"      H before : {H_before:.4f} nats")
    print(f"      H after  : {H_after:.4f} nats  (higher = more exploration)")
    print(f"      H target : 0.0000 nats  (Beta(1,1) maximum)")
    gap = abs(H_after - 0.0)
    if gap < 0.05:
        print(f"      ✓ Scale is sufficient — entropy within 0.05 nats of maximum.")
    elif gap < 0.3:
        print(f"      ~ Scale is close but not maximum (gap={gap:.3f} nats).")
        print(f"        IL concentrations may be higher than zero-init baseline.")
        print(f"        Consider a lower steer_concentration_scale if entropy is still too low.")
    else:
        print(f"      ✗ Scale is NOT sufficient (gap={gap:.3f} nats).")
        print(f"        IL concentrations are very high — try scale ≈ {1.0/(sa_raw.mean().item()/2.69*2.69):.3f}")

    print(f"\n  (b) IL knowledge preserved (steer means):")
    print(f"      mean before: {mean_before:.6f}")
    print(f"      mean after : {mean_after:.6f}")
    print(f"      max |Δmean|: {mean_diff:.2e}")
    if mean_diff < 1e-4:
        print(f"      ✓ IL directional knowledge intact.")
    else:
        print(f"      ✗ Means shifted — check _get_dist() scale application.")

    # Steer mean range after reset — should still be non-trivial (not all ~0.5)
    with torch.no_grad():
        merged = policy._get_merged(imgs, egos)
        dist_after = policy._get_dist(merged)
        means_after = (dist_after.concentration1 /
                       (dist_after.concentration1 + dist_after.concentration0))[:, 0]
    spread = means_after.max().item() - means_after.min().item()
    print(f"\n      Steer mean spread across batch: {spread:.3f}")
    if spread > 0.05:
        print(f"      ✓ Backbone drives varied steer means — IL knowledge active.")
    else:
        print(f"      ✗ Steer means collapsed to near-constant — backbone not driving steer.")

    print()


if __name__ == "__main__":
    main()
