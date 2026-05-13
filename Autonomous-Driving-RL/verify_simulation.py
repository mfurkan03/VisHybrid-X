#!/usr/bin/env python3
"""
Observation Pipeline Verification
===================================
Runs a trained policy in the MetaDrive simulation for N steps and verifies
that the observation preprocessing exactly matches what the model saw during
training, using a reference file produced by `train_rl.py --save_obs_ref`.

Use this whenever you change env_wrapper.py, upgrade DPT weights, or add any
optimisation that touches the image pipeline, to confirm the model still sees
the same distribution of inputs.

Examples
--------
# 1. During training, save a reference:
    python train_rl.py --il_checkpoint ../models/policy_model_best.pth \\
        --save_obs_ref obs_ref.json --timesteps 200000

# 2. Verify simulation matches (with or without an existing reference):
    python verify_simulation.py --checkpoint models/rl/policy_best.pth \\
        --obs_ref obs_ref.json

# 3. Just print obs stats (no comparison):
    python verify_simulation.py --checkpoint models/rl/policy_best.pth \\
        --steps 200
"""
import sys
import os
from pathlib import Path

sys.modules['xformers'] = None  # prevent xformers CPU crash (must be first)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import torch

from src.models import build_policy, EGO_DIM
from rl.env_wrapper import MetaDriveRLWrapper
from rl.il_actor_critic import ILActorCritic
from rl.obs_verifier import ObsVerifier


def parse_args():
    p = argparse.ArgumentParser(description="Verify simulation obs pipeline vs training")
    # ── Checkpoint ────────────────────────────────────────────────────────────
    p.add_argument("--checkpoint",    type=str, default=None,
                   help="RL checkpoint (policy key)")
    p.add_argument("--il_checkpoint", type=str, default=None,
                   help="IL checkpoint (model key) — load IL policy into RL wrapper")
    p.add_argument("--arch",          type=str, default="impala",
                   choices=["simple", "impala", "impala_v2"])
    p.add_argument("--image_size",    type=int, default=84)
    p.add_argument("--dpt_path",      type=str, default=None)
    # ── Verification ──────────────────────────────────────────────────────────
    p.add_argument("--steps",         type=int, default=500,
                   help="Number of env steps to collect for statistics")
    p.add_argument("--obs_ref",       type=str, default=None, metavar="PATH",
                   help="Reference JSON saved by train_rl.py --save_obs_ref. "
                        "If omitted, stats are printed but no comparison is made.")
    p.add_argument("--tol",           type=float, default=0.10,
                   help="Relative tolerance for mean/std mismatch (default 0.10 = 10%%)")
    p.add_argument("--save_stats",    type=str, default=None, metavar="PATH",
                   help="Optionally save the simulation obs stats to a new JSON file")
    # ── Env ───────────────────────────────────────────────────────────────────
    p.add_argument("--scenarios",     type=int, default=5)
    p.add_argument("--render",        action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.checkpoint and not args.il_checkpoint:
        print("Error: supply --checkpoint (RL) or --il_checkpoint (IL).")
        return

    # ── Load policy ───────────────────────────────────────────────────────────
    il_model = build_policy(args.arch, args.image_size)
    policy   = ILActorCritic(il_model).to(device)

    if args.il_checkpoint:
        if not os.path.exists(args.il_checkpoint):
            print(f"IL checkpoint not found: {args.il_checkpoint}")
            return
        policy.load_from_il_checkpoint(args.il_checkpoint, device)
        print("[Mode] IL policy running inside RL environment (deterministic)")
    else:
        if not os.path.exists(args.checkpoint):
            print(f"Checkpoint not found: {args.checkpoint}")
            return
        ckpt = torch.load(args.checkpoint, map_location=device)
        if isinstance(ckpt, dict) and "policy" in ckpt:
            policy.load_state_dict(ckpt["policy"])
            print(f"Loaded RL checkpoint (step={ckpt.get('global_step','?')})")
        else:
            policy.load_state_dict(ckpt)
            print("Loaded raw state-dict")

    policy.eval()

    # ── Environment ───────────────────────────────────────────────────────────
    env = MetaDriveRLWrapper(
        env_config={
            "use_render": args.render,
            "show_interface": args.render,
            "num_scenarios": args.scenarios,
        },
        show_perception=args.render,
        image_size=args.image_size,
        dpt_path=args.dpt_path,
    )

    # ── Collect steps ─────────────────────────────────────────────────────────
    verifier = ObsVerifier()
    obs      = env.reset()
    print(f"\nCollecting {args.steps} steps for obs statistics ...\n")

    for step in range(args.steps):
        img, ego = obs
        verifier.record(img, ego)

        obs_img = torch.from_numpy(img).unsqueeze(0).to(device)
        obs_ego = torch.from_numpy(ego).unsqueeze(0).to(device)

        with torch.inference_mode():
            action_mean, _ = policy(obs_img, obs_ego)

        action = np.clip(action_mean.squeeze(0).cpu().numpy(), -1.0, 1.0)
        obs, _, done, _ = env.step(action)

        if done:
            obs = env.reset()

        if (step + 1) % 100 == 0:
            print(f"  {step+1}/{args.steps} steps collected")

    env.close()

    # ── Report ────────────────────────────────────────────────────────────────
    verifier.print_stats()

    if args.save_stats:
        verifier.save(args.save_stats)

    if args.obs_ref:
        if not os.path.exists(args.obs_ref):
            print(f"[ObsVerifier] Reference file not found: {args.obs_ref}")
            print("  Run training with --save_obs_ref to create one.")
            return
        ok = verifier.compare(args.obs_ref, tol=args.tol)
        sys.exit(0 if ok else 1)
    else:
        print("\nNo --obs_ref supplied — stats printed above but no comparison made.")
        print("Pass --obs_ref PATH to compare against your training reference.\n")


if __name__ == "__main__":
    main()
