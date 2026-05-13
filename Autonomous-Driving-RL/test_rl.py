#!/usr/bin/env python3
"""
RL Policy Test (deterministic rollout)
=======================================
Run from the Autonomous-Driving-RL/ directory.

Examples
--------
    python test_rl.py --checkpoint checkpoints/policy_best.pth --arch impala
    python test_rl.py --checkpoint checkpoints/policy_best.pth --arch impala_v2 --render
"""
import sys
sys.modules['xformers'] = None  # prevent xformers CPU crash (must be before metadrive)

import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import torch

from src.models import build_policy, EGO_DIM
from rl.env_wrapper import MetaDriveRLWrapper
from rl.il_actor_critic import ILActorCritic
from rl.obs_verifier import ObsVerifier


def parse_args():
    p = argparse.ArgumentParser(description="MetaDrive RL Policy Test")
    p.add_argument("--checkpoint",    type=str, default=None,
                   help="RL checkpoint (policy key). Mutually exclusive with --il_checkpoint.")
    p.add_argument("--il_checkpoint", type=str, default=None,
                   help="IL checkpoint (model key) — test IL policy inside the RL env.")
    p.add_argument("--arch",          type=str, default="impala",
                   choices=["simple", "impala", "impala_v2"])
    p.add_argument("--image_size",    type=int, default=84)
    p.add_argument("--scenarios",     type=int, default=5)
    p.add_argument("--dpt_path",      type=str, default=None)
    p.add_argument("--render",        action="store_true", default=True)
    p.add_argument("--obs_ref",       type=str, default=None, metavar="PATH",
                   help="Reference JSON from train_rl.py --save_obs_ref. "
                        "Prints a match report after the test run.")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # ── Load policy ───────────────────────────────────────────────────────────
    if not args.checkpoint and not args.il_checkpoint:
        print("Error: provide --checkpoint (RL) or --il_checkpoint (IL).")
        return

    il_model = build_policy(args.arch, args.image_size)
    policy   = ILActorCritic(il_model).to(device)

    if args.il_checkpoint:
        if not os.path.exists(args.il_checkpoint):
            print(f"IL checkpoint not found: {args.il_checkpoint}")
            return
        policy.load_from_il_checkpoint(args.il_checkpoint, device)
        print(f"[Mode] IL policy running inside RL environment (deterministic, no sampling)")
    else:
        if not os.path.exists(args.checkpoint):
            print(f"Checkpoint not found: {args.checkpoint}")
            return
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "policy" in ckpt:
            policy.load_state_dict(ckpt["policy"])
            print(f"Loaded RL checkpoint (step={ckpt.get('global_step','?')}): {args.checkpoint}")
        else:
            policy.load_state_dict(ckpt)
            print(f"Loaded raw state-dict: {args.checkpoint}")

    policy.eval()

    # ── Environment ───────────────────────────────────────────────────────────
    env_config = {
        "use_render": args.render,
        "show_interface": args.render,
        "manual_control": False,
        "num_scenarios": args.scenarios,
    }
    env = MetaDriveRLWrapper(
        env_config=env_config,
        show_perception=args.render,
        image_size=args.image_size,
        dpt_path=args.dpt_path,
    )

    print(f"\n{'='*60}")
    print(f"  RL Policy Test  ({args.scenarios} scenarios)")
    print(f"{'='*60}\n")

    verifier = ObsVerifier() if args.obs_ref else None

    try:
        ep_count   = 1
        obs        = env.reset(seed=0)
        ep_reward  = 0.0
        step_count = 0

        while True:
            img, ego = obs
            if verifier is not None:
                verifier.record(img, ego)

            obs_img = torch.from_numpy(img).unsqueeze(0).to(device)
            obs_ego = torch.from_numpy(ego).unsqueeze(0).to(device)

            with torch.inference_mode():
                action_mean, _ = policy(obs_img, obs_ego)

            # Deterministic + clipped (BUG FIX: original test_rl.py forgot to clip)
            action = np.clip(action_mean.squeeze(0).cpu().numpy(), -1.0, 1.0)

            obs, reward, done, info = env.step(action)
            ep_reward  += reward
            step_count += 1

            if done:
                route = info.get("route_completion", 0.0)
                speed = info.get("speed_km_h", 0.0)
                print(
                    f"Episode {ep_count:3d} | "
                    f"Reward: {ep_reward:+7.2f} | "
                    f"Route: {route*100:5.1f}% | "
                    f"Steps: {step_count:4d} | "
                    f"Speed: {speed:.1f} km/h"
                )
                if ep_count >= args.scenarios:
                    print("\nTest complete.")
                    break
                obs        = env.reset(seed=ep_count)
                ep_reward  = 0.0
                step_count = 0
                ep_count  += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError during test: {e}")
        raise
    finally:
        env.close()

    if verifier is not None and len(verifier) > 0:
        if os.path.exists(args.obs_ref):
            verifier.compare(args.obs_ref)
        else:
            print(f"[ObsVerifier] Reference not found: {args.obs_ref}")
            verifier.print_stats()


if __name__ == "__main__":
    main()
