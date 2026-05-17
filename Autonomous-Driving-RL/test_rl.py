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

STUCK_SPEED    = 0.05   # normalized total_speed (0-1) below which the car is considered stationary
STUCK_PATIENCE = 500    # consecutive steps below threshold before terminating

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
    p.add_argument("--scenarios",     type=int, default=10)
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
    start_seed = 316181
    env_config = {
        "use_render": args.render,
        "show_interface": args.render,
        "manual_control": False,
        "num_scenarios": args.scenarios,
        "start_seed": start_seed,
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

    success_flags, route_completions = [], []
    out_of_roads, crash_vehicles, crash_objects = [], [], []
    survival_times, average_speeds, jitter_rates, safety_scores, total_rewards = [], [], [], [], []

    try:
        ep_count   = 1
        obs        = env.reset(seed=start_seed)
        ep_reward   = 0.0
        step_count  = 0
        speeds      = []
        steers      = []
        stuck_steps = 0
        stuck       = False

        while True:
            img, ego = obs
            if verifier is not None:
                verifier.record(img, ego)

            obs_img = torch.from_numpy(img).unsqueeze(0).to(device)
            obs_ego = torch.from_numpy(ego).unsqueeze(0).to(device)

            with torch.inference_mode():
                action = policy.act_deterministic(obs_img, obs_ego).squeeze(0).cpu().numpy()

            obs, reward, done, info = env.step(action)
            ep_reward  += reward
            step_count += 1
            steers.append(float(action[0]))
            speeds.append(info.get("speed_km_h", 0.0))

            if float(ego[0]) < STUCK_SPEED:
                stuck_steps += 1
            else:
                stuck_steps = 0
            if stuck_steps >= STUCK_PATIENCE:
                stuck = True
                done  = True

            if done:
                route    = info.get("route_completion", 0.0)
                success  = bool(info.get("arrive_dest", False))
                oor      = bool(info.get("out_of_road", False))
                crash_v  = bool(info.get("crash_vehicle", False))
                crash_o  = bool(info.get("crash_object", False))
                avg_spd  = float(np.mean(speeds)) if speeds else 0.0
                jitter   = float(np.mean(np.abs(np.diff(steers)))) if len(steers) > 1 else 0.0

                success_flags.append(success)
                route_completions.append(route)
                out_of_roads.append(oor)
                crash_vehicles.append(crash_v)
                crash_objects.append(crash_o)
                survival_times.append(step_count)
                average_speeds.append(avg_spd)
                jitter_rates.append(jitter)
                safety_scores.append(
                    0.35 * (not crash_v)
                    + 0.35 * (not oor)
                    + 0.30 * max(0.0, 1.0 - jitter / 0.3)
                )
                total_rewards.append(ep_reward)

                reason = "success" if success else (
                    "stuck"          if stuck   else (
                    "out_of_road"    if oor     else (
                    "crash_vehicle"  if crash_v else (
                    "crash_object"   if crash_o else "timeout/other"))))

                print(
                    f"Episode {ep_count:3d} | "
                    f"Reward: {ep_reward:+7.2f} | "
                    f"Route: {route*100:5.1f}% | "
                    f"Steps: {step_count:4d} | "
                    f"Spd: {avg_spd:5.1f} km/h | "
                    f"Jitter: {jitter:.4f} | "
                    f"Safety: {safety_scores[-1]*100:.1f}% | "
                    f"{reason}"
                )
                if ep_count >= args.scenarios:
                    print("\nTest complete.")
                    break
                obs         = env.reset(seed=start_seed + ep_count)
                ep_reward   = 0.0
                step_count  = 0
                speeds      = []
                steers      = []
                stuck_steps = 0
                stuck       = False
                ep_count   += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError during test: {e}")
        raise
    finally:
        env.close()

    if success_flags:
        print(
            f"\n=== ONLINE SUMMARY ===\n"
            f"Success Rate:         {np.mean(success_flags)*100:.1f}%\n"
            f"Route Completion:     {np.mean(route_completions)*100:.1f}%\n"
            f"Out of Road Rate:     {np.mean(out_of_roads)*100:.1f}%\n"
            f"Crash Vehicle Rate:   {np.mean(crash_vehicles)*100:.1f}%\n"
            f"Crash Object Rate:    {np.mean(crash_objects)*100:.1f}%\n"
            f"Avg Survival Time:    {np.mean(survival_times):.1f} steps\n"
            f"Avg Driving Speed:    {np.mean(average_speeds):.2f} km/h\n"
            f"Avg Steering Jitter:  {np.mean(jitter_rates):.4f}\n"
            f"Safe Driving Score:   {np.mean(safety_scores)*100:.1f}%  "
            f"(35% collision-free + 35% road-adherence + 30% steering-smoothness)\n"
            f"Avg Episode Reward:   {np.mean(total_rewards):.2f}"
        )

    if verifier is not None and len(verifier) > 0:
        if os.path.exists(args.obs_ref):
            verifier.compare(args.obs_ref)
        else:
            print(f"[ObsVerifier] Reference not found: {args.obs_ref}")
            verifier.print_stats()


if __name__ == "__main__":
    main()
