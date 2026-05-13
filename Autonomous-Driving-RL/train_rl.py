#!/usr/bin/env python3
"""
RL Training (PPO) — IL→RL fine-tuning support
===============================================
Run from the Autonomous-Driving-RL/ directory.

Examples
--------
# Fine-tune from an IL checkpoint:
    python train_rl.py --il_checkpoint ../models/policy_model_best.pth --arch impala

# Train from scratch with ImpalaNetV2:
    python train_rl.py --arch impala_v2 --image_size 84 --timesteps 500000

# Resume an RL checkpoint:
    python train_rl.py --rl_checkpoint checkpoints/policy_latest.pth --arch impala
"""
import sys
import os
from pathlib import Path

# Allow importing from the parent IL repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time
import numpy as np
import torch

from src.models import build_policy, EGO_DIM
from rl.rewards import RewardConfig
from rl.env_wrapper import MetaDriveRLWrapper
from rl.il_actor_critic import ILActorCritic
from rl.ppo import PPOConfig, RolloutBuffer, ppo_update
from rl.obs_verifier import ObsVerifier


def parse_args():
    p = argparse.ArgumentParser(description="MetaDrive RL Training (PPO)")
    p.add_argument("--timesteps",     type=int,   default=200_000)
    p.add_argument("--lr",            type=float, default=3e-4,
                   help="LR for value_head and log_std")
    p.add_argument("--backbone_lr",   type=float, default=1e-5,
                   help="LR for the IL backbone (much lower to avoid forgetting)")
    p.add_argument("--warmup_updates", type=int,  default=5,
                   help="Freeze backbone for this many PPO updates while critic warms up")
    p.add_argument("--rollout",       type=int,   default=2048)
    p.add_argument("--batch",         type=int,   default=64)
    p.add_argument("--epochs",        type=int,   default=10)
    p.add_argument("--arch",          type=str,   default="impala",
                   choices=["simple", "impala", "impala_v2"])
    p.add_argument("--image_size",    type=int,   default=84)
    p.add_argument("--il_checkpoint", type=str,   default=None,
                   help="Path to an IL training checkpoint to fine-tune from")
    p.add_argument("--rl_checkpoint", type=str,   default=None,
                   help="Path to a previous RL checkpoint to resume from")
    p.add_argument("--dpt_path",      type=str,   default=None,
                   help="Path to fine-tuned DPT weights (optional)")
    p.add_argument("--render",        action="store_true")
    p.add_argument("--save_dir",      type=str,   default="models/rl")
    p.add_argument("--scenarios",     type=int,   default=50)
    p.add_argument("--seed",          type=int,   default=42)
    # ── Optimisation flags ────────────────────────────────────────────────────
    p.add_argument("--amp",     action="store_true",
                   help="Mixed-precision PPO update (CUDA only, ~1.5× faster update step)")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the IL backbone (PyTorch ≥2.0, ~1.3× faster inference)")
    # ── Obs-pipeline verification ─────────────────────────────────────────────
    p.add_argument("--save_obs_ref", type=str, default=None, metavar="PATH",
                   help="After the first rollout save per-channel obs stats to PATH (JSON). "
                        "Use verify_simulation.py --obs_ref PATH to check sim matches training.")
    return p.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  MetaDrive RL Training (PPO)")
    print(f"  Device       : {device}")
    print(f"  Arch         : {args.arch}  image_size={args.image_size}")
    print(f"  Total steps  : {args.timesteps:,}")
    print(f"  Rollout      : {args.rollout}  batch={args.batch}  epochs={args.epochs}")
    print(f"  Seed         : {args.seed}")
    print(f"  LR (heads)   : {args.lr}  backbone_lr={args.backbone_lr}  warmup={args.warmup_updates} updates")
    print(f"  AMP          : {args.amp}   compile={args.compile}")
    if args.il_checkpoint:
        print(f"  IL checkpoint: {args.il_checkpoint}")
    if args.rl_checkpoint:
        print(f"  RL checkpoint: {args.rl_checkpoint}")
    if args.save_obs_ref:
        print(f"  Obs ref out  : {args.save_obs_ref}")
    print(f"{'='*60}\n")

    # ── Reward config ─────────────────────────────────────────────────────────
    reward_cfg = RewardConfig()
    print("Active reward weights:")
    for k, v in vars(reward_cfg).items():
        print(f"  {k:35s} = {v}")
    print()

    # ── Environment ───────────────────────────────────────────────────────────
    env_config = {
        "use_render": args.render,
        "num_scenarios": args.scenarios,
        "start_seed": args.seed,
    }
    env = MetaDriveRLWrapper(
        reward_cfg=reward_cfg,
        env_config=env_config,
        show_perception=args.render,
        image_size=args.image_size,
        dpt_path=args.dpt_path,
    )

    # ── Policy ────────────────────────────────────────────────────────────────
    il_model = build_policy(args.arch, args.image_size)
    policy   = ILActorCritic(il_model).to(device)

    if args.il_checkpoint and os.path.exists(args.il_checkpoint):
        policy.load_from_il_checkpoint(args.il_checkpoint, device)
    elif args.il_checkpoint:
        print(f"[WARNING] IL checkpoint not found: {args.il_checkpoint}")

    # Optionally resume a full RL checkpoint (overrides IL weights).
    start_global_step = 0
    best_avg_route    = 0.0
    if args.rl_checkpoint and os.path.exists(args.rl_checkpoint):
        rl_ckpt = torch.load(args.rl_checkpoint, map_location=device)
        policy.load_state_dict(rl_ckpt["policy"])
        start_global_step = rl_ckpt.get("global_step", 0)
        best_avg_route    = rl_ckpt.get("route", 0.0)
        print(f"[RL Resume] step={start_global_step:,}  best_route={best_avg_route*100:.1f}%")

    # Separate LRs: critic/log_std get full LR; IL backbone gets much smaller LR
    # to avoid overwriting learned representations with early noisy gradients.
    backbone_params = list(policy.il_model.parameters())
    head_params     = list(policy.value_head.parameters()) + [policy.log_std]
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params,     "lr": args.lr},
    ])

    # Freeze backbone during warmup so the value head gets sensible before
    # backbone gradients flow (prevents corrupted advantages from damaging IL weights).
    for p in backbone_params:
        p.requires_grad_(False)
    print(f"[Warmup] Backbone frozen for first {args.warmup_updates} PPO updates.")

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {total_params:,}\n")

    # ── Optional torch.compile ────────────────────────────────────────────────
    if args.compile:
        try:
            print("[Compile] Compiling IL backbone with torch.compile ...")
            policy.il_model = torch.compile(policy.il_model)
            print("[Compile] Done.\n")
        except Exception as exc:
            print(f"[Compile] torch.compile failed ({exc}); continuing without.\n")

    # ── AMP scaler (created once; carries scale state across updates) ─────────
    amp_scaler = None
    if args.amp:
        if device.type != "cuda":
            print("[AMP] Ignored — AMP requires CUDA.\n")
        else:
            amp_scaler = torch.amp.GradScaler("cuda")
            print("[AMP] Mixed-precision update enabled.\n")

    # ── PPO config & buffer ───────────────────────────────────────────────────
    ppo_cfg = PPOConfig(
        rollout_steps=args.rollout,
        epochs_per_update=args.epochs,
        mini_batch_size=args.batch,
        lr=args.lr,
        total_timesteps=args.timesteps,
        use_amp=args.amp and device.type == "cuda",
    )

    buffer = RolloutBuffer(
        size=ppo_cfg.rollout_steps,
        img_shape=(4, args.image_size, args.image_size),
        ego_dim=EGO_DIM,
        act_dim=env.ACT_DIM,
        device=device,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Training state ────────────────────────────────────────────────────────
    global_step    = start_global_step
    update_count   = 0
    episode_count  = 0
    episode_rewards = []
    episode_lengths = []
    episode_routes  = []

    ep_reward = 0.0
    ep_length = 0
    obs = env.reset()  # obs = (img_np, ego_np)

    obs_verifier = ObsVerifier() if args.save_obs_ref else None
    obs_ref_saved = False

    start_time = time.time()
    print("Training started...\n")

    while global_step < ppo_cfg.total_timesteps:

        # ── Rollout collection ────────────────────────────────────────────────
        buffer.reset()
        policy.eval()

        for step in range(ppo_cfg.rollout_steps):
            global_step += 1

            img, ego = obs
            # torch.from_numpy shares memory (no CPU copy); non_blocking async H2D
            obs_img = torch.from_numpy(img).unsqueeze(0).to(device, non_blocking=True)
            obs_ego = torch.from_numpy(ego).unsqueeze(0).to(device, non_blocking=True)

            if obs_verifier is not None:
                obs_verifier.record(img, ego)

            with torch.no_grad():
                action, log_prob, _, value = policy.get_action_and_value(obs_img, obs_ego)

            action_np      = action.cpu().numpy()[0]
            action_clipped = np.clip(action_np, -1.0, 1.0)

            next_obs, reward, done, info = env.step(action_clipped)

            buffer.store(
                img, ego, action_clipped, reward, float(done),
                log_prob.item(), value.item()
            )

            obs        = next_obs
            ep_reward += reward
            ep_length += 1

            # Live step display
            pct         = global_step / ppo_cfg.total_timesteps * 100
            step_speed  = info.get("speed_km_h", 0)
            step_route  = info.get("route_completion", 0)
            sys.stdout.write(
                f"\r  [{global_step:>8,}/{ppo_cfg.total_timesteps:,} {pct:4.1f}%] "
                f"EP {episode_count+1:3d} | "
                f"St:{action_clipped[0]:+.2f} Th:{action_clipped[1]:+.2f} | "
                f"Spd:{step_speed:5.1f} | R:{ep_reward:+7.1f} | Rt:{step_route*100:4.1f}%"
                f"{'':10s}"
            )
            sys.stdout.flush()

            if done:
                episode_count += 1
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_length)

                route   = info.get("route_completion", 0)
                speed   = info.get("speed_km_h", 0)
                details = info.get("reward_details", {})
                episode_routes.append(route)

                penalty_str = " | ".join(
                    f"{k}: {v:+.2f}" for k, v in details.items() if abs(v) > 0.001
                )
                print(
                    f"\n  EP {episode_count:4d} | "
                    f"R: {ep_reward:+7.2f} | Len: {ep_length:4d} | "
                    f"Route: {route*100:5.1f}% | Spd: {speed:5.1f} | {penalty_str}"
                )

                ep_reward = 0.0
                ep_length = 0
                obs = env.reset()

        # ── Save obs reference (first rollout only) ───────────────────────────
        if obs_verifier is not None and not obs_ref_saved:
            obs_verifier.save(args.save_obs_ref)
            obs_ref_saved = True

        # ── GAE ───────────────────────────────────────────────────────────────
        last_img, last_ego = obs
        last_img_t = torch.from_numpy(last_img).unsqueeze(0).to(device)
        last_ego_t = torch.from_numpy(last_ego).unsqueeze(0).to(device)
        with torch.no_grad():
            last_value = policy.get_value(last_img_t, last_ego_t).item()
        buffer.compute_gae(last_value, ppo_cfg.gamma, ppo_cfg.gae_lambda)

        # ── PPO update ────────────────────────────────────────────────────────
        policy.train()
        losses = ppo_update(policy, optimizer, buffer, ppo_cfg, scaler=amp_scaler)
        update_count += 1

        if update_count == args.warmup_updates:
            for p in backbone_params:
                p.requires_grad_(True)
            print(f"[Warmup] Backbone unfrozen at update {update_count} — fine-tuning with lr={args.backbone_lr}.")

        elapsed = time.time() - start_time
        fps     = global_step / max(elapsed, 1e-6)

        recent_rewards = episode_rewards[-20:] if episode_rewards else [0]
        avg_reward     = np.mean(recent_rewards)
        avg_length     = np.mean(episode_lengths[-20:]) if episode_lengths else 0
        avg_route      = np.mean(episode_routes[-20:])  if episode_routes  else 0

        print(
            f"\nUpdate {update_count:3d} | "
            f"Step {global_step:>8,}/{ppo_cfg.total_timesteps:,} | "
            f"FPS: {fps:.0f} | "
            f"Avg R(20): {avg_reward:+.2f} | "
            f"Avg Route: {avg_route*100:.1f}% | "
            f"Avg Len: {avg_length:.0f} | "
            f"P_loss: {losses['policy_loss']:.4f} | "
            f"V_loss: {losses['value_loss']:.4f} | "
            f"Entropy: {losses['entropy']:.4f}\n"
        )

        # ── Checkpoints ───────────────────────────────────────────────────────
        ckpt = {
            "policy":        policy.state_dict(),
            "il_model_arch": args.arch,
            "image_size":    args.image_size,
            "global_step":   global_step,
            "update":        update_count,
            "route":         best_avg_route,
            "seed":          args.seed,
        }
        torch.save(ckpt, os.path.join(args.save_dir, "policy_latest.pth"))

        if avg_route > best_avg_route and len(episode_routes) >= 10:
            best_avg_route  = avg_route
            ckpt["route"]   = best_avg_route
            torch.save(ckpt, os.path.join(args.save_dir, "policy_best.pth"))
            print(f"  *** New best model (Avg Route: {best_avg_route*100:.1f}%)\n")

    # ── Done ──────────────────────────────────────────────────────────────────
    env.close()
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Total time  : {total_time/60:.1f} min")
    print(f"  Total steps : {global_step:,}")
    print(f"  Episodes    : {episode_count}")
    print(f"  Avg reward  : {avg_reward:+.2f}")
    print(f"  Avg route   : {avg_route*100:.1f}%")
    print(f"  Best route  : {best_avg_route*100:.1f}%")
    print(f"  Saved to    : {args.save_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
