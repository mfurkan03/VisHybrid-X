#!/usr/bin/env python3
"""
RL Training (PPO) — IL→RL fine-tuning with parallel environments
=================================================================
Run from the Autonomous-Driving-RL/ directory.

Examples
--------
# Fine-tune from an IL checkpoint (4 parallel envs):
    python train_rl.py --il_checkpoint ../models/policy_model_best.pth --arch impala --n_envs 4

# Single-env (backward-compatible, no subprocess overhead):
    python train_rl.py --il_checkpoint ../models/policy_model_best.pth --arch impala

# Resume an RL checkpoint:
    python train_rl.py --rl_checkpoint models/rl/policy_latest.pth --arch impala --n_envs 4

Parallelization design
-----------------------
With --n_envs N > 1, N worker processes each run one MetaDrive instance
(no DPT). The main process batches DPT depth inference over all N raw-RGB
observations at once, runs the policy forward over the batch, then sends
actions back. This yields ~N× more env transitions per wall-clock second.

With --n_envs 1, a DummyVecEnv wraps a single env — the code path is
identical so results are reproducible with a single process for debugging.
"""
import sys
import os
from pathlib import Path

# Allow importing from the parent IL repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F

from src.models import build_policy, EGO_DIM, DepthEstimationModel
from rl.rewards import RewardConfig
from rl.vec_env import SubprocVecEnv, DummyVecEnv, MakeEnvFn
from rl.il_actor_critic import ILActorCritic
from rl.ppo import PPOConfig, RolloutBuffer, ppo_update
from rl.obs_verifier import ObsVerifier


# ── Observation assembly ──────────────────────────────────────────────────────

def build_obs_batch(
    raw_rgbs: np.ndarray,
    depth_model: DepthEstimationModel,
    image_size: int,
) -> np.ndarray:
    """
    Convert raw uint8 RGB frames from N workers into 4-channel policy input.

    Parameters
    ----------
    raw_rgbs    : (N, 196, 196, 3) uint8 numpy — straight from MetaDrive sensors
    depth_model : DepthEstimationModel in the main process (GPU)
    image_size  : target spatial resolution (e.g. 84)

    Returns
    -------
    imgs_np : (N, 4, image_size, image_size) float32 numpy
              ch0 = inverted depth (closer = higher, matching IL convention)
              ch1-3 = RGB ∈ [0, 1]

    Depth normalization is per-image (not across the batch) to match the
    single-env behaviour in env_wrapper._get_obs().  All ops are fully batched
    to maximise GPU utilisation.
    """
    dev = depth_model.device
    N   = raw_rgbs.shape[0]

    with torch.no_grad():
        # Single batched DPT call — predict_batch_with_grad handles ImageNet norm.
        depth_raw = depth_model.predict_batch_with_grad(raw_rgbs)
        # depth_raw: (N, 1, H_dpt, W_dpt) on dev

    # Per-image depth inversion — fully batched, no Python loop
    d     = depth_raw[:, 0]                                # (N, H_dpt, W_dpt)
    d_min = d.flatten(1).min(1).values[:, None, None]     # (N, 1, 1)
    d_max = d.flatten(1).max(1).values[:, None, None]     # (N, 1, 1)
    depth_inv = 1.0 - (d - d_min) / (d_max - d_min + 1e-6)   # (N, H_dpt, W_dpt)

    # RGB: (N, H, W, 3) → (N, 3, H, W), scaled to [0, 1]
    rgb_t = torch.from_numpy(raw_rgbs).float().to(dev) / 255.0  # (N, H, W, 3)
    rgb_t = rgb_t.permute(0, 3, 1, 2)                           # (N, 3, H, W)

    # Assemble 4-ch tensor and resize in one pass
    combined = torch.cat([depth_inv.unsqueeze(1), rgb_t], dim=1)  # (N, 4, H, W)
    if combined.shape[-1] != image_size:
        combined = F.interpolate(
            combined,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )

    return combined.cpu().float().numpy()   # (N, 4, H, W)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MetaDrive RL Training (PPO)")
    p.add_argument("--timesteps",      type=int,   default=200_000)
    p.add_argument("--lr",             type=float, default=3e-4,
                   help="LR for value_head and log_std")
    p.add_argument("--backbone_lr",    type=float, default=1e-5,
                   help="LR for the IL backbone (much lower to avoid forgetting)")
    p.add_argument("--warmup_updates", type=int,   default=20,
                   help="Freeze backbone+log_std for this many PPO updates (critic-only warmup)")
    p.add_argument("--rollout",        type=int,   default=2048,
                   help="Rollout steps *per env* per PPO update")
    p.add_argument("--batch",          type=int,   default=64)
    p.add_argument("--epochs",         type=int,   default=10)
    p.add_argument("--arch",           type=str,   default="impala",
                   choices=["simple", "impala", "impala_v2"])
    p.add_argument("--image_size",     type=int,   default=84)
    p.add_argument("--n_envs",         type=int,   default=1,
                   help="Number of parallel env workers. 1=DummyVecEnv (no subprocess), "
                        "N>1=SubprocVecEnv. Total transitions/update = n_envs * rollout.")
    p.add_argument("--il_checkpoint",  type=str,   default=None,
                   help="Path to an IL training checkpoint to fine-tune from")
    p.add_argument("--rl_checkpoint",  type=str,   default=None,
                   help="Path to a previous RL checkpoint to resume from")
    p.add_argument("--dpt_path",       type=str,   default=None,
                   help="Path to fine-tuned DPT weights (optional)")
    p.add_argument("--render",         action="store_true")
    p.add_argument("--save_dir",       type=str,   default="models/rl")
    p.add_argument("--scenarios",      type=int,   default=50)
    p.add_argument("--seed",           type=int,   default=42)
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_envs = args.n_envs

    print(f"\n{'='*60}")
    print(f"  MetaDrive RL Training (PPO)")
    print(f"  Device       : {device}")
    print(f"  Arch         : {args.arch}  image_size={args.image_size}")
    print(f"  Envs         : {n_envs}  ({'SubprocVecEnv' if n_envs > 1 else 'DummyVecEnv'})")
    print(f"  Total steps  : {args.timesteps:,}")
    print(f"  Rollout/env  : {args.rollout}  (total/update = {args.rollout * n_envs:,})")
    print(f"  Batch        : {args.batch}  epochs={args.epochs}")
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

    # ── DPT depth model in main process ──────────────────────────────────────
    # Workers return raw uint8 RGB; DPT runs here on a batch of N images.
    depth_model = DepthEstimationModel(finetuned_path=args.dpt_path)
    depth_model.set_eval_mode()

    # ── Vectorized environments ───────────────────────────────────────────────
    make_env_fns = [
        MakeEnvFn(
            seed=args.seed,
            worker_idx=i,
            scenarios=args.scenarios,
            reward_cfg=reward_cfg,
            image_size=args.image_size,
            dpt_path=args.dpt_path,
        )
        for i in range(n_envs)
    ]

    if n_envs > 1:
        vec_env = SubprocVecEnv(n_envs, make_env_fns)
        print(f"[VecEnv] {n_envs} worker processes started.\n")
    else:
        vec_env = DummyVecEnv(make_env_fns[0])
        print("[VecEnv] DummyVecEnv (single env, no subprocess).\n")

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

    # During warmup: freeze backbone AND log_std, train only value_head.
    # Reason: policy gradient (which flows through log_std) uses advantage estimates
    # from the randomly-initialized value head. Those estimates are unreliable early on
    # and can push log_std down (entropy collapse) before the critic is calibrated,
    # causing the policy to become near-deterministic and degrade toward standing still.
    for p in backbone_params:
        p.requires_grad_(False)
    policy.log_std.requires_grad_(False)
    print(f"[Warmup] Backbone + log_std frozen for first {args.warmup_updates} PPO updates (critic-only warmup).")

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
        act_dim=2,
        device=device,
        n_envs=n_envs,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Training state ────────────────────────────────────────────────────────
    global_step    = start_global_step
    update_count   = 0
    episode_count  = 0
    episode_rewards = []
    episode_lengths = []
    episode_routes  = []

    # Per-env accumulators (N separate episode stats)
    ep_rewards = np.zeros(n_envs, dtype=np.float32)
    ep_lengths = np.zeros(n_envs, dtype=np.int32)

    # Initial reset — get raw RGB from all workers
    raw_rgbs, egos = vec_env.reset()                               # (N,196,196,3), (N,3)
    imgs_np = build_obs_batch(raw_rgbs, depth_model, args.image_size)  # (N,4,H,W)

    obs_verifier  = ObsVerifier() if args.save_obs_ref else None
    obs_ref_saved = False

    start_time = time.time()
    print("Training started...\n")

    while global_step < ppo_cfg.total_timesteps:

        # ── Rollout collection ────────────────────────────────────────────────
        buffer.reset()
        policy.eval()

        for step in range(ppo_cfg.rollout_steps):
            global_step += n_envs   # each step advances all N envs simultaneously

            # Current obs → tensors
            imgs_t = torch.from_numpy(imgs_np).to(device, non_blocking=True)  # (N,4,H,W)
            egos_t = torch.from_numpy(egos).to(device, non_blocking=True)      # (N,3)

            if obs_verifier is not None:
                for i in range(n_envs):
                    obs_verifier.record(imgs_np[i], egos[i])

            with torch.no_grad():
                actions, log_probs, _, values = policy.get_action_and_value(imgs_t, egos_t)
                # actions: (N,2)  log_probs: (N,)  values: (N,)

            actions_np = np.clip(actions.cpu().numpy(), -1.0, 1.0)   # (N, 2)

            # Step all envs; workers auto-reset on done (SB3 convention)
            raw_rgbs, next_egos, rewards, dones, infos = vec_env.step(actions_np)

            # Store the CURRENT obs (before overwriting with next obs)
            buffer.store(
                imgs_np,                        # (N, 4, H, W)
                egos,                           # (N, 3)
                actions_np,                     # (N, 2)
                rewards,                        # (N,)
                dones.astype(np.float32),       # (N,)
                log_probs.cpu().numpy(),        # (N,)
                values.cpu().numpy(),           # (N,)
            )

            # Build next obs in main process (batched DPT)
            imgs_np = build_obs_batch(raw_rgbs, depth_model, args.image_size)
            egos    = next_egos

            # Accumulate per-env episode stats
            ep_rewards += rewards
            ep_lengths += 1

            # Live status line (env 0)
            pct        = global_step / ppo_cfg.total_timesteps * 100
            step_speed = infos[0].get("speed_km_h", 0)
            step_route = infos[0].get("route_completion", 0)
            sys.stdout.write(
                f"\r  [{global_step:>8,}/{ppo_cfg.total_timesteps:,} {pct:4.1f}%] "
                f"EP {episode_count+1:3d} [env0] | "
                f"St:{actions_np[0,0]:+.2f} Th:{actions_np[0,1]:+.2f} | "
                f"Spd:{step_speed:5.1f} | R:{ep_rewards[0]:+7.1f} | Rt:{step_route*100:4.1f}%"
                f"{'':10s}"
            )
            sys.stdout.flush()

            # Log completed episodes (any env may finish at this step)
            for env_idx in range(n_envs):
                if dones[env_idx]:
                    episode_count += 1
                    episode_rewards.append(ep_rewards[env_idx])
                    episode_lengths.append(ep_lengths[env_idx])

                    route   = infos[env_idx].get("route_completion", 0)
                    speed   = infos[env_idx].get("speed_km_h", 0)
                    details = infos[env_idx].get("reward_details", {})
                    episode_routes.append(route)

                    penalty_str = " | ".join(
                        f"{k}: {v:+.2f}" for k, v in details.items() if abs(v) > 0.001
                    )
                    print(
                        f"\n  EP {episode_count:4d} [env{env_idx}] | "
                        f"R: {ep_rewards[env_idx]:+7.2f} | Len: {ep_lengths[env_idx]:4d} | "
                        f"Route: {route*100:5.1f}% | Spd: {speed:5.1f} | {penalty_str}"
                    )

                    ep_rewards[env_idx] = 0.0
                    ep_lengths[env_idx] = 0

        # ── Save obs reference (first rollout only) ───────────────────────────
        if obs_verifier is not None and not obs_ref_saved:
            obs_verifier.save(args.save_obs_ref)
            obs_ref_saved = True

        # ── GAE bootstrap ─────────────────────────────────────────────────────
        # imgs_np / egos now hold the obs *after* the last rollout step.
        # For envs that just reset, this is the first obs of the new episode.
        last_imgs_t = torch.from_numpy(imgs_np).to(device)   # (N, 4, H, W)
        last_egos_t = torch.from_numpy(egos).to(device)      # (N, 3)
        with torch.no_grad():
            last_values = policy.get_value(last_imgs_t, last_egos_t).cpu().numpy()   # (N,)

        # dones[T-1] already stored in buffer — compute_gae uses it to mask bootstrap
        buffer.compute_gae(last_values, ppo_cfg.gamma, ppo_cfg.gae_lambda)

        # ── PPO update ────────────────────────────────────────────────────────
        policy.train()
        losses = ppo_update(policy, optimizer, buffer, ppo_cfg, scaler=amp_scaler)
        update_count += 1

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if update_count == args.warmup_updates:
            for p in backbone_params:
                p.requires_grad_(True)
            policy.log_std.requires_grad_(True)
            print(f"[Warmup] Backbone + log_std unfrozen at update {update_count} — fine-tuning with lr={args.backbone_lr}.")

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
            f"Envs: {n_envs} | "
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
            "n_envs":        n_envs,
        }
        torch.save(ckpt, os.path.join(args.save_dir, "policy_latest.pth"))

        if avg_route > best_avg_route and len(episode_routes) >= 10:
            best_avg_route  = avg_route
            ckpt["route"]   = best_avg_route
            torch.save(ckpt, os.path.join(args.save_dir, "policy_best.pth"))
            print(f"  *** New best model (Avg Route: {best_avg_route*100:.1f}%)\n")

    # ── Done ──────────────────────────────────────────────────────────────────
    vec_env.close()
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
