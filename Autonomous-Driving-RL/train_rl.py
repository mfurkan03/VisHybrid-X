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
                   help="LR for value_head (random-init RL critic)")
    p.add_argument("--backbone_lr",    type=float, default=5e-6,
                   help="LR for the IL backbone (much lower to avoid forgetting)")
    p.add_argument("--dist_head_lr",   type=float, default=3e-5,
                   help="LR for Beta distribution heads (steer/throttle mu/nu). "
                        "Lower than value_head lr to prevent entropy collapse after warmup.")
    p.add_argument("--warmup_updates", type=int,   default=20,
                   help="Freeze backbone + action heads for this many PPO updates (critic-only warmup)")
    p.add_argument("--rollout",        type=int,   default=2048,
                   help="Rollout steps *per env* per PPO update")
    p.add_argument("--batch",          type=int,   default=256)
    p.add_argument("--epochs",         type=int,   default=10)
    p.add_argument("--force_lr",        action="store_true",
                   help="Override LRs in restored optimizer state (use when resuming with new --lr / --backbone_lr)")
    p.add_argument("--target_kl",      type=float, default=0.05,
                   help="Per-epoch avg KL early-stopping threshold. 0=disabled. "
                        "~0.05 for IL->RL fine-tuning, ~0.01 for scratch PPO.")
    p.add_argument("--arch",           type=str,   default="impala",
                   choices=["simple", "impala", "impala_v2"])
    p.add_argument("--image_size",     type=int,   default=84)
    p.add_argument("--camera_fov",     type=float, default=60,
                   help="Camera horizontal FOV in degrees (default 60)")
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
    # ── Weights & Biases ──────────────────────────────────────────────────────
    p.add_argument("--wandb",         action="store_true",
                   help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", type=str, default="metadrive-rl",
                   help="W&B project name")
    p.add_argument("--wandb_run_name", type=str, default="rl_run",
                   help="W&B run display name (auto-generated if omitted)")
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
    print(f"  Batch        : {args.batch}  epochs={args.epochs}  target_kl={args.target_kl}")
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
            render=(args.render and i == 0),
            camera_fov=args.camera_fov,
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
        # Reset nu heads to maximum entropy state before PPO fine-tuning
        policy.reset_nu_heads_for_ppo()
    elif args.il_checkpoint:
        print(f"[WARNING] IL checkpoint not found: {args.il_checkpoint}")

    # Optionally resume a full RL checkpoint (overrides IL weights).
    start_global_step  = 0
    start_update_count = 0
    best_avg_route     = 0.0
    wandb_run_id       = None
    resumed_optimizer_state = None
    if args.rl_checkpoint and os.path.exists(args.rl_checkpoint):
        rl_ckpt = torch.load(args.rl_checkpoint, map_location=device, weights_only=False)
        policy.load_state_dict(rl_ckpt["policy"])
        start_global_step       = rl_ckpt.get("global_step", 0)
        start_update_count      = rl_ckpt.get("update", 0)
        best_avg_route          = rl_ckpt.get("route", 0.0)
        wandb_run_id            = rl_ckpt.get("wandb_run_id", None)
        resumed_optimizer_state = rl_ckpt.get("optimizer", None)
        if resumed_optimizer_state is None:
            print(f"[RL Resume] step={start_global_step:,}  update={start_update_count}  best_route={best_avg_route*100:.1f}%  "
                  f"(no optimizer state in checkpoint — Adam starts cold)")
        else:
            print(f"[RL Resume] step={start_global_step:,}  update={start_update_count}  best_route={best_avg_route*100:.1f}%")

    # ── Weights & Biases ──────────────────────────────────────────────────────
    wb_run = None
    if args.wandb:
        try:
            import wandb
            is_resume = wandb_run_id is not None
            wb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                id=wandb_run_id,
                resume="allow",
                config=vars(args),
            )
            wandb_run_id = wb_run.id
            print(f"[W&B] Run: {wb_run.name}  id={wb_run.id}  "
                  f"({'resumed' if is_resume else 'new'})\n")
        except ImportError:
            print("[W&B] wandb not installed — logging disabled. pip install wandb\n")
            wb_run = None

    # Four LR groups:
    #   cnn_params       — CNN backbone + projection + ego MLP: backbone_lr (preserve visual features)
    #   dist_head_params — Beta distribution heads: dist_head_lr (IL-trained, must update slowly
    #                      to avoid entropy collapse; much lower than value_head lr)
    #   head_params      — value_head: lr (random init, must learn fast)
    _DIST_HEAD_NAMES = {"steer_mu_head", "steer_nu_head", "throttle_mu_head", "throttle_nu_head"}
    dist_head_params = []
    cnn_params       = []
    for name, param in policy.il_model.named_parameters():
        if name.split(".")[0] in _DIST_HEAD_NAMES:
            dist_head_params.append(param)
        else:
            cnn_params.append(param)
    il_params   = cnn_params + dist_head_params   # all il_model params, used for warmup freeze/unfreeze
    head_params = list(policy.value_head.parameters())

    optimizer = torch.optim.Adam([
        {"params": cnn_params,       "lr": args.backbone_lr},
        {"params": dist_head_params, "lr": args.dist_head_lr},
        {"params": head_params,      "lr": args.lr},
    ])
    if resumed_optimizer_state is not None:
        optimizer.load_state_dict(resumed_optimizer_state)
        if args.force_lr:
            optimizer.param_groups[0]["lr"] = args.backbone_lr   # cnn_params
            optimizer.param_groups[1]["lr"] = args.dist_head_lr  # dist_head_params
            optimizer.param_groups[2]["lr"] = args.lr            # value_head
            print(f"[RL Resume] Optimizer state restored + LRs overridden: "
                  f"backbone={args.backbone_lr}, dist_heads={args.dist_head_lr}, value_head={args.lr}")
        else:
            print("[RL Resume] Optimizer state restored.")

    # During warmup: freeze the entire IL model (CNN + distribution heads)
    # so the policy doesn't move while the critic calibrates on noisy early advantages.
    warmup_already_done = start_update_count >= args.warmup_updates
    if not warmup_already_done:
        for p in il_params:
            p.requires_grad_(False)
        print(f"[Warmup] IL model frozen for first {args.warmup_updates} PPO updates (critic-only warmup).")
    else:
        print(f"[Warmup] Skipped — warmup already completed at update {start_update_count} (IL model stays unfrozen).")

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
        target_kl=args.target_kl,
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
    global_step       = start_global_step
    update_count      = start_update_count
    backbone_unfrozen = warmup_already_done
    _dist_diag_done   = False   # one-shot distribution diagnostic before first PPO update
    episode_count  = 0
    episode_rewards = []
    episode_lengths = []
    episode_routes  = []

    # Per-env persistent observation state — updated after every env step.
    imgs_np = np.zeros((n_envs, 4, args.image_size, args.image_size), dtype=np.float32)
    egos    = np.zeros((n_envs, EGO_DIM), dtype=np.float32)

    # Initial reset — all workers start simultaneously.
    raw_rgbs, egos_init = vec_env.reset()                              # (N,196,196,3), (N,3)
    imgs_np[:] = build_obs_batch(raw_rgbs, depth_model, args.image_size)
    egos[:]    = egos_init

    # Per-env accumulators (reset at episode boundary, not at rollout boundary)
    ep_rewards = np.zeros(n_envs, dtype=np.float32)
    ep_lengths = np.zeros(n_envs, dtype=np.int32)

    obs_verifier  = ObsVerifier() if args.save_obs_ref else None
    obs_ref_saved = False

    start_time = time.time()
    print("Training started...\n")

    while global_step < ppo_cfg.total_timesteps:

        # ── Rollout collection ────────────────────────────────────────────────
        # Per-env buffer pointers: how many transitions each env has stored.
        # The rollout ends when every env has contributed rollout_steps entries.
        # Envs that reset slowly simply finish later; fast envs never pause.
        buffer.reset()
        policy.eval()

        env_ptrs           = [0] * n_envs
        rollout_t0         = time.time()
        rollout_transitions = 0

        while min(env_ptrs) < ppo_cfg.rollout_steps:

            # ── Non-blocking: pick up completed episode resets ────────────────
            reset_obs = vec_env.poll_resets()   # {} when nothing is ready yet
            if reset_obs:
                r_idxs = sorted(reset_obs.keys())
                r_rgbs = np.stack([reset_obs[i][0] for i in r_idxs])
                r_imgs = build_obs_batch(r_rgbs, depth_model, args.image_size)
                for j, i in enumerate(r_idxs):
                    imgs_np[i] = r_imgs[j]
                    egos[i]    = reset_obs[i][1]

            # ── Determine which envs to step this iteration ───────────────────
            # Ready = active (not resetting) AND still need more transitions.
            ready = [i for i in vec_env.active_indices
                     if env_ptrs[i] < ppo_cfg.rollout_steps]

            if not ready:
                # All active envs hit their quota; remaining ones are resetting.
                if all(env_ptrs[i] >= ppo_cfg.rollout_steps for i in range(n_envs)):
                    break   # everyone done — exit collection loop
                # At least one env is still resetting; block until it's back.
                reset_obs = vec_env.wait_any_reset()
                r_idxs = sorted(reset_obs.keys())
                r_rgbs = np.stack([reset_obs[i][0] for i in r_idxs])
                r_imgs = build_obs_batch(r_rgbs, depth_model, args.image_size)
                for j, i in enumerate(r_idxs):
                    imgs_np[i] = r_imgs[j]
                    egos[i]    = reset_obs[i][1]
                continue

            # ── Policy forward on the ready batch (variable size k ≤ N) ──────
            imgs_batch = np.stack([imgs_np[i] for i in ready])   # (k, 4, H, W)
            egos_batch = np.stack([egos[i]    for i in ready])   # (k, EGO_DIM)
            imgs_t = torch.from_numpy(imgs_batch).to(device, non_blocking=True)
            egos_t = torch.from_numpy(egos_batch).to(device, non_blocking=True)

            if obs_verifier is not None:
                for i in ready:
                    obs_verifier.record(imgs_np[i], egos[i])

            with torch.no_grad():
                actions, log_probs, _, values = policy.get_action_and_value(imgs_t, egos_t)

            actions_01  = actions.cpu().numpy()          # (k, 2) in [0, 1] — stored in buffer
            actions_env = actions_01 * 2.0 - 1.0         # (k, 2) in [-1, 1] — sent to env

            # ── Step the ready envs — done workers dispatch RESET immediately ─
            step_results = vec_env.step(ready, actions_env)

            # ── Batch DPT for non-done envs in one GPU call ───────────────────
            cont_envs = [i for i in ready if not step_results[i][3]]
            if cont_envs:
                cont_rgbs = np.stack([step_results[i][0] for i in cont_envs])
                cont_imgs = build_obs_batch(cont_rgbs, depth_model, args.image_size)

            # ── Store transitions and update per-env state ────────────────────
            cont_j = 0
            for k, i in enumerate(ready):
                _, ego_new, reward, done, info = step_results[i]
                t = env_ptrs[i]

                # Write directly into the (T, N, ...) buffer arrays by env index.
                buffer.imgs[t, i]      = imgs_np[i]       # obs that produced the action
                buffer.egos[t, i]      = egos[i]
                buffer.actions[t, i]   = actions_01[k]
                buffer.rewards[t, i]   = reward
                buffer.dones[t, i]     = float(done)
                buffer.log_probs[t, i] = log_probs[k].item()
                buffer.values[t, i]    = values[k].item()
                env_ptrs[i]           += 1
                global_step           += 1
                rollout_transitions   += 1
                ep_rewards[i]         += reward
                ep_lengths[i]         += 1

                if not done:
                    imgs_np[i] = cont_imgs[cont_j]
                    egos[i]    = ego_new
                    cont_j    += 1
                else:
                    # Episode ended — log stats before clearing accumulators.
                    ep_r = float(ep_rewards[i])
                    ep_l = int(ep_lengths[i])
                    episode_count += 1
                    episode_rewards.append(ep_r)
                    episode_lengths.append(ep_l)
                    route   = info.get("route_completion", 0)
                    speed   = info.get("speed_km_h", 0)
                    details = info.get("reward_details", {})
                    episode_routes.append(route)
                    ep_rewards[i] = 0.0
                    ep_lengths[i] = 0
                    # imgs_np[i] / egos[i] intentionally NOT updated here;
                    # poll_resets() will fill them when the reset completes.

                    pstr = " | ".join(
                        f"{kk}: {v:+.2f}" for kk, v in details.items() if abs(v) > 0.001
                    )
                    print(
                        f"\n  EP {episode_count:4d} [env{i}] | "
                        f"R: {ep_r:+7.2f} | Len: {ep_l:4d} | "
                        f"Route: {route*100:5.1f}% | Spd: {speed:5.1f} | {pstr}"
                    )
                    if wb_run is not None:
                        ep_log = {
                            "episode/reward":    ep_r,
                            "episode/length":    ep_l,
                            "episode/route_pct": route * 100,
                            "episode/speed_kmh": speed,
                        }
                        ep_log.update({f"episode/reward_{kk}": v for kk, v in details.items()})
                        wb_run.log(ep_log, step=global_step)

            # ── Live status line ──────────────────────────────────────────────
            pct        = global_step / ppo_cfg.total_timesteps * 100
            fps_now    = rollout_transitions / max(time.time() - rollout_t0, 1e-6)
            n_active   = len(vec_env.active_indices)
            buf_min    = min(env_ptrs)
            sys.stdout.write(
                f"\r  [{global_step:>8,}/{ppo_cfg.total_timesteps:,} {pct:4.1f}%] "
                f"EP {episode_count+1:3d} | "
                f"Active:{n_active}/{n_envs} | "
                f"FPS:{fps_now:6.1f} | "
                f"Buf:{buf_min}/{ppo_cfg.rollout_steps}"
                f"{'':10s}"
            )
            sys.stdout.flush()

        # ── Save obs reference (first rollout only) ───────────────────────────
        if obs_verifier is not None and not obs_ref_saved:
            obs_verifier.save(args.save_obs_ref)
            obs_ref_saved = True

        # ── GAE bootstrap ─────────────────────────────────────────────────────
        # imgs_np[i] / egos[i] = obs after each env's last stored transition.
        # For envs whose last step was done=True, the bootstrap value is
        # automatically masked to 0 by compute_gae, so stale obs is harmless.
        last_imgs_t = torch.from_numpy(imgs_np).to(device)
        last_egos_t = torch.from_numpy(egos).to(device)
        with torch.no_grad():
            last_values = policy.get_value(last_imgs_t, last_egos_t).cpu().numpy()  # (N,)

        buffer.compute_gae(last_values, ppo_cfg.gamma, ppo_cfg.gae_lambda)

        # ── One-shot distribution diagnostic (real obs, before first PPO update) ─
        if not _dist_diag_done:
            _dist_diag_done = True
            _n = min(64, buffer._total)
            _imgs_d = torch.from_numpy(
                buffer.imgs.reshape(buffer._total, *buffer.imgs.shape[2:])[:_n].copy()
            ).to(device)
            _egos_d = torch.from_numpy(
                buffer.egos.reshape(buffer._total, buffer.egos.shape[2])[:_n].copy()
            ).to(device)
            policy.eval()
            with torch.no_grad():
                _merged = policy._get_merged(_imgs_d, _egos_d)
                _il = policy.il_model
                _mu_s = _il.steer_mu_head(_merged)
                _nu_s = torch.clamp(_il.steer_nu_head(_merged) + 2.0, 2.0, 10.0)
                _mu_t = _il.throttle_mu_head(_merged)
                _nu_t = torch.clamp(_il.throttle_nu_head(_merged) + 2.0, 2.0, 10.0)
                _mu_sc = _mu_s.clamp(1e-6, 1.0 - 1e-6)
                _mu_tc = _mu_t.clamp(1e-6, 1.0 - 1e-6)
                _ent_s = torch.distributions.Beta(_mu_sc * _nu_s, (1 - _mu_sc) * _nu_s).entropy()
                _ent_t = torch.distributions.Beta(_mu_tc * _nu_t, (1 - _mu_tc) * _nu_t).entropy()
            print("\n" + "-" * 57)
            print(f"[DiagDist] steer_mu    mean={_mu_s.mean():.4f}  std={_mu_s.std():.4f}  min={_mu_s.min():.4f}  max={_mu_s.max():.4f}")
            print(f"[DiagDist] steer_nu    mean={_nu_s.mean():.4f}  max={_nu_s.max():.4f}")
            print(f"[DiagDist] throttle_mu mean={_mu_t.mean():.4f}  std={_mu_t.std():.4f}  min={_mu_t.min():.4f}  max={_mu_t.max():.4f}")
            print(f"[DiagDist] throttle_nu mean={_nu_t.mean():.4f}  max={_nu_t.max():.4f}")
            print(f"[DiagDist] entropy     steer={_ent_s.mean():.4f}  throttle={_ent_t.mean():.4f}  "
                  f"total={(_ent_s + _ent_t).mean():.4f}  (batch={_n}, before update {update_count + 1})")
            print("-" * 57 + "\n")
            del _merged, _il, _mu_s, _nu_s, _mu_t, _nu_t, _mu_sc, _mu_tc, _ent_s, _ent_t
            del _imgs_d, _egos_d

        # ── PPO update ────────────────────────────────────────────────────────
        policy.train()
        losses = ppo_update(policy, optimizer, buffer, ppo_cfg, scaler=amp_scaler)
        update_count += 1

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if update_count >= args.warmup_updates and not backbone_unfrozen:
            backbone_unfrozen = True
            for p in il_params:
                p.requires_grad_(True)
            print(f"[Warmup] IL model unfrozen at update {update_count} "
                  f"— CNN at lr={args.backbone_lr}, dist heads at lr={args.lr}.")

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
            f"Entropy: {losses['entropy']:.4f} | "
            f"KL: {losses['approx_kl']:.5f}\n"
        )
        if wb_run is not None:
            wb_run.log({
                "train/policy_loss":  losses["policy_loss"],
                "train/value_loss":   losses["value_loss"],
                "train/entropy":      losses["entropy"],
                "train/approx_kl":    losses["approx_kl"],
                "train/avg_reward":   avg_reward,
                "train/avg_route_pct": avg_route * 100,
                "train/avg_length":   avg_length,
                "train/fps":          fps,
            }, step=global_step)

        # ── Checkpoints ───────────────────────────────────────────────────────
        ckpt = {
            "policy":        policy.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "il_model_arch": args.arch,
            "image_size":    args.image_size,
            "global_step":   global_step,
            "update":        update_count,
            "route":         best_avg_route,
            "seed":          args.seed,
            "n_envs":        n_envs,
            "wandb_run_id":  wandb_run_id,
        }
        torch.save(ckpt, os.path.join(args.save_dir, "policy_latest.pth"))

        if avg_route > best_avg_route and len(episode_routes) >= 10:
            best_avg_route  = avg_route
            ckpt["route"]   = best_avg_route
            torch.save(ckpt, os.path.join(args.save_dir, "policy_best.pth"))
            print(f"  *** New best model (Avg Route: {best_avg_route*100:.1f}%)\n")

    # ── Done ──────────────────────────────────────────────────────────────────
    if wb_run is not None:
        wb_run.finish()
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
