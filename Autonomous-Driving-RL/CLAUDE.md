# CLAUDE.md — Autonomous-Driving-RL

PPO reinforcement-learning fine-tuning of an IL policy in MetaDrive. Run every command from this directory (`Autonomous-Driving-RL/`). The parent IL repo lives one level up (`../`).

## Commands

### Train
```bash
# Fine-tune from an IL checkpoint (4 parallel envs — recommended)
python train_rl.py --il_checkpoint ../models/policy_model_best.pth --arch impala --image_size 84 --n_envs 4

# Single-env (backward-compatible, no subprocess overhead, useful for debugging)
python train_rl.py --il_checkpoint ../models/policy_model_best.pth --arch impala --image_size 84

# Resume a previous RL run
python train_rl.py --rl_checkpoint models/rl/policy_latest.pth --arch impala --n_envs 4

# Mixed-precision + compiled backbone (faster, CUDA only, PyTorch ≥2.0)
python train_rl.py --il_checkpoint ../models/policy_model_best.pth --arch impala --amp --compile --n_envs 4

# Save obs reference for pipeline verification (see Verification section)
python train_rl.py --il_checkpoint ../models/policy_model_best.pth \
    --save_obs_ref models/rl/obs_ref.json
```

### Test
```bash
# Deterministic rollout (uses action mean, no sampling)
python test_rl.py --checkpoint models/rl/policy_best.pth --arch impala

# Test and verify obs preprocessing matches training
python test_rl.py --checkpoint models/rl/policy_best.pth --arch impala \
    --obs_ref models/rl/obs_ref.json
```

### Obs-pipeline verification (standalone)
```bash
# Check that simulation sees the same input distribution as training did
python verify_simulation.py --checkpoint models/rl/policy_best.pth \
    --obs_ref models/rl/obs_ref.json --steps 500

# Print obs stats only (no comparison)
python verify_simulation.py --checkpoint models/rl/policy_best.pth --steps 200
```

**Always pass the same `--arch` and `--image_size` used during IL training.**

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--il_checkpoint` | `None` | IL checkpoint to start from |
| `--rl_checkpoint` | `None` | RL checkpoint to resume |
| `--arch` | `impala` | `simple`, `impala`, or `impala_v2` |
| `--image_size` | `84` | Input resolution — must match IL |
| `--timesteps` | `200000` | Total env steps |
| `--n_envs` | `1` | Parallel env workers; `1`=DummyVecEnv, `N>1`=SubprocVecEnv. Total transitions/update = `n_envs × rollout` |
| `--rollout` | `2048` | Steps *per env* per PPO update |
| `--batch` | `64` | Mini-batch size |
| `--epochs` | `10` | PPO epochs per rollout |
| `--lr` | `3e-4` | LR for value_head (random-init RL critic) |
| `--backbone_lr` | `1e-5` | LR for IL backbone (much lower) |
| `--warmup_updates` | `5` | PPO updates to keep backbone frozen |
| `--scenarios` | `50` | Number of MetaDrive maps |
| `--dpt_path` | `None` | Fine-tuned DPT weights |
| `--render` | off | Open MetaDrive 3D window |
| `--save_dir` | `models/rl` | Checkpoint output directory |
| `--amp` | off | Mixed-precision PPO update (CUDA only) |
| `--compile` | off | `torch.compile` the IL backbone |
| `--save_obs_ref` | `None` | Save obs stats JSON after first rollout |

## Architecture

### ILActorCritic (`rl/il_actor_critic.py`)
Wraps any IL backbone as a PPO actor-critic without modifying IL weights:

```
IL backbone (ImpalaNet / ImpalaNetV2 / DrivingPolicyNet)
    visual CNN         → 512-d  ┐
    ego MLP            →  32-d  ┘  concat → 544-d merged feature
                                        ├── steer_head / throttle_head (IL weights, fine-tuned; Tanh-bounded mean)
                                        └── value_head → V(s)          (new, random init)

Action distribution: Normal(mean, std) — mean from steer_head/throttle_head,
std fixed at [0.05, 0.05] (non-learned buffer, never optimized).
```

The IL model owns steer_head/throttle_head and outputs a plain point
estimate. `ILActorCritic._get_dist()` wraps that estimate as the mean of a
fixed-covariance Normal — IL's learned point estimate is preserved at RL
start, and RL adds exploration noise on top via a fixed, non-learned std.
Only `value_head` is new.

LR groups: `backbone_params = il_model.*` (all IL weights including distribution heads) at `backbone_lr`;
`head_params = value_head.*` at `lr`.

- `load_from_il_checkpoint(path, device)` — loads `ckpt["model"]` with `strict=False`; value_head stays at random init (expected and correct)
- `get_action_and_value(image, ego, action=None)` — for rollout (sample) or update (evaluate)
- `get_value(image, ego)` — critic-only, for GAE bootstrap

### Environment (`rl/env_wrapper.py`)
Produces the **same 4-channel observation as IL training**:
- **Channel 0**: inverted depth — `1 - (raw - min)/(max - min + 1e-6)` so closer = higher
- **Channels 1-3**: RGB ∈ [0, 1] (MetaDrive outputs RGB natively, no BGR swap)
- **Ego state** (separate array, EGO_DIM=3): `[total_speed, last_steer, heading_delta]`

Depth is computed with the same `DepthEstimationModel` as the IL pipeline. If you change this formula, obs stats will shift and the policy will perform badly — use `verify_simulation.py` to catch this.

### PPO (`rl/ppo.py`)

**`PPOConfig`** — all hyperparameters. Set `use_amp=True` when using `--amp`.

**`RolloutBuffer`** — stores `(4, H, W)` images and `(EGO_DIM,)` ego vectors separately as pinned-memory numpy arrays. `get_batches()` uses `torch.from_numpy + non_blocking=True` for async H2D transfers.

**`ppo_update(policy, optimizer, buffer, cfg, scaler=None)`** — standard PPO:
1. Advantage normalisation per mini-batch
2. Clipped surrogate loss (ε=0.2)
3. Value loss (MSE, coef=0.5)
4. Entropy bonus (coef=0.01)
5. Grad clip (max_norm=0.5)

When `cfg.use_amp=True` and a `GradScaler` is passed, wraps the forward in `torch.autocast("cuda")` and uses `scaler.unscale_()` before grad-clip.

### Rewards (`rl/rewards.py`)
`RewardConfig` + `compute_reward(info, action, prev_route, speed, cfg, prev_action)`.

| Component | Default | Notes |
|---|---|---|
| Route progress | `×100` per Δ | Primary signal every step |
| Arrive dest | `+50` | Strong sparse terminal bonus |
| Out of road | `−10` | Terminal |
| Crash vehicle | `−20` | Terminal, heaviest |
| Crash object | `−10` | Terminal |
| Harsh steering | `−0.1 × (|steer|−0.3) × speed_factor` | Only above 0.3 threshold; speed-scaled |
| Steering jerk | `−0.2 × Δsteer × speed_factor` | speed_factor = min(speed,40)/40 |
| Speed bonus (>5 km/h) | `+0.1 × min(speed,40)/40` | Per step |
| Standing still | `−0.05` | Per step at or below 5 km/h |
| Lateral offset | `−0.3 × excess × (1 − \|steer\|/0.15)` | Passive drift penalised; suppressed when actively steering (passing/avoidance) |

To add a term: add a field to `RewardConfig` and a block inside `compute_reward()` — nothing else changes.

## Checkpoint Format

```python
{
    "policy":        ILActorCritic.state_dict(),  # il_model.*, value_head.*, action_std (buffer, not optimized)
    "il_model_arch": "impala",
    "image_size":    84,
    "global_step":   int,
    "route":         float,   # best avg route completion seen so far
}
```

`models/rl/policy_latest.pth` — saved every PPO update.  
`models/rl/policy_best.pth` — saved when avg route (last 20 eps) improves, requires ≥10 episodes.

To run an RL checkpoint in the IL sim tester:
```bash
python ../src/test_simulation_policy.py \
    --model_path models/rl/policy_best.pth --arch impala --image_size 84
```
`test_simulation_policy.py` auto-detects the `"policy"` key and strips the `il_model.` prefix.

## Observation Pipeline Verification

Optimisations and `env_wrapper.py` edits can silently break the obs preprocessing (depth inversion, normalisation, channel order), causing the policy to see different inputs than it was trained on.

**Workflow:**
```bash
# Step 1 — during training, save a reference
python train_rl.py ... --save_obs_ref models/rl/obs_ref.json

# Step 2 — after any env/optimisation change, verify
python verify_simulation.py --checkpoint models/rl/policy_best.pth \
    --obs_ref models/rl/obs_ref.json
```

`verify_simulation.py` exits with code 1 on mismatch (CI-friendly). Uses `rl/obs_verifier.py` — an `ObsVerifier` instance can also be used inline: `verifier.record(img, ego)` then `verifier.compare("obs_ref.json")`.

## Key Design Decisions

- **No BatchNorm in ImpalaNet** — BatchNorm is undefined at batch_size=1 during RL rollouts and causes train/eval stat drift. Do not add it.
- **Backbone warmup freeze** — backbone frozen for first `--warmup_updates` PPO updates so the critic reaches sensible values before backbone gradients flow. Prevents early noisy advantages from corrupting IL representations.
- **Separate LRs** — backbone gets `1e-5` (preserve IL features); value_head gets `3e-4` (learn fast, random init); dist heads (steer_head/throttle_head) get `dist_head_lr` (shared with IL, must move slowly). Never merge into a single LR.
- **`strict=False` when loading IL checkpoint** — only value_head is not in the IL file; it stays at random init. steer_head/throttle_head load directly (no conversion needed — IL and RL share the same heads).
- **Fixed, non-learned action std** — RL exploration noise (`action_std` buffer, `[0.05, 0.05]`) is not optimized; only the mean (from the shared IL heads) is fine-tuned during PPO.
- **Inverted depth** (`1 - normalised`) — IL convention; closer objects have higher values. The RL env wrapper must reproduce this exactly. Verify with `verify_simulation.py` after any env change.
- **Pinned buffer memory** — `RolloutBuffer` allocates float32 arrays in page-locked memory when CUDA is available, enabling async H2D DMA.
- **`rl/policy.py`** — legacy `ActorCritic` class (2-channel depth+lane input). Not used by the current pipeline; kept for reference only.

## File Structure

```
Autonomous-Driving-RL/
├── train_rl.py             # Training entry point (PPO)
├── test_rl.py              # Deterministic test rollout
├── verify_simulation.py    # Obs-pipeline consistency check
└── rl/
    ├── il_actor_critic.py  # ILActorCritic: IL backbone + PPO heads
    ├── env_wrapper.py      # MetaDrive env, DPT depth, ego state; use_depth_model=False for workers
    ├── vec_env.py          # SubprocVecEnv, DummyVecEnv, MakeEnvFn — parallel env management
    ├── ppo.py              # PPOConfig, RolloutBuffer (n_envs-aware), ppo_update
    ├── rewards.py          # RewardConfig + compute_reward
    ├── obs_verifier.py     # ObsVerifier: record/compare obs stats
    └── policy.py           # LEGACY — not used in current pipeline
```
