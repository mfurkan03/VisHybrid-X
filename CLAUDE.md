# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Autonomous driving agent trained via **Behavioral Cloning (Imitation Learning)** on the [MetaDrive](https://github.com/metadriverse/metadrive) simulator, with optional **PPO Reinforcement Learning fine-tuning**. The IL pipeline collects expert demonstrations and trains a CNN policy offline; the RL pipeline then fine-tunes it with environment rewards from `Autonomous-Driving-RL/`.

## Pipeline

The project runs in sequential stages:

**1. Collect expert data**
```bash
python src/generate_expert_dataset.py --episodes 100 --image_on_cuda --num_workers 4
```

**2. Fine-tune depth model (optional)**
```bash
python src/train_dpt.py --mode train --epochs 5 --data_dir dataset --model_path models/dpt_finetuned.pth
```

**3. Precompute depth predictions**
```bash
python src/train_dpt.py --mode precompute --model_path models/dpt_finetuned.pth \
    --data_dir dataset --out_dir data/processed/dpt_pred
```

**4. Train policy**
```bash
python src/train_test_policy.py --mode train --epochs 30 \
    --pred_dir data/processed/dpt_pred \
    --model_path models/policy_model.pth \
    --arch impala --image_size 84
```

Add `--benchmark` to run the same training over 5 seeds (0–4), saving checkpoints as `policy_model_seed_N.pth`. Augmentation flags: `--aug_pixel_noise` (default 1.0), `--aug_hflip` (default 0.5), `--aug_grayscale` (default 0.1); pass `0` to disable any augmentation.

**4b. Fine-tune an existing checkpoint**
```bash
python src/train_test_policy.py --mode finetune \
    --finetune_from models/policy_model_best.pth \
    --model_path models/policy_finetuned.pth \
    --pred_dir data/processed/dpt_pred \
    --epochs 10 --lr 2e-5 --freeze_backbone
```

**5. Offline test**
```bash
python src/train_test_policy.py --mode test \
    --model_path models/policy_model_best.pth \
    --pred_dir data/processed/dpt_pred
```

**6. Live simulation test**
```bash
python src/test_simulation_policy.py --model_path models/policy_model_best.pth \
    --dpt_path models/dpt_finetuned.pth --arch impala --image_size 84
```

`test_simulation_policy.py` accepts both IL checkpoints (`ckpt["model"]`) and RL checkpoints (`ckpt["policy"]`). For RL checkpoints it strips the `il_model.` prefix automatically.

**7. RL fine-tuning (from `Autonomous-Driving-RL/`)**
```bash
python train_rl.py --il_checkpoint ../models/policy_model_best.pth --arch impala --image_size 84 --timesteps 1000000
```

**8. RL simulation test**
```bash
python test_rl.py --checkpoint checkpoints/policy_best.pth --arch impala
# or full sim:
python ../src/test_simulation_policy.py --model_path checkpoints/policy_best.pth --arch impala --image_size 84
```

There is no traditional test suite. Evaluation happens either offline (metrics computed over the test split) or online (live MetaDrive environment).

## Architecture

### Policy Networks (`src/models.py`)

Four architectures, selected with `--arch`:

- **`DrivingPolicyNet`** (`--arch simple`) — plain 3-layer CNN (conv 8×8/4 → 4×4/2 → 3×3/1), single fusion head. Simpler but less expressive.
- **`ImpalaNet`** (`--arch impala`) — IMPALA-style residual CNN, no BatchNorm. Three MaxPool stages (32→64→64 ch) with pre-activation residual blocks → 512-d shared visual feature. Dual output heads: `steer_head` and `throttle_head` each specialise from the shared visual representation.
- **`ImpalaNetV2`** (`--arch impala_v2`) — Stronger IMPALA variant for better RGB handling. Wider stages (48→96→96 ch), SE (Squeeze-and-Excitation) channel attention in every res-block, deeper vis_proj (flat→1024→512), larger heads (256-d). RGB channels (1-3) are ImageNet-normalised inside `forward()`; depth channel (0) is left as-is.
- **`ImpalaNetV2AB`** (`--arch impala_v2_ab`) — Same backbone as `ImpalaNetV2`, but steer/throttle each get `alpha_head`/`beta_head` (Softplus) producing `Beta(alpha, beta)` concentration parameters directly instead of a Tanh point estimate. `forward()` returns `(alpha, beta)` — each `(B, 2)` — instead of a single `(B, 2)` action tensor. Trained with `custom_driving_loss_beta` (Beta NLL); the point-estimate action used for metrics/inference is the distribution mean `alpha / (alpha + beta)` mapped from `[0,1]` back to `[-1,1]`. **IL-only** — not wired into the RL pipeline (`ILActorCritic` still assumes a single Tanh action per head).

`DrivingPolicyNet`, `ImpalaNet`, and `ImpalaNetV2` (the plain-regression architectures):
- Accept **4-channel input** (depth + 3-ch RGB) at `image_size × image_size`
- Fuse a 512-d visual feature with a 32-d ego-state feature → action output `[steering, accel/brake]` ∈ [-1, 1]
- Use `build_policy(arch, image_size)` as the factory
- Support `forward(x, ego, return_features=True)` to expose the 544-d merged feature vector for the RL critic (backwards-compatible; default is `False`)
- IL training fits `steer_head`/`throttle_head` directly with a regression loss (`custom_driving_loss`, Smooth L1 + asymmetric braking penalty) — the model makes a plain point-estimate prediction, no learned distribution.

`ImpalaNetV2AB` shares the same input/ego/`return_features` API but returns `(alpha, beta)` from `forward()` instead of an action tensor — callers (`trainer.py::run_epoch`, `train_test_policy.py::test_policy`, `test_simulation_policy.py`) branch on `isinstance(pred, tuple)` to pick the Beta-NLL loss and Beta-mean point estimate.

Ego input (`EGO_DIM=3`): `[total_speed, last_steer, heading_delta]` from `extract_ego_state()`. Three additional fields (forward/lateral speed, timestamp) are logged in `ego_state_full` but not fed to the model.

### Depth Estimation (`src/models.py`, `src/train_dpt.py`)
- `DepthEstimationModel` wraps [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) (ViT-based, `vits` encoder by default)
- Fine-tuned on MetaDrive frames via **Scale-Shift Invariant Loss**
- Depth is inverted (`1 - normalised`) so closer objects have higher values
- At training time, depth is precomputed and cached (`PrecomputedDepthDataset` in `src/policy/datasets.py`)

### Lane Mask (`src/policy/trainer.py`)
- `get_lane_mask_visual()` zeroes the top 55% of the image (sky/far road), then applies a **grayscale threshold ≥ 180** to detect bright lane markings
- `apply_lane_mask()` blends the result with a curriculum α:
  - `epoch < fully_masked_epochs` → α=0 (only lane markings visible, non-lane areas zeroed)
  - `fully_masked_epochs ≤ epoch < fully_masked_epochs + curriculum_epochs` → α ramps 0→1
  - `epoch ≥ fully_masked_epochs + curriculum_epochs` → α=1 (full RGB)
  - `always_lane_masked=True` (e.g. for teacher training) → α=0 always
- The final 4-ch tensor (depth + blended RGB) is interpolated to `image_size` at the end of `apply_lane_mask`

### Loss Functions (`src/policy/losses.py`)
- `custom_driving_loss`: Smooth L1 with **3× asymmetric braking penalty** on the accel channel when `target < -0.1`
- `compute_offline_metrics`: steering/accel MAE & MSE, direction accuracy, braking accuracy, Pearson correlation
- `compute_predictive_metrics` (val/test only, requires temporal order):
  - `steer_p95_error` — 95th-percentile steering error
  - `active_turn_mae` — MAE only on turns (`|steer| > 0.05`)
  - `critical_turn_mae` — MAE on hard turns (`|steer| > 0.1` and `speed > 0.3`)
  - `jitter_ratio` — model vs. expert frame-to-frame steering change ratio
  - `out_of_bounds_rate` — % of predictions outside expert range
  - `speed_weighted_steer_mae`, `pre_brake_anticipation`
- `compute_heading_metrics`: requires `ego_state_full` (5-dim); estimates heading divergence over rolling windows

### Data Collection (`src/generate_expert_dataset.py`)
- Parallel workers drive MetaDrive with its built-in expert policy
- Multi-camera rig configured in `src/data/cameras.py`
- Auto-splits into train/val/test (80/10/10) on episode boundaries
- Output: `.npz` files with keys `rgb`, `action`, `ego_state`, `ego_state_full`

### Datasets (`src/policy/datasets.py`)
- `MetaDriveRGBDataset` — loads raw RGB `.npz` files; live DPT inference during training
- `PrecomputedDepthDataset` — loads cached depth + raw RGB; lane masks computed at training time. Rejects old-format files missing the `rgb` key with a clear re-run instruction.

### Utilities (`src/utils/`)
- `checkpoints.py` — `save_checkpoint`, `load_checkpoint`, `freeze_backbone`, `print_trainable_params`
- `fps.py` — FPS measurement

## RL Pipeline (`Autonomous-Driving-RL/`)

### ILActorCritic (`rl/il_actor_critic.py`)
Wraps any IL policy as a PPO actor-critic without modifying the IL model:
- **Actor**: reuses IL model's `steer_head` / `throttle_head`
- **Critic**: new `value_head = Linear(544→128→ReLU→1)` — random init, not in IL checkpoint
- **Distribution**: `Normal(action_mean, action_std)` where `action_std` is a fixed buffer `[0.05, 0.05]` (non-learned)
- `load_from_il_checkpoint(path)`: loads `ckpt["model"]` into `self.il_model` with `strict=False`; prints matched/missing keys

### RL Environment (`rl/env_wrapper.py`)
- Uses IL's `DepthEstimationModel` for correct inverted depth (`1 - normalized`) — not the buggy per-frame normalization from the original RL repo
- MetaDrive's RGBCamera outputs RGB natively — no BGR conversion applied
- Returns `(img_np (4, H, H), ego_np (3,))` tuple, matching IL model input format
- Tracks `last_steer` for ego state extraction via `extract_ego_state()`

### PPO (`rl/ppo.py`)
- `RolloutBuffer` stores `imgs` and `egos` as separate arrays (not concatenated)
- Standard GAE + clipped surrogate loss + entropy bonus + gradient clipping

### Rewards (`rl/rewards.py`)
- Route progress (primary), arrival bonus, crash/off-road penalties, speed bonus, steering penalty
- Harsh steering penalty is uniform (`-0.5 × |steer|`), not speed-scaled

### RL Checkpoint Format
```python
{"policy": ILActorCritic.state_dict(), "il_model_arch": str, "image_size": int, "global_step": int, "route": float}
```
Keys inside `"policy"`: `il_model.*` (backbone, including steer_head/throttle_head), `value_head.*` (critic), `action_std` (fixed exploration buffer).

## Key Design Decisions

- **No BatchNorm in ImpalaNet** — intentional; BatchNorm is undefined at batch_size=1 during RL rollouts and causes train/eval stat drift
- **Depth cached to disk** before policy training — depth inference is expensive; caching lets training iterate quickly without re-running DPT every epoch
- **Asymmetric braking loss** — braking events are rare but safety-critical; the 3× multiplier counteracts class imbalance without rule-based hacks
- **Curriculum lane masking** — starting with α=0 (clean, colour-independent representation) and gradually blending in full RGB prevents over-reliance on lane colour cues
- **Dual heads in ImpalaNet** — steering and acceleration specialise independently from the shared 512-d visual feature; the depth channel lets steering see obstacles
- **`return_features` in all policy forward()** — exposes the 544-d merged feature for the RL value head without changing IL inference (default `False`)
- **ILActorCritic wraps, not replaces** — IL weights load directly with `strict=False`; only `value_head` is new (random init); steer_head/throttle_head are IL-owned and fine-tuned during RL with a shared low learning rate

## Dependencies

Install with:
```bash
pip install -r requirements.txt
```

For Google Colab:
```bash
pip install -r requirements_colab.txt
```

Core dependencies: `torch`, `torchvision`, `metadrive-simulator`, `Depth-Anything-V2` (submodule in `Depth-Anything-V2/`), `numpy`, `scipy`.
