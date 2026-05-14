# VisHybrid-X

An autonomous driving agent trained via **Behavioral Cloning (Imitation Learning)** on the [MetaDrive](https://github.com/metadriverse/metadrive) simulator. An expert policy collects driving demonstrations; a CNN policy network with depth estimation is trained offline to imitate it.

## Architecture

Three policy networks are available, selected with `--arch`:

| `--arch` | Network | Description |
|---|---|---|
| `simple` | `DrivingPolicyNet` | 3-layer CNN (8×8/4 → 4×4/2 → 3×3/1), single fusion head |
| `impala` | `ImpalaNet` | IMPALA-style residual CNN, 3 MaxPool stages (32→64→64 ch), dual output heads |
| `impala_v2` | `ImpalaNetV2` | Wider stages (48→96→96 ch), SE channel attention, deeper vis_proj (flat→1024→512), 256-d heads, ImageNet-normalised RGB |

All networks accept a **4-channel input** (depth + 3-ch RGB) at `image_size × image_size`, fuse a 512-d visual feature with a 32-d ego-state feature, and output `[steering, accel/brake]` ∈ [-1, 1].

**Ego state (EGO_DIM=3):** `[total_speed, last_steer, heading_delta]` — additional fields (forward/lateral speed, timestamp) are logged but not fed to the model.

**No BatchNorm in ImpalaNet/ImpalaNetV2** — intentional; BatchNorm is undefined at batch_size=1 during RL rollouts and causes train/eval stat drift.

### Key Features

**Asymmetric Braking Loss** — `custom_driving_loss` applies a **3× penalty** when the model misses a required braking action (`target < -0.1`). Braking events are rare but safety-critical; this counteracts class imbalance without rule-based hacks.

**Curriculum Lane Masking** — training starts with only lane markings visible (α=0, non-lane areas zeroed), then gradually blends full RGB back in over `curriculum_epochs`. Prevents over-reliance on lane colour cues.

**Depth Estimation** — `DepthEstimationModel` wraps [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) (ViT-based, `vits` encoder). Fine-tuned on MetaDrive frames via Scale-Shift Invariant Loss. Depth is inverted (`1 - normalised`) so closer objects have higher values. Precomputed and cached to disk before policy training for speed.

## Installation

```bash
# 1. Create environment
conda create -n metadrive-auto python=3.11 -y
conda activate metadrive-auto

# 2. Install PyTorch (adjust CUDA version: cu118, cu121, etc.)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install project requirements
pip install -r requirements.txt

# 4. Clone Depth-Anything-V2 (or use the submodule)
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2 && pip install -r requirements.txt && cd ..
```

For Google Colab: `pip install -r requirements_colab.txt`

Download the Depth-Anything-V2 checkpoint (`depth_anything_v2_vits.pth`) from the [official repo](https://github.com/DepthAnything/Depth-Anything-V2) and place it in `Depth-Anything-V2/checkpoints/`.

---

## Tutorial

A full walkthrough of every part of the pipeline, from raw environment to RL-fine-tuned agent.

### Part 1 — Check Your Setup

Before running anything, confirm the directory layout looks like this:

```
Metadrive Autonomous/
├── src/
├── models/                    ← checkpoints go here
├── dataset/                   ← expert data goes here (created by step 1)
├── data/processed/dpt_pred/   ← depth cache goes here (created by step 3)
├── Depth-Anything-V2/
│   └── checkpoints/
│       └── depth_anything_v2_vits.pth   ← must exist before training
└── Autonomous-Driving-RL/
    └── checkpoints/           ← RL checkpoints (created automatically)
```

Quick sanity check:
```bash
python -c "import metadrive; import torch; print('OK', torch.cuda.is_available())"
python -c "import sys; sys.path.insert(0,'src'); from models import build_policy; m=build_policy('impala',84); print('model OK', sum(p.numel() for p in m.parameters()), 'params')"
```

---

### Part 2 — Collect Expert Data

```bash
python src/generate_expert_dataset.py --episodes 100 --num_workers 4
# With GPU camera processing (faster):
python src/generate_expert_dataset.py --episodes 100 --image_on_cuda --num_workers 4
```

**What to expect:** A progress bar per worker. Each episode takes 5–30 s. Output lands in `dataset/train/`, `dataset/val/`, `dataset/test/` as `.npz` files. 100 episodes ≈ 50–80k frames total.

**What's in each file:** `rgb` (H×W×3 uint8), `action` (N×2 float32), `ego_state` (N×3), `ego_state_full` (N×5).

**Minimum viable dataset:** 30 episodes for a quick smoke-test; 100+ for meaningful training.

---

### Part 3 — Depth Model (Optional Fine-tune + Mandatory Precompute)

**Fine-tune** (optional, improves depth quality on MetaDrive frames):
```bash
python src/train_dpt.py --mode train --epochs 5 \
    --data_dir dataset --model_path models/dpt_finetuned.pth
```
Skip this if you want to move fast — the base DPT weights work fine.

**Precompute** (run this even if you skip fine-tuning):
```bash
# With fine-tuned weights:
python src/train_dpt.py --mode precompute \
    --model_path models/dpt_finetuned.pth \
    --data_dir dataset --out_dir data/processed/dpt_pred

# With base weights (no fine-tuning):
python src/train_dpt.py --mode precompute \
    --data_dir dataset --out_dir data/processed/dpt_pred
```

**What to expect:** Runs DPT on every frame once and saves the results. Takes 10–30 min on GPU. Subsequent policy training will be 5–10× faster because it skips live DPT inference.

**Output:** `data/processed/dpt_pred/train/`, `val/`, `test/` — each a `.npz` with `depth_pred`, `rgb`, `action`, `ego_state`.

---

### Part 4 — Train the IL Policy

```bash
python src/train_test_policy.py --mode train --epochs 30 \
    --pred_dir data/processed/dpt_pred \
    --model_path models/policy_model.pth \
    --arch impala --image_size 84
```

**Key flags:**
- `--arch impala` — recommended; `impala_v2` is stronger but slower
- `--image_size 84` — standard; higher (e.g. 112) gives more detail but more VRAM
- `--epochs 30` — enough to see convergence; add more if val loss is still dropping

**What to watch in the logs:**
```
Epoch  5/30 | train_loss: 0.142 | val_loss: 0.118 | steer_mae: 0.071 | brake_acc: 0.84
Epoch 10/30 | train_loss: 0.098 | val_loss: 0.089 | steer_mae: 0.054 | brake_acc: 0.91
```
- `val_loss` dropping = learning. If it rises for 5+ epochs → early stopping triggers.
- `steer_mae` < 0.08 is decent; < 0.05 is good.
- `brake_acc` > 0.85 means it's correctly detecting when to brake.

**Checkpoints saved:**
- `models/policy_model.pth` — latest epoch
- `models/policy_model_best.pth` — lowest val_loss so far ← use this one

**Fine-tune from a checkpoint** (e.g. after getting more data):
```bash
python src/train_test_policy.py --mode finetune \
    --finetune_from models/policy_model_best.pth \
    --model_path models/policy_finetuned.pth \
    --pred_dir data/processed/dpt_pred \
    --epochs 10 --lr 2e-5 --freeze_backbone
```

---

### Part 5 — Offline Evaluation

```bash
python src/train_test_policy.py --mode test \
    --model_path models/policy_model_best.pth \
    --pred_dir data/processed/dpt_pred
```

**Output example:**
```
steering_mae:       0.048
accel_mae:          0.071
steering_dir_acc:   0.923   ← turns in the right direction 92% of the time
brake_acc:          0.887   ← correctly brakes 89% of braking events
steering_corr:      0.941   ← Pearson correlation with expert
active_turn_mae:    0.063   ← MAE only on turns (|steer| > 0.05)
critical_turn_mae:  0.078   ← MAE on hard turns at speed
jitter_ratio:       1.12    ← 1.0 = same smoothness as expert; < 1.5 is good
```

These are offline metrics on the held-out test split. A model can look good here and still crash in simulation — always do the live test too.

---

### Part 6 — Live Simulation Test (IL)

Run from the repo root:
```bash
python src/test_simulation_policy.py \
    --model_path models/policy_model_best.pth \
    --dpt_path models/dpt_finetuned.pth \
    --arch impala --image_size 84 \
    --episodes 5
```

**What you see:** A MetaDrive window opens with the car driving. A second window shows the depth map (left) and RGB input (right) with an ego-state HUD at the top.

**Console output per episode:**
```
Episode 1 done. Reason: success | Route: 100.0% | Avg Spd: 0.41
Episode 2 done. Reason: out_of_road | Route: 34.2% | Avg Spd: 0.38
```

**Reasons:** `success` (reached destination), `out_of_road`, `crash_vehicle`, `crash_object`, `timeout/other`.

**Summary at the end:**
```
Success Rate:      40.0%
Route Completion:  67.3%
```
A freshly trained IL model might get 30–60% route completion. That's a reasonable starting point for RL.

---

### Part 7 — RL Fine-tuning (IL → RL)

Switch to the RL directory:
```bash
cd Autonomous-Driving-RL
```

**Start fine-tuning from your IL checkpoint:**
```bash
python train_rl.py \
    --il_checkpoint ../models/policy_model_best.pth \
    --arch impala --image_size 84 \
    --timesteps 1000000
```

**`--arch` and `--image_size` must exactly match what you used in IL training.**

**What you see on startup:**
```
[IL→RL] Loading full IL checkpoint (epoch=29, val_loss=0.0412): ../models/policy_model_best.pth
[IL→RL] Matched 18/18 parameter tensors.
[IL→RL] Missing (randomly init'd): ['value_head.0.weight', 'value_head.0.bias', 'value_head.2.weight', 'value_head.2.bias']
[IL→RL] value_head and log_std start from random init — this is correct.
```
The missing `value_head` keys are expected — the IL model had no critic. Every other weight carries over.

**What to watch in the logs:**
```
EP   1 | R:  +2.14 | Len:  143 | Route: 28.1% | Spd: 31.2 | route_progress: +3.12
EP   2 | R:  -8.43 | Len:   18 | Route:  3.8% | Spd: 12.1 | out_of_road: -10.00

Update  1 | Step  2,048/1,000,000 | FPS: 780 | Avg R(20): +1.2 | Avg Route: 18.3% | Entropy: 1.34
```
- **Early episodes**: expect crashes and short runs — that's normal
- **Avg Route** is the main metric; should trend upward over hundreds of updates
- **Entropy** should stay above ~0.5; if it collapses to ~0 the policy has stopped exploring
- **P_loss** oscillating around 0 is healthy; large sustained drift means something is wrong

**Resume after a crash or to run more steps:**
```bash
python train_rl.py \
    --rl_checkpoint checkpoints/policy_latest.pth \
    --arch impala --image_size 84 \
    --timesteps 2000000
```

**With visualisation** (slows training ~20%):
```bash
python train_rl.py --il_checkpoint ../models/policy_model_best.pth \
    --arch impala --image_size 84 --render
```

**Rough time guide:**

| Timesteps | GPU estimate | What to expect |
|---|---|---|
| 50K | ~30 min | Verify nothing crashes; route% may not improve yet |
| 200K | ~2 h | Basic behaviour visible; still erratic |
| 1M | ~8–12 h | Policy stabilising; route% should be higher than IL baseline |

---

### Part 8 — Test the RL Policy

**Quick test in the RL environment** (route completion + rewards):
```bash
python test_rl.py --checkpoint checkpoints/policy_best.pth --arch impala --scenarios 10
```

**Full simulation test** (same HUD as IL, works with RL checkpoints):
```bash
# from Autonomous-Driving-RL/:
python ../src/test_simulation_policy.py \
    --model_path checkpoints/policy_best.pth \
    --arch impala --image_size 84 \
    --episodes 5
```

The script auto-detects the RL checkpoint format and prints:
```
[INFO] RL checkpoint detected — extracted 18 IL backbone tensors.
```

---

### Quickest Path to a Working Agent

If you just want something driving as fast as possible:

```bash
# 1. Collect a small dataset (20 min)
python src/generate_expert_dataset.py --episodes 50 --num_workers 4

# 2. Precompute depth (10 min on GPU)
python src/train_dpt.py --mode precompute --data_dir dataset --out_dir data/processed/dpt_pred

# 3. Train IL for 20 epochs (30–60 min on GPU)
python src/train_test_policy.py --mode train --epochs 20 \
    --pred_dir data/processed/dpt_pred \
    --model_path models/policy_model.pth --arch impala --image_size 84

# 4. Test it live
python src/test_simulation_policy.py \
    --model_path models/policy_model_best.pth --arch impala --image_size 84

# 5. RL fine-tune for 500K steps (~4–6 h on GPU)
cd Autonomous-Driving-RL
python train_rl.py --il_checkpoint ../models/policy_model_best.pth \
    --arch impala --image_size 84 --timesteps 500000
```

## Pipeline

### 1. Collect Expert Data

```bash
python src/generate_expert_dataset.py --episodes 100 --image_on_cuda --num_workers 4
```

Parallel workers drive MetaDrive with its built-in expert policy. Auto-splits into train/val/test (80/10/10) on episode boundaries. Output: `.npz` files with keys `rgb`, `action`, `ego_state`, `ego_state_full`.

### 2. Fine-tune Depth Model (Optional)

```bash
python src/train_dpt.py --mode train --epochs 5 --data_dir dataset --model_path models/dpt_finetuned.pth
```

### 3. Precompute Depth Predictions

```bash
python src/train_dpt.py --mode precompute \
    --model_path models/dpt_finetuned.pth \
    --data_dir dataset \
    --out_dir data/processed/dpt_pred
```

Caches depth tensors to disk. Skipping live DPT inference during policy training provides a significant speedup.

### 4. Train Policy

```bash
python src/train_test_policy.py --mode train --epochs 30 \
    --pred_dir data/processed/dpt_pred \
    --model_path models/policy_model.pth \
    --arch impala --image_size 84
```

### 4b. Fine-tune an Existing Checkpoint

```bash
python src/train_test_policy.py --mode finetune \
    --finetune_from models/policy_model_best.pth \
    --model_path models/policy_finetuned.pth \
    --pred_dir data/processed/dpt_pred \
    --epochs 10 --lr 2e-5 --freeze_backbone
```

### 5. Offline Test

```bash
python src/train_test_policy.py --mode test \
    --model_path models/policy_model_best.pth \
    --pred_dir data/processed/dpt_pred
```

Reports steering/accel MAE & MSE, direction accuracy, braking accuracy, Pearson correlation, and predictive metrics (p95 steering error, active/critical turn MAE, jitter ratio, out-of-bounds rate).

### 6. Live Simulation Test

```bash
python src/test_simulation_policy.py \
    --model_path models/policy_model_best.pth \
    --dpt_path models/dpt_finetuned.pth \
    --arch impala --image_size 84
```

---

## RL Fine-tuning (IL → RL)

Once you have a trained IL model, you can continue fine-tuning it with PPO from the `Autonomous-Driving-RL/` directory.

### How it works

The `ILActorCritic` wrapper loads any IL policy backbone and adds a critic (value head) on top of its shared 544-d feature vector. The actor reuses the IL model's existing `steer_head` / `accel_head`, so IL-learned weights are preserved and fine-tuned by PPO. Only the value head starts from random init — that's expected.

### 7. RL Fine-tuning

```bash
cd Autonomous-Driving-RL

# Fine-tune from IL checkpoint (recommended)
python train_rl.py \
    --il_checkpoint ../models/policy_model_best.pth \
    --arch impala --image_size 84 \
    --timesteps 1000000

# Watch it drive while training
python train_rl.py \
    --il_checkpoint ../models/policy_model_best.pth \
    --arch impala --image_size 84 \
    --render

# Resume a previous RL run
python train_rl.py \
    --rl_checkpoint checkpoints/policy_latest.pth \
    --arch impala --image_size 84
```

`--arch` and `--image_size` must match what was used during IL training.

### 8. Test RL Policy

```bash
# Test within the RL environment (route completion, rewards)
python test_rl.py --checkpoint checkpoints/policy_best.pth --arch impala

# Test in the full IL simulation (HUD, depth vis, metrics)
python ../src/test_simulation_policy.py \
    --model_path checkpoints/policy_best.pth \
    --arch impala --image_size 84
```

`test_simulation_policy.py` auto-detects RL checkpoints and strips the actor-critic wrapper to load just the IL backbone weights.

### RL Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--il_checkpoint` | `None` | IL model to start from (IL→RL) |
| `--rl_checkpoint` | `None` | Previous RL checkpoint to resume |
| `--arch` | `impala` | Must match the IL training arch |
| `--image_size` | `84` | Must match the IL training image size |
| `--timesteps` | `200000` | Total env steps (1M+ recommended) |
| `--rollout` | `2048` | Steps per PPO update |
| `--lr` | `3e-4` | Learning rate |
| `--scenarios` | `50` | Map variety (higher = harder generalisation) |
| `--dpt_path` | `None` | Fine-tuned DPT weights (optional) |
| `--render` | off | Open MetaDrive 3D window while training |

### Checkpoints

RL checkpoints are saved to `Autonomous-Driving-RL/checkpoints/`:
- `policy_latest.pth` — saved every PPO update
- `policy_best.pth` — saved when average route completion improves

Both IL and RL checkpoints are interchangeable in `test_simulation_policy.py`.

---

## Key Arguments

| Script | Argument | Description |
|---|---|---|
| `generate_expert_dataset.py` | `--episodes` | Number of expert episodes to collect |
| | `--image_on_cuda` | Process camera data on GPU (recommended) |
| | `--num_workers` | Parallel collection workers |
| `train_dpt.py` | `--mode` | `train` or `precompute` |
| `train_test_policy.py` | `--arch` | `simple`, `impala`, or `impala_v2` |
| | `--pred_dir` | Path to precomputed depth cache |
| | `--image_size` | Input resolution (default: 84) |
| | `--fully_masked_epochs` | Epochs with lane-only input before curriculum starts |
| | `--curriculum_epochs` | Epochs to blend from lane-only to full RGB |
| | `--freeze_backbone` | Freeze CNN during fine-tuning |

## Project Structure

```
src/
├── generate_expert_dataset.py  # Expert data collection
├── train_dpt.py                # Depth model fine-tuning & precomputation
├── train_test_policy.py        # Policy training, fine-tuning, offline test
├── test_simulation_policy.py   # Live MetaDrive simulation test (IL + RL)
├── models.py                   # All network definitions + ego-state utilities
├── data/
│   └── cameras.py              # Multi-camera rig configuration
├── policy/
│   ├── datasets.py             # MetaDriveRGBDataset, PrecomputedDepthDataset
│   ├── losses.py               # custom_driving_loss, offline/predictive metrics
│   └── trainer.py              # Epoch loop, lane masking, curriculum logic
└── utils/
    ├── checkpoints.py          # save/load checkpoint, freeze backbone
    └── fps.py                  # FPS measurement

Autonomous-Driving-RL/
├── train_rl.py                 # PPO training entry point
├── test_rl.py                  # RL policy test
└── rl/
    ├── il_actor_critic.py      # ILActorCritic: IL backbone + value head
    ├── env_wrapper.py          # MetaDrive wrapper (IL-correct depth + ego state)
    ├── ppo.py                  # PPO algorithm + rollout buffer
    └── rewards.py              # Modular reward/penalty system
```
