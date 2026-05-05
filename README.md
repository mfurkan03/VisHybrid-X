# Metadrive Autonomous

An autonomous driving agent trained via **Behavioral Cloning (Imitation Learning)** on the [MetaDrive](https://github.com/metadriverse/metadrive) simulator. An expert policy collects driving demonstrations; a CNN policy network with depth estimation is trained offline to imitate it.

## Architecture

Three policy networks are available, selected with `--arch`:

| `--arch` | Network | Description |
|---|---|---|
| `simple` | `DrivingPolicyNet` | 3-layer CNN (8×8/4 → 4×4/2 → 3×3/1), single fusion head |
| `impala` | `ImpalaNet` | IMPALA-style residual CNN, 3 MaxPool stages (32→64→64 ch), dual output heads |
| `impala_v2` | `ImpalaNetV2` | Wider stages (48→96→96 ch), SE channel attention, deeper vis_proj (flat→1024→512), 256-d heads, ImageNet-normalised RGB |

All networks accept a **4-channel input** (depth + 3-ch RGB) at `image_size × image_size`, fuse a 512-d visual feature with a 32-d ego-state feature, and output `[steering, accel/brake]` ∈ [-1, 1].

**Ego state (EGO_DIM=2):** `[total_speed, last_steer]` — additional fields (forward/lateral speed, heading delta, timestamp) are logged but not fed to the model.

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
    --arch impala
```

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
├── test_simulation_policy.py   # Live MetaDrive simulation test
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
```
