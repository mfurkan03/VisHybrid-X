# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Autonomous driving agent trained via **Behavioral Cloning (Imitation Learning)** on the [MetaDrive](https://github.com/metadriverse/metadrive) simulator. An expert policy collects driving demonstrations; a CNN policy network (with depth estimation) is trained offline to imitate it. A **teacher-student knowledge distillation** framework further improves robustness.

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

**4a. Train policy (single-phase baseline)**
```bash
python src/train_offline_policy.py --mode train --epochs 30 --pred_dir data/processed/dpt_pred \
    --model_path models/policy_model.pth --policy deep --curriculum_epochs 40
```

**4b. Train policy (teacher-student, two-phase)**
```bash
# Phase 1 — train teacher on clean lane-masked input
python src/train_teacher_student.py --mode train_teacher \
    --pred_dir data/processed/dpt_pred \
    --teacher_path models/teacher.pth --epochs 30

# Phase 2 — train student with curriculum + distillation
python src/train_teacher_student.py --mode train_student \
    --pred_dir data/processed/dpt_pred \
    --teacher_path models/teacher_best.pth \
    --student_path models/student.pth \
    --epochs 70 --lambda_output 1.0 --lambda_feature 0.1
```

**Test in simulation**
```bash
python src/test_simulation_policy.py --model_path models/policy_model.pth --dpt_path models/dpt_finetuned.pth
```

There is no traditional test suite. Evaluation happens either offline (metrics computed over the test split) or online (live MetaDrive environment).

## Architecture

### Policy Network (`src/models.py`)
- `DrivingPolicyNet` — IMPALA-style CNN with residual blocks, no BatchNorm
- Outputs `[steering, accel/brake]` ∈ [-1, 1]
- Input is a **4-channel image** (depth + 3-ch RGB), processed by two independent streams (one for steering, one for acceleration) — the dual-stream design lets the steering branch see obstacles through the depth channel
- 512-d visual bottleneck used for feature distillation

### Teacher-Student Distillation (`src/policy/distillation.py`, `src/train_teacher_student.py`)
- **Teacher**: trained on depth + fully lane-masked RGB (α=0 always); learns a clean, colour-independent representation
- **Student**: trained with curriculum-blended input (α transitions from 0 → 1 over `curriculum_epochs`) and supervised by three losses:
  - `L_imitation`: smooth-L1 vs. ground-truth expert actions
  - `L_output_dist`: smooth-L1 vs. teacher's predicted actions (soft targets)
  - `L_feature_dist`: MSE between student and teacher 512-d visual bottlenecks
- `lambda_output` and `lambda_feature` control the distillation strength (set to 0 to recover the baseline)
- Student CSV additionally logs per-epoch `loss_imitation`, `loss_output_dist`, `loss_feature_dist`

### Depth Estimation (`src/models.py`, `src/train_dpt.py`)
- Wraps [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) (ViT-based)
- Fine-tuned on MetaDrive frames via **Scale-Shift Invariant Loss**
- At training time, depth is precomputed and cached to disk (`PrecomputedDepthDataset` in `src/policy/datasets.py`)

### Lane Mask (`src/policy/trainer.py`)
- `_batch_lane_mask()` detects both **white and yellow** lane markings:
  - White: grayscale ≥ 180/255
  - Yellow: `r > g + 0.02`, `r > b`, `(r - g) < 0.35`, `r > 0.28`
- Applied **every epoch** (not just during curriculum) to prevent catastrophic forgetting
- Schedule: `fully_masked_epochs` (α=0) → `curriculum_epochs` (α: 0→1) → post-curriculum (α=1 with random masking at `lane_mask_prob`)
- Teacher always uses α=0 via `always_lane_masked=True`

### Loss Functions (`src/policy/losses.py`)
- `custom_driving_loss`: Smooth L1 with **3× asymmetric braking penalty** (prevents ignoring braking events)
- Offline metrics: steering/accel MAE, braking accuracy, direction accuracy
- Predictive metrics (`compute_predictive_metrics`):
  - `steer_95th_pctl_err` — catches catastrophic steering mistakes
  - `active_turn_mae` — MAE only on turns (steering > 0.05), ignores straight-line driving
  - `jitter_ratio` — ratio of model vs. expert frame-to-frame steering changes (detects oscillation)
  - `out_of_bounds_rate` — % of predictions outside the expert's observed range

### Metrics Logging (`src/policy/trainer.py`, `src/policy/distillation.py`)
- Every training run writes `{model_path}_metrics.csv` with full epoch-by-epoch history
- Columns: `epoch`, `train_loss`, `train_steering_mae`, `val_loss`, `val_steering_mae`, `val_steering_dir_acc`, `val_brake_acc`, `val_active_turn_mae`, `val_jitter_ratio`, `val_steer_95th_pctl_err`, `lr`

### Data Collection (`src/generate_expert_dataset.py`)
- Parallel workers drive MetaDrive with its built-in expert policy
- Multi-camera rig configured in `src/data/cameras.py` (arbitrary angles, GPU/CPU processing)
- Auto-splits into train/val/test (80/10/10) on episode boundaries

### Utilities (`src/utils/`)
- `visualize_lane_mask.py` — inspect lane mask quality on dataset frames
- `visualize_test_predictions.py` — compare model predictions vs. expert on test split
- `checkpoints.py` — save/load checkpoint handling
- `fps.py` — FPS measurement

## Key Design Decisions

- **Teacher-student over single policy** — distillation lets the student learn from curriculum-blended input while the frozen teacher provides clean lane-masked supervision, improving generalisation
- **No BatchNorm** in `DrivingPolicyNet` — intentional, for stability when the model may later be fine-tuned with RL
- **Depth cached to disk** before policy training — depth inference is expensive; caching lets training iterate quickly
- **Asymmetric braking loss** — braking events are rare but safety-critical; the 3× multiplier counteracts class imbalance
- **Always-applied lane mask** — masking continues post-curriculum with `lane_mask_prob` to prevent the model from unlearning its lane-agnostic representation
- **Yellow + white lane detection** — yellow centre lines are as important as white edge markings for lane keeping; both are now masked

## Dependencies

Install with:
```bash
pip install -r requirements.txt
```

For Google Colab:
```bash
pip install -r requirements_colab.txt
```

Core dependencies: `torch`, `torchvision`, `metadrive-simulator`, `Depth-Anything-V2` (submodule in `Depth-Anything-V2/`), `PyDrive2`, `numpy`.
