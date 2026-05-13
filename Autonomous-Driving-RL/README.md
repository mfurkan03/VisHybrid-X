# Autonomous Driving RL — PPO Fine-tuning

PPO-based reinforcement learning that continues training from an IL (Imitation Learning) checkpoint. Run from this directory (`Autonomous-Driving-RL/`).

## Quick Start

```bash
# Fine-tune from an IL checkpoint (recommended)
python train_rl.py --il_checkpoint ../models/policy_model_best.pth --arch impala --image_size 84

# Train from scratch (no IL init)
python train_rl.py --arch impala --image_size 84

# Resume a previous RL run
python train_rl.py --rl_checkpoint checkpoints/policy_latest.pth --arch impala --image_size 84

# Watch it drive while training
python train_rl.py --il_checkpoint ../models/policy_model_best.pth --arch impala --render

# Test the RL policy
python test_rl.py --checkpoint checkpoints/policy_best.pth --arch impala
```

**`--arch` and `--image_size` must match what was used in IL training.**

## How IL→RL Works

The `ILActorCritic` wrapper loads any IL policy backbone (ImpalaNet, ImpalaNetV2, or DrivingPolicyNet) and adds a PPO critic on top:

```
IL backbone (frozen or fine-tuned)
  ├── visual CNN  →  512-d feature
  └── ego MLP     →   32-d feature
        ↓ concat (544-d merged feature)
        ├── steer_head  →  steering mean   ┐ Actor (from IL)
        ├── accel_head  →  throttle mean   ┘ + learnable log_std
        └── value_head  →  V(s)              Critic (new, random init)
```

The actor (steer + accel heads) carries over IL-learned weights. The critic (`value_head`) starts from random init — this is correct and expected.

Observation format matches the IL pipeline exactly:
- Channel 0: inverted depth — `1 - (raw - min) / (max - min)`, so closer = higher
- Channels 1-3: RGB [0, 1]
- Ego state (separate): `[total_speed, last_steer, heading_delta]`

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--il_checkpoint` | `None` | IL checkpoint to fine-tune from |
| `--rl_checkpoint` | `None` | RL checkpoint to resume |
| `--arch` | `impala` | `simple`, `impala`, or `impala_v2` |
| `--image_size` | `84` | Input resolution (match IL) |
| `--timesteps` | `200000` | Total environment steps |
| `--rollout` | `2048` | Steps collected before each PPO update |
| `--batch` | `64` | Mini-batch size |
| `--epochs` | `10` | PPO epochs per update |
| `--lr` | `3e-4` | Learning rate |
| `--scenarios` | `50` | Map variety |
| `--dpt_path` | `None` | Fine-tuned DPT weights (optional) |
| `--render` | off | Open MetaDrive 3D window |
| `--save_dir` | `checkpoints` | Checkpoint directory |

## Checkpoints

Saved to `checkpoints/` during training:
- `policy_latest.pth` — every PPO update
- `policy_best.pth` — when average route completion (last 20 episodes) improves

Checkpoint format:
```python
{
    "policy":        <ILActorCritic state dict>,
    "il_model_arch": "impala",
    "image_size":    84,
    "global_step":   ...,
    "route":         <best avg route completion>,
}
```

To use an RL checkpoint in the IL simulation test:
```bash
python ../src/test_simulation_policy.py \
    --model_path checkpoints/policy_best.pth \
    --arch impala --image_size 84
```
The script auto-detects RL checkpoints and strips the actor-critic wrapper.

## Reward System

Configured in `rl/rewards.py` via `RewardConfig`:

| Component | Default | Description |
|---|---|---|
| Route progress | `×100` per delta | Primary training signal |
| Arrive destination | `+50` | Strong sparse reward |
| Out of road | `-10` | Terminal penalty |
| Crash vehicle | `-20` | Terminal penalty |
| Crash object | `-10` | Terminal penalty |
| Harsh steering | `-0.5 × \|steer\|` | Discourages oscillation |
| Steering jerk | `-0.2 × Δsteer` | Discourages sudden changes |
| Overspeed | `-0.5 × excess/limit` | Above 40 km/h |
| Speed bonus | `+0.1 × speed/limit` | Moving above 5 km/h |
| Standing still | `-0.05` | Per step below 5 km/h |

To add a reward term, add a field to `RewardConfig` and a block to `compute_reward()` in `rl/rewards.py` — no other files need changing.

## Training Time Estimates

| Timesteps | GPU estimate | Notes |
|---|---|---|
| 50K | ~15–30 min | Verify the system works |
| 200K | ~1–2 h | Basic behaviours emerge |
| 1M+ | ~5–10 h | Stable, generalising policy |

DPT inference runs every step, so GPU is strongly recommended.

## Reading the Console Output

```
EP   42 | R:  +3.25 | Len:  187 | Route: 34.2% | Spd: 28.3 | route_progress: +3.42 | harsh_steering: -0.17
EP   43 | R:  -4.12 | Len:   23 | Route:  5.1% | Spd: 45.2 | out_of_road: -10.00

Update  12 | Step   24,576/200,000 | FPS: 850 | Avg R(20): +1.34 | Avg Route: 18.3% | P_loss: 0.023 | V_loss: 0.145 | Entropy: 1.23
```

- **R** — episode total reward (higher is better)
- **Len** — episode length in steps (longer = surviving longer)
- **Route** — route completion % (primary metric)
- **Avg R(20)** — rolling mean of last 20 episodes
- **Entropy** — exploration level; if it collapses to ~0 the policy has stopped exploring

## File Structure

```
Autonomous-Driving-RL/
├── train_rl.py           # Training entry point
├── test_rl.py            # Deterministic test rollout
└── rl/
    ├── il_actor_critic.py  # ILActorCritic wrapper (IL backbone + value head)
    ├── env_wrapper.py      # MetaDrive env with IL-correct depth + ego state
    ├── ppo.py              # PPO algorithm, GAE, rollout buffer
    └── rewards.py          # Modular reward/penalty system
```
