"""
PPO (Proximal Policy Optimization) Training Loop
=================================================
Rollout buffer + mini-batch update.

The buffer stores image observations and ego-state vectors separately so
that the policy's forward(image, ego) signature is preserved throughout.

Optimisations vs. the original:
  - Buffer arrays are pinned when CUDA is available → async H2D transfers.
  - torch.from_numpy + non_blocking=True in get_batches.
  - Optional AMP (FP16 forward, FP32 grad scaler) via PPOConfig.use_amp.
"""
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class PPOConfig:
    rollout_steps: int     = 2048
    epochs_per_update: int = 10
    mini_batch_size: int   = 64
    gamma: float           = 0.99
    gae_lambda: float      = 0.95
    clip_epsilon: float    = 0.2
    entropy_coef: float    = 0.01
    value_coef: float      = 0.5
    max_grad_norm: float   = 0.5
    lr: float              = 3e-4
    total_timesteps: int   = 200_000
    use_amp: bool          = False   # mixed-precision PPO update (CUDA only)


class RolloutBuffer:
    """
    Stores one rollout (rollout_steps transitions).

    Image observations and ego vectors are kept in separate arrays so the
    policy receives them as distinct tensors, matching the IL model's API.
    """

    def __init__(
        self,
        size: int,
        img_shape: tuple,
        ego_dim: int,
        act_dim: int,
        device: torch.device,
    ):
        self.size   = size
        self.device = device
        # Pin memory when training on CUDA: enables non-blocking async H2D transfers.
        use_pin = (device.type == "cuda")

        def _buf(*shape):
            t = torch.zeros(*shape, dtype=torch.float32)
            return (t.pin_memory() if use_pin else t).numpy()

        self.imgs       = _buf(size, *img_shape)
        self.egos       = _buf(size, ego_dim)
        self.actions    = _buf(size, act_dim)
        self.rewards    = _buf(size)
        self.dones      = _buf(size)
        self.log_probs  = _buf(size)
        self.values     = _buf(size)
        self.advantages = _buf(size)
        self.returns    = _buf(size)
        self.ptr = 0

    def store(self, img, ego, action, reward, done, log_prob, value):
        i = self.ptr
        self.imgs[i]      = img
        self.egos[i]      = ego
        self.actions[i]   = action
        self.rewards[i]   = reward
        self.dones[i]     = done
        self.log_probs[i] = log_prob
        self.values[i]    = value
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float, lam: float):
        """Generalized Advantage Estimation (backward pass)."""
        gae = 0.0
        for t in reversed(range(self.size)):
            next_value        = last_value if t == self.size - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae   = delta + gamma * lam * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int):
        """Yield shuffled mini-batches as dicts of device tensors.

        torch.from_numpy avoids a CPU copy; non_blocking=True lets DMA
        overlap with GPU compute when the buffer arrays are pinned.
        """
        dev = self.device
        nb  = dev.type == "cuda"
        indices = np.random.permutation(self.size)
        for start in range(0, self.size, batch_size):
            end = start + batch_size
            if end > self.size:
                break
            idx = indices[start:end]
            yield {
                "imgs":       torch.from_numpy(self.imgs[idx].copy()).to(dev, non_blocking=nb),
                "egos":       torch.from_numpy(self.egos[idx].copy()).to(dev, non_blocking=nb),
                "actions":    torch.from_numpy(self.actions[idx].copy()).to(dev, non_blocking=nb),
                "log_probs":  torch.from_numpy(self.log_probs[idx].copy()).to(dev, non_blocking=nb),
                "advantages": torch.from_numpy(self.advantages[idx].copy()).to(dev, non_blocking=nb),
                "returns":    torch.from_numpy(self.returns[idx].copy()).to(dev, non_blocking=nb),
            }

    def reset(self):
        self.ptr = 0


def ppo_update(policy, optimizer, buffer: RolloutBuffer, cfg: PPOConfig,
               scaler=None):
    """
    Run one PPO update over the current rollout buffer.

    Parameters
    ----------
    scaler : torch.cuda.amp.GradScaler or None
        When cfg.use_amp is True, pass a persistent GradScaler created once
        in the training script.  None → standard FP32 update.
    """
    use_amp = cfg.use_amp and buffer.device.type == "cuda"
    total_policy_loss = 0.0
    total_value_loss  = 0.0
    total_entropy     = 0.0
    num_updates = 0

    for _ in range(cfg.epochs_per_update):
        for batch in buffer.get_batches(cfg.mini_batch_size):
            imgs        = batch["imgs"]
            egos        = batch["egos"]
            old_actions = batch["actions"]
            old_log_p   = batch["log_probs"]
            advantages  = batch["advantages"]
            returns     = batch["returns"]

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            with torch.autocast("cuda", enabled=use_amp):
                _, new_log_p, entropy, new_values = policy.get_action_and_value(
                    imgs, egos, old_actions
                )

                ratio = (new_log_p - old_log_p).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(new_values, returns)

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()

            optimizer.zero_grad()
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()
            total_entropy     += entropy.mean().item()
            num_updates += 1

    n = max(num_updates, 1)
    return {
        "policy_loss": total_policy_loss / n,
        "value_loss":  total_value_loss  / n,
        "entropy":     total_entropy     / n,
    }
