"""
PPO (Proximal Policy Optimization) Training Loop
=================================================
Rollout buffer + mini-batch update.

The buffer stores image observations and ego-state vectors separately so
that the policy's forward(image, ego) signature is preserved throughout.

Multi-env support
-----------------
RolloutBuffer accepts n_envs ≥ 1.  All arrays are shaped (T, N, ...) where
T = rollout_steps and N = n_envs.  get_batches() flattens to (T*N, ...) and
shuffles before yielding mini-batches, so ppo_update() is unchanged.

With n_envs=1 the shapes collapse to (T, 1, ...) → (T,) after flattening,
which is behaviourally identical to the old single-env implementation.

Optimisations
-------------
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
    epochs_per_update: int = 4
    mini_batch_size: int   = 64
    gamma: float           = 0.99
    gae_lambda: float      = 0.95
    clip_epsilon: float    = 0.2
    entropy_coef: float    = 0.001
    value_coef: float      = 0.25
    max_grad_norm: float   = 0.5
    lr: float              = 3e-4
    total_timesteps: int   = 200_000
    use_amp: bool          = False   # mixed-precision PPO update (CUDA only)
    # Early stopping: halt the epoch loop if per-epoch avg KL exceeds this.
    # Checked once per epoch (not per mini-batch) to avoid noisy early cuts.
    # 0.15 allows more gradient steps per rollout; tighten to 0.05 once rewards stabilise.
    # Set to 0 to disable early stopping entirely.
    target_kl: float       = 0.15


class RolloutBuffer:
    """
    Stores one rollout (rollout_steps × n_envs transitions).

    Image observations and ego vectors are kept in separate arrays so the
    policy receives them as distinct tensors, matching the IL model's API.

    Shape convention
    ----------------
    All arrays: (rollout_steps, n_envs, ...)
    get_batches() flattens to (rollout_steps * n_envs, ...) before shuffling.
    """

    def __init__(
        self,
        size: int,
        img_shape: tuple,
        ego_dim: int,
        act_dim: int,
        device: torch.device,
        n_envs: int = 1,
    ):
        self.size   = size
        self.n_envs = n_envs
        self.device = device
        self._total = size * n_envs   # flattened size used by get_batches

        # Pin small scalar arrays for async H2D transfers.
        # imgs is NOT pinned: (T, N, 4, H, W) float32 can be hundreds of MB of
        # non-swappable page-locked RAM, which outweighs the async-transfer benefit
        # for 64-sample mini-batch slices (~7 MB each).
        use_pin = (device.type == "cuda")

        def _buf(*shape, pin=False):
            t = torch.zeros(*shape, dtype=torch.float32)
            return (t.pin_memory() if (use_pin and pin) else t).numpy()

        # Shape: (T, N, ...) — N=1 is backward-compatible after flattening
        self.imgs       = _buf(size, n_envs, *img_shape)           # not pinned (large)
        self.egos       = _buf(size, n_envs, ego_dim)
        self.actions    = _buf(size, n_envs, act_dim)
        self.rewards    = _buf(size, n_envs, pin=True)
        self.dones      = _buf(size, n_envs, pin=True)
        self.log_probs  = _buf(size, n_envs, pin=True)
        self.values     = _buf(size, n_envs, pin=True)
        self.advantages = _buf(size, n_envs)
        self.returns    = _buf(size, n_envs)
        self.ptr = 0

    def store(self, imgs, egos, actions, rewards, dones, log_probs, values):
        """
        Store one time-step of transitions for all N envs.

        All inputs should be (N, ...) shaped numpy arrays (or scalars/1-D
        arrays when N=1 from DummyVecEnv).
        """
        i = self.ptr
        self.imgs[i]      = imgs
        self.egos[i]      = egos
        self.actions[i]   = actions
        self.rewards[i]   = rewards
        self.dones[i]     = dones
        self.log_probs[i] = log_probs
        self.values[i]    = values
        self.ptr += 1

    def compute_gae(self, last_values: np.ndarray, gamma: float, lam: float):
        """
        Generalized Advantage Estimation — vectorized over N envs.

        Parameters
        ----------
        last_values : (N,) float32 — critic value at the observation *after*
                      the last rollout step (bootstrapped from main process).
        gamma, lam  : GAE hyperparameters.

        Done-flag convention
        --------------------
        dones[t] == 1 means the episode ended at step t (the action taken from
        obs[t] led to a terminal state).  The same scalar done that is stored
        in the buffer is used to zero-out the bootstrap:

            next_non_terminal = 1 - dones[t]

        This also zero-resets the GAE carry so no advantage bleeds across
        episode boundaries within the same env stream.

        For the final step (t = size-1) the bootstrap value is last_values[n]
        masked by 1 - dones[size-1, n] (already stored in the buffer).
        """
        gae = np.zeros(self.n_envs, dtype=np.float32)   # (N,)

        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_values = last_values                    # (N,)
            else:
                next_values = self.values[t + 1]             # (N,)

            next_non_terminal = 1.0 - self.dones[t]         # (N,) — done at step t

            delta = (
                self.rewards[t]
                + gamma * next_values * next_non_terminal
                - self.values[t]
            )                                                 # (N,)
            gae = delta + gamma * lam * next_non_terminal * gae   # (N,)
            self.advantages[t] = gae                          # (N,)

        self.returns = self.advantages + self.values           # (T, N)

    def get_batches(self, batch_size: int):
        """
        Flatten (T, N, ...) → (T*N, ...), shuffle, yield mini-batches.

        torch.from_numpy avoids a CPU copy; non_blocking=True lets DMA
        overlap with GPU compute when the buffer arrays are pinned.
        """
        dev   = self.device
        nb    = dev.type == "cuda"
        total = self._total

        # Flatten time × env axes into a single sample axis
        imgs_f       = self.imgs.reshape(total, *self.imgs.shape[2:])
        egos_f       = self.egos.reshape(total, self.egos.shape[2])
        actions_f    = self.actions.reshape(total, self.actions.shape[2])
        log_probs_f  = self.log_probs.reshape(total)
        advantages_f = self.advantages.reshape(total)
        returns_f    = self.returns.reshape(total)

        indices = np.random.permutation(total)
        for start in range(0, total, batch_size):
            end = start + batch_size
            if end > total:
                break
            idx = indices[start:end]
            yield {
                "imgs":       torch.from_numpy(imgs_f[idx].copy()).to(dev, non_blocking=nb),
                "egos":       torch.from_numpy(egos_f[idx].copy()).to(dev, non_blocking=nb),
                "actions":    torch.from_numpy(actions_f[idx].copy()).to(dev, non_blocking=nb),
                "log_probs":  torch.from_numpy(log_probs_f[idx].copy()).to(dev, non_blocking=nb),
                "advantages": torch.from_numpy(advantages_f[idx].copy()).to(dev, non_blocking=nb),
                "returns":    torch.from_numpy(returns_f[idx].copy()).to(dev, non_blocking=nb),
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
    total_approx_kl   = 0.0
    num_updates = 0

    for epoch in range(cfg.epochs_per_update):
        epoch_kl = 0.0
        epoch_batches = 0

        for batch in buffer.get_batches(cfg.mini_batch_size):
            imgs        = batch["imgs"]
            egos        = batch["egos"]
            old_actions = batch["actions"]
            old_log_p   = batch["log_probs"]
            advantages  = batch["advantages"]
            returns     = batch["returns"]

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = advantages.clamp(-5.0, 5.0)

            with torch.autocast("cuda", enabled=use_amp):
                _, new_log_p, entropy, new_values = policy.get_action_and_value(
                    imgs, egos, old_actions
                )

                log_ratio = new_log_p - old_log_p
                ratio = log_ratio.exp()
                # Schulman's approximation: KL ≈ (r-1) - log(r)
                approx_kl = ((ratio - 1) - log_ratio).mean()

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.huber_loss(new_values, returns, delta=10.0)

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
            total_approx_kl   += approx_kl.item()
            num_updates      += 1
            epoch_kl         += approx_kl.item()
            epoch_batches    += 1

        # Check per-epoch avg KL — more stable than per-mini-batch instantaneous KL.
        if cfg.target_kl > 0 and epoch_batches > 0:
            epoch_avg_kl = epoch_kl / epoch_batches
            if epoch_avg_kl > cfg.target_kl:
                print(f"[PPO] Early stopping after epoch {epoch + 1}/{cfg.epochs_per_update} "
                      f"— epoch avg KL {epoch_avg_kl:.4f} > threshold {cfg.target_kl:.4f}")
                break

    n = max(num_updates, 1)
    return {
        "policy_loss": total_policy_loss / n,
        "value_loss":  total_value_loss  / n,
        "entropy":     total_entropy     / n,
        "approx_kl":   total_approx_kl   / n,
    }
