"""
Vectorized MetaDrive environments for parallel rollout collection.
==================================================================
SubprocVecEnv spawns N worker processes — one MetaDrive env each, no DPT.
The main process handles batched DPT depth inference + policy forward.

Async reset design
------------------
Workers never stall the training loop during episode resets.

  1. step(active_indices, actions)
       Steps only the envs in active_indices.  When a result is done=True,
       a RESET is dispatched to that worker immediately and it is moved from
       _active → _resetting.  step() returns without waiting for the reset.

  2. poll_resets()
       Non-blocking check: returns {idx: (rgb, ego)} for any workers whose
       reset has already finished.  Call every loop iteration.

  3. wait_any_reset()
       Blocking: waits until at least one pending reset completes.  Use this
       when active_indices is empty so the loop always has work to do.

  4. The training loop fills the rollout buffer per-env via independent
       pointers (env_ptrs[i]).  Envs that reset slowly simply finish later;
       fast envs keep collecting without waiting.

Net effect: MetaDrive's expensive episode resets (2–5 s on Windows) never
stall the training loop.  Other envs keep stepping during any env's reset.

Usage
-----
    make_env_fns = [
        MakeEnvFn(seed=42, worker_idx=i, scenarios=50, reward_cfg=cfg, image_size=84)
        for i in range(n_envs)
    ]
    vec_env = SubprocVecEnv(n_envs, make_env_fns)   # n_envs > 1
    # or:
    vec_env = DummyVecEnv(make_env_fns[0])           # n_envs == 1

    raw_rgbs, egos = vec_env.reset()    # (N,196,196,3), (N,EGO_DIM)

    # Async rollout skeleton:
    reset_obs = vec_env.poll_resets()           # {idx: (rgb, ego)}
    results   = vec_env.step(active, actions)   # {idx: (rgb,ego,rew,done,info)}
    if not vec_env.active_indices:
        reset_obs = vec_env.wait_any_reset()
    vec_env.close()
"""
import sys
import traceback
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rl.rewards import RewardConfig


# ── Worker process ────────────────────────────────────────────────────────────

def _worker_fn(conn: mp.connection.Connection, make_env_fn) -> None:
    """
    Entry point for each worker process.
    Responds to: RESET → (rgb, ego) | STEP, action → (rgb, ego, reward, done, info) | CLOSE
    """
    try:
        env = make_env_fn()
        while True:
            cmd = conn.recv()
            if cmd[0] == "RESET":
                rgb, ego = env.reset()
                conn.send((rgb, ego))
            elif cmd[0] == "STEP":
                obs, reward, done, info = env.step(cmd[1])
                rgb, ego = obs
                conn.send((rgb, ego, reward, done, info))
            elif cmd[0] == "CLOSE":
                env.close()
                conn.close()
                return
    except Exception:
        conn.send(("ERROR", traceback.format_exc()))
        conn.close()


# ── Picklable env factory ─────────────────────────────────────────────────────

@dataclass
class MakeEnvFn:
    """
    Picklable callable that creates one MetaDriveRLWrapper(use_depth_model=False).

    Lambdas are not picklable on Windows (spawn start method).
    Each worker gets a distinct scenario slice:
        start_seed = seed + worker_idx * 100
    """
    seed:       int
    worker_idx: int
    scenarios:  int
    reward_cfg: RewardConfig
    image_size: int
    dpt_path:   str   = None   # ignored — workers never use DPT
    render:     bool  = False  # only meaningful for worker 0 / DummyVecEnv
    camera_fov: float = 60

    def __call__(self):
        from rl.env_wrapper import MetaDriveRLWrapper
        cfg = {
            "start_seed":    self.seed + self.worker_idx * 100,
            "num_scenarios": self.scenarios,
        }
        return MetaDriveRLWrapper(
            reward_cfg=self.reward_cfg,
            env_config=cfg,
            use_depth_model=False,
            image_size=self.image_size,
            render=self.render,
            camera_fov=self.camera_fov,
        )


# ── Vectorized envs ───────────────────────────────────────────────────────────

class SubprocVecEnv:
    """
    N parallel MetaDrive environments in separate worker processes.

    Only active envs are stepped; done envs transition to _resetting and
    are picked up by poll_resets() / wait_any_reset() when ready.
    """

    def __init__(self, n_envs: int, make_env_fns):
        if callable(make_env_fns) and not isinstance(make_env_fns, list):
            make_env_fns = [make_env_fns] * n_envs
        assert len(make_env_fns) == n_envs, \
            f"Expected {n_envs} factories, got {len(make_env_fns)}"

        self.n_envs            = n_envs
        self._workers: list[mp.Process]               = []
        self._conns:   list[mp.connection.Connection] = []
        self._conn_to_idx: dict                       = {}
        self._active:    set[int] = set()
        self._resetting: set[int] = set()

        ctx = mp.get_context("spawn")
        for i in range(n_envs):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            p = ctx.Process(
                target=_worker_fn,
                args=(child_conn, make_env_fns[i]),
                daemon=True,
            )
            p.start()
            child_conn.close()
            self._workers.append(p)
            self._conns.append(parent_conn)
            self._conn_to_idx[parent_conn] = i

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Reset ALL envs. Returns (raw_rgbs, egos) for all N workers."""
        for conn in self._conns:
            conn.send(("RESET",))
        results = self._collect(set(range(self.n_envs)))
        self._active    = set(range(self.n_envs))
        self._resetting = set()
        raw_rgbs = np.stack([results[i][0] for i in range(self.n_envs)])
        egos     = np.stack([results[i][1] for i in range(self.n_envs)])
        return raw_rgbs, egos

    def step(self, active_indices: list[int], actions: np.ndarray) -> dict[int, tuple]:
        """
        Step only the envs in active_indices (must be a subset of _active).

        Done envs have RESET dispatched immediately and move to _resetting.
        Returns without waiting for any reset to complete.

        Parameters
        ----------
        active_indices : which env indices to step
        actions        : (len(active_indices), 2) float32

        Returns
        -------
        dict mapping env_idx → (rgb, ego, reward, done, info)
        rgb/ego are TERMINAL obs for done envs — call poll_resets() /
        wait_any_reset() to get the first-frame obs of the new episode.
        """
        for i, action in zip(active_indices, actions):
            self._conns[i].send(("STEP", action))

        results: dict[int, tuple] = {}
        pending = set(active_indices)

        while pending:
            ready = mp.connection.wait(
                [self._conns[i] for i in pending], timeout=120.0
            )
            if not ready:
                self._check_alive(pending, "STEP")
            for conn in ready:
                i = self._conn_to_idx[conn]
                r = self._recv_one(conn, i)
                results[i] = r
                pending.discard(i)
                if r[3]:   # done — dispatch reset, move to resetting pool
                    conn.send(("RESET",))
                    self._active.discard(i)
                    self._resetting.add(i)

        return results

    def poll_resets(self) -> dict[int, tuple]:
        """
        Non-blocking check for completed episode resets.

        Returns {env_idx: (rgb, ego)} for any workers whose reset has already
        finished.  Those envs are moved back to _active.  Returns {} if no
        reset has completed yet.
        """
        if not self._resetting:
            return {}
        ready = mp.connection.wait(
            [self._conns[i] for i in self._resetting], timeout=0.0
        )
        return self._collect_ready(ready)

    def wait_any_reset(self) -> dict[int, tuple]:
        """
        Block until at least one pending reset completes.

        Use this when active_indices is empty (all envs are resetting) so the
        training loop always has at least one env available to step.
        Returns {env_idx: (rgb, ego)} — all resets that finished concurrently.
        """
        if not self._resetting:
            return {}
        ready = mp.connection.wait(
            [self._conns[i] for i in self._resetting], timeout=120.0
        )
        if not ready:
            self._check_alive(self._resetting, "RESET")
        return self._collect_ready(ready)

    @property
    def active_indices(self) -> list[int]:
        return sorted(self._active)

    @property
    def resetting_count(self) -> int:
        return len(self._resetting)

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(("CLOSE",))
            except Exception:
                pass
        for conn in self._conns:
            conn.close()
        for p in self._workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _collect_ready(self, ready_conns) -> dict[int, tuple]:
        results = {}
        for conn in ready_conns:
            i = self._conn_to_idx[conn]
            results[i] = self._recv_one(conn, i)
            self._resetting.discard(i)
            self._active.add(i)
        return results

    def _recv_one(self, conn, worker_idx: int):
        r = conn.recv()
        if isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], str) and r[0] == "ERROR":
            raise RuntimeError(f"Worker {worker_idx} crashed:\n{r[1]}")
        return r

    def _check_alive(self, pending: set, phase: str):
        dead = [i for i in pending if not self._workers[i].is_alive()]
        if dead:
            i = dead[0]
            raise RuntimeError(
                f"Worker {i} (PID={self._workers[i].pid}) died during {phase}. "
                f"Exit code: {self._workers[i].exitcode}."
            )
        raise TimeoutError(f"Worker(s) {pending} did not respond within 120s ({phase}).")

    def _collect(self, pending: set) -> dict:
        results = {}
        pending = set(pending)
        while pending:
            ready = mp.connection.wait(
                [self._conns[i] for i in pending], timeout=120.0
            )
            if not ready:
                self._check_alive(pending, "collect")
            for conn in ready:
                i = self._conn_to_idx[conn]
                results[i] = self._recv_one(conn, i)
                pending.discard(i)
        return results


class DummyVecEnv:
    """
    Single-env wrapper with the same async API as SubprocVecEnv.

    For n_envs=1 there is no subprocess parallelism, but the API is identical
    so the training loop is unchanged regardless of env count.

    When step() returns done=True it synchronously resets the env and caches
    the first-frame obs.  poll_resets() returns that cached obs immediately on
    the next call, so the training loop sees the same async pattern.
    """

    def __init__(self, make_env_fn):
        self.n_envs      = 1
        self._env        = make_env_fn()
        self._reset_cache: dict[int, tuple] = {}   # {0: (rgb, ego)} after done

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        rgb, ego = self._env.reset()
        self._reset_cache.clear()
        return rgb[np.newaxis], ego[np.newaxis]

    def step(self, active_indices: list[int], actions: np.ndarray) -> dict[int, tuple]:
        obs, reward, done, info = self._env.step(actions[0])
        rgb, ego = obs
        if done:
            reset_rgb, reset_ego = self._env.reset()
            self._reset_cache[0] = (reset_rgb, reset_ego)
        return {0: (rgb, ego, reward, done, info)}

    def poll_resets(self) -> dict[int, tuple]:
        result = dict(self._reset_cache)
        self._reset_cache.clear()
        return result

    def wait_any_reset(self) -> dict[int, tuple]:
        return self.poll_resets()

    @property
    def active_indices(self) -> list[int]:
        return [0]

    @property
    def resetting_count(self) -> int:
        return 0

    def close(self) -> None:
        self._env.close()
