"""
Vectorized MetaDrive environments for parallel rollout collection.
==================================================================
SubprocVecEnv spawns N worker processes — one MetaDrive env each, no DPT.
The main process handles batched DPT depth inference + policy forward.

Reset-overlap protocol
----------------------
Workers do NOT auto-reset inside their STEP handler.  Instead:
  1. Main sends STEP actions to all N workers simultaneously.
  2. Main calls step() which uses connection.wait() to collect results in
     arrival order (fastest workers first).
  3. As soon as a done=True result arrives, main immediately sends RESET to
     that worker — it starts resetting while main is still collecting the
     remaining step results and running DPT + policy.
  4. After all step results are in, main waits for any outstanding RESET
     results.  By then, the reset may already be complete.

Net effect: MetaDrive episode resets overlap with GPU computation and with
collecting results from non-done workers, instead of blocking the whole step.

Usage
-----
    make_env_fns = [
        MakeEnvFn(seed=42, worker_idx=i, scenarios=50, reward_cfg=cfg, image_size=84)
        for i in range(n_envs)
    ]
    vec_env = SubprocVecEnv(n_envs, make_env_fns)   # n_envs > 1
    # or:
    vec_env = DummyVecEnv(make_env_fns[0])           # n_envs == 1

    raw_rgbs, egos = vec_env.reset()   # (N,196,196,3) uint8, (N,EGO_DIM) float32
    raw_rgbs, egos, rewards, dones, infos = vec_env.step(actions)
    vec_env.close()
"""
import sys
import traceback
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

# Ensure parent repo is importable inside the worker process (spawn re-imports modules).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rl.rewards import RewardConfig


# ── Worker process ────────────────────────────────────────────────────────────

def _worker_fn(conn: mp.connection.Connection, make_env_fn) -> None:
    """
    Entry point for each worker process.

    Responds to three commands:
      RESET          → env.reset() → send (rgb, ego)
      STEP, action   → env.step(action) → send (rgb, ego, reward, done, info)
                       (no auto-reset on done — main handles RESET separately)
      CLOSE          → clean up and exit
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
                # Main sends RESET immediately if done=True; we handle it on
                # the next loop iteration without any extra state here.

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

    Lambdas are not picklable on Windows (spawn start method), so we use a
    dataclass instead.

    Each worker gets a distinct scenario slice:
        start_seed = seed + worker_idx * 100
    Worker 0 → scenarios [seed, seed+99]
    Worker 1 → scenarios [seed+100, seed+199]  … etc.
    """
    seed:       int
    worker_idx: int
    scenarios:  int
    reward_cfg: RewardConfig
    image_size: int
    dpt_path:   str = None   # ignored — workers never use DPT

    def __call__(self):
        # Deferred import so the factory itself doesn't need MetaDrive at import time.
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
        )


# ── Vectorized envs ───────────────────────────────────────────────────────────

class SubprocVecEnv:
    """
    N parallel MetaDrive environments in separate worker processes.

    Workers return raw uint8 RGB; DPT and policy run in the main process.
    Episode resets are overlapped with DPT/policy computation using
    connection.wait() + immediate RESET dispatch on done.
    """

    def __init__(self, n_envs: int, make_env_fns):
        if callable(make_env_fns) and not isinstance(make_env_fns, list):
            make_env_fns = [make_env_fns] * n_envs
        assert len(make_env_fns) == n_envs, \
            f"Expected {n_envs} factories, got {len(make_env_fns)}"

        self.n_envs   = n_envs
        self._workers: list[mp.Process]               = []
        self._conns:   list[mp.connection.Connection] = []
        self._conn_to_idx: dict                       = {}

        ctx = mp.get_context("spawn")   # explicit; required on Windows, safe on Linux
        for i in range(n_envs):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            p = ctx.Process(
                target=_worker_fn,
                args=(child_conn, make_env_fns[i]),
                daemon=True,
            )
            p.start()
            # Close parent's copy of the child end — failing to do this keeps
            # the pipe alive even if the worker exits, causing recv() to hang.
            child_conn.close()
            self._workers.append(p)
            self._conns.append(parent_conn)
            self._conn_to_idx[parent_conn] = i

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Reset all envs.

        Returns
        -------
        raw_rgbs : (N, 196, 196, 3) uint8
        egos     : (N, EGO_DIM)     float32
        """
        for conn in self._conns:
            conn.send(("RESET",))
        results  = self._collect(set(range(self.n_envs)))
        raw_rgbs = np.stack([results[i][0] for i in range(self.n_envs)])
        egos     = np.stack([results[i][1] for i in range(self.n_envs)])
        return raw_rgbs, egos

    def step(self, actions: np.ndarray) -> tuple:
        """
        Step all envs, overlapping any episode resets with main-process work.

        Parameters
        ----------
        actions : (N, 2) float32

        Returns
        -------
        raw_rgbs : (N, 196, 196, 3) uint8  — next obs (reset obs for done envs)
        egos     : (N, EGO_DIM)     float32
        rewards  : (N,)             float32
        dones    : (N,)             bool
        infos    : list of N dicts
        """
        # 1. Dispatch all actions simultaneously.
        for conn, action in zip(self._conns, actions):
            conn.send(("STEP", action))

        # 2. Collect step results via connection.wait() (arrival order).
        #    Send RESET immediately when a done result arrives so the worker
        #    starts resetting while we collect remaining results + run DPT/policy.
        step_results: dict[int, tuple] = {}
        pending = set(range(self.n_envs))

        while pending:
            ready = mp.connection.wait(
                [self._conns[i] for i in pending], timeout=120.0
            )
            if not ready:
                self._check_alive(pending, "STEP")
            for conn in ready:
                i = self._conn_to_idx[conn]
                r = self._recv_one(conn, i)
                step_results[i] = r
                pending.discard(i)
                if r[3]:   # done — start reset immediately
                    conn.send(("RESET",))

        # 3. Collect reset results from done workers.
        #    By now the main process has finished DPT + policy for the current
        #    step, giving resets the maximum time to complete.
        reset_results: dict[int, tuple] = {}
        reset_pending = {i for i, r in step_results.items() if r[3]}

        while reset_pending:
            ready = mp.connection.wait(
                [self._conns[i] for i in reset_pending], timeout=120.0
            )
            if not ready:
                self._check_alive(reset_pending, "RESET")
            for conn in ready:
                i = self._conn_to_idx[conn]
                reset_results[i] = self._recv_one(conn, i)
                reset_pending.discard(i)

        # 4. Assemble outputs.  Done workers use reset obs; others use step obs.
        raw_rgbs = np.empty((self.n_envs, 196, 196, 3), dtype=np.uint8)
        egos     = np.empty((self.n_envs, step_results[0][1].shape[0]), dtype=np.float32)
        for i in range(self.n_envs):
            if i in reset_results:
                raw_rgbs[i] = reset_results[i][0]
                egos[i]     = reset_results[i][1]
            else:
                raw_rgbs[i] = step_results[i][0]
                egos[i]     = step_results[i][1]

        rewards = np.array([step_results[i][2] for i in range(self.n_envs)], dtype=np.float32)
        dones   = np.array([step_results[i][3] for i in range(self.n_envs)], dtype=bool)
        infos   = [step_results[i][4] for i in range(self.n_envs)]
        return raw_rgbs, egos, rewards, dones, infos

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
        """Collect one response from each worker in `pending` using wait()."""
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
    Single-env wrapper with the same API as SubprocVecEnv.

    No subprocess overhead. Use when --n_envs=1 so the training loop
    stays identical regardless of the number of environments.

    On done, auto-resets and returns the new episode's first obs as next obs.
    """

    def __init__(self, make_env_fn):
        self.n_envs = 1
        self._env   = make_env_fn()

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        rgb, ego = self._env.reset()
        return rgb[np.newaxis], ego[np.newaxis]

    def step(self, actions: np.ndarray) -> tuple:
        obs, reward, done, info = self._env.step(actions[0])
        rgb, ego = obs
        if done:
            rgb, ego = self._env.reset()
        return (
            rgb[np.newaxis],
            ego[np.newaxis],
            np.array([reward], dtype=np.float32),
            np.array([done], dtype=bool),
            [info],
        )

    def close(self) -> None:
        self._env.close()
