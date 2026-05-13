"""
Observation Pipeline Verifier
==============================
Records per-channel statistics from rollout observations and compares them
against a saved reference so you can verify that the simulation preprocessing
exactly matches what the model saw during training.

Quick usage
-----------
Training — save a reference after the first rollout:

    verifier = ObsVerifier()
    for step in rollout:
        img, ego = obs
        verifier.record(img, ego)
    verifier.save("obs_ref.json")

Simulation / test — compare against the reference:

    verifier = ObsVerifier()
    for step in sim:
        img, ego = obs
        verifier.record(img, ego)
    ok = verifier.compare("obs_ref.json")   # prints a table; returns bool
"""

import json
import numpy as np

_CH_NAMES = ("depth", "R", "G", "B")


class ObsVerifier:
    """Collects per-channel statistics for obs-pipeline verification."""

    def __init__(self):
        self._imgs: list = []
        self._egos: list = []

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(self, img_np: np.ndarray, ego_np: np.ndarray) -> None:
        """Append one (4, H, W) image and (EGO_DIM,) ego vector."""
        self._imgs.append(img_np)
        self._egos.append(ego_np)

    def reset(self) -> None:
        self._imgs.clear()
        self._egos.clear()

    def __len__(self) -> int:
        return len(self._imgs)

    # ── Statistics ────────────────────────────────────────────────────────────

    def compute_stats(self) -> dict:
        """Return per-channel and ego statistics as a JSON-serialisable dict."""
        if not self._imgs:
            return {}
        imgs = np.stack(self._imgs)   # (N, 4, H, W)
        egos = np.stack(self._egos)   # (N, EGO_DIM)

        stats: dict = {"n_samples": len(self._imgs)}
        for ch, name in enumerate(_CH_NAMES):
            d = imgs[:, ch].ravel()
            stats[name] = {
                "min":  float(d.min()),
                "max":  float(d.max()),
                "mean": float(d.mean()),
                "std":  float(d.std()),
            }
        stats["ego"] = {
            "min":  float(egos.min()),
            "max":  float(egos.max()),
            "mean": egos.mean(axis=0).tolist(),
            "std":  egos.std(axis=0).tolist(),
        }
        return stats

    def print_stats(self) -> None:
        stats = self.compute_stats()
        n = stats.get("n_samples", 0)
        print(f"\n[ObsVerifier] Stats over {n} frames:")
        for ch in _CH_NAMES:
            s = stats[ch]
            print(f"  {ch:<6}: min={s['min']:6.3f}  max={s['max']:6.3f}  "
                  f"mean={s['mean']:7.4f}  std={s['std']:7.4f}")
        eg = stats["ego"]
        means = ", ".join(f"{v:.4f}" for v in eg["mean"])
        stds  = ", ".join(f"{v:.4f}" for v in eg["std"])
        print(f"  {'ego':<6}: mean=[{means}]  std=[{stds}]")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Write reference statistics to a JSON file."""
        stats = self.compute_stats()
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[ObsVerifier] Saved reference ({stats['n_samples']} frames) → {path}")

    # ── Comparison ────────────────────────────────────────────────────────────

    def compare(self, ref_path: str, tol: float = 0.10) -> bool:
        """
        Compare current statistics against a saved reference.

        Parameters
        ----------
        ref_path : path written by save()
        tol      : max relative deviation for mean/std before flagging a mismatch

        Returns True when all channels are within tolerance.
        """
        with open(ref_path) as f:
            ref = json.load(f)
        cur = self.compute_stats()

        print(
            f"\n[ObsVerifier] {cur['n_samples']} sim frames  vs  "
            f"{ref['n_samples']} training frames  (tol={tol*100:.0f}%)"
        )
        print(f"  {'Ch':<8} {'Stat':<6} {'Train':>9} {'Sim':>9} {'Δ%':>7}  Status")
        print(f"  {'-'*52}")

        all_ok = True
        for ch in _CH_NAMES:
            if ch not in ref or ch not in cur:
                continue
            for stat in ("mean", "std"):
                r = ref[ch][stat]
                c = cur[ch][stat]
                rel = abs(c - r) / (abs(r) + 1e-8)
                ok = rel <= tol
                if not ok:
                    all_ok = False
                mark = "OK" if ok else "!! MISMATCH"
                print(f"  {ch:<8} {stat:<6} {r:>9.4f} {c:>9.4f} {rel*100:>6.1f}%  {mark}")

        print()
        if all_ok:
            print(
                "[ObsVerifier] Pipeline consistent — obs preprocessing matches training.\n"
            )
        else:
            print(
                "[ObsVerifier] WARNING: Obs statistics differ from training reference!\n"
                "  Possible causes: depth inversion change, normalization bug, resize\n"
                "  mode mismatch, or RGB channel order swap. Check env_wrapper.py.\n"
            )
        return all_ok
