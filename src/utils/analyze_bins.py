"""
Toy with WeightedRandomSampler bin boundaries and targets.

Usage:
    python src/utils/analyze_bins.py
    python src/utils/analyze_bins.py --data_dir data/processed/dpt_pred_mixed
    python src/utils/analyze_bins.py --edges 0.05 0.15 0.3 --targets 0.3 0.4 0.2 0.1
"""

import argparse
import glob
import os
import numpy as np


DEFAULT_DATA_DIR = "data/processed/dpt_pred_mixed"
DEFAULT_EDGES    = [0.05, 0.2]           # |steer| thresholds (current trainer values)
DEFAULT_TARGETS  = [0.48, 0.45, 0.07]   # desired proportion per bin (must sum to 1)


def load_actions(data_dir: str, split: str = "train") -> np.ndarray:
    files = glob.glob(os.path.join(data_dir, split, "*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}/{split}")
    parts = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        if "action" in d:
            parts.append(d["action"])
    return np.concatenate(parts)


def analyze(actions: np.ndarray, edges: list, targets: list) -> None:
    n_bins = len(edges) + 1
    if len(targets) != n_bins:
        raise ValueError(f"{n_bins} bins but {len(targets)} targets -- need exactly {n_bins} targets")
    if abs(sum(targets) - 1.0) > 1e-4:
        raise ValueError(f"targets sum to {sum(targets):.4f}, must sum to 1.0")

    steers    = actions[:, 0]
    accels    = actions[:, 1]
    steer_mag = np.abs(steers)
    total     = len(steers)

    bins = np.digitize(steer_mag, edges)

    SEP = "-" * 62

    # --- Raw distribution ---------------------------------------------------
    print(f"\n{SEP}")
    print(f"  RAW DISTRIBUTION  (N={total:,})")
    print(SEP)

    edge_strs = ["0"] + [str(e) for e in edges] + ["inf"]
    for b in range(n_bins):
        count = (bins == b).sum()
        lo, hi = edge_strs[b], edge_strs[b + 1]
        bar = "#" * int(40 * count / total)
        print(f"  Bin {b}  [{lo:>5}, {hi:<5})  {count:7,}  ({100*count/total:5.1f}%)  {bar}")

    # --- Effective distribution after reweighting ---------------------------
    weights = np.zeros(total, dtype=np.float64)
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        weights[mask] = targets[b] / mask.sum()

    w_sum = weights.sum()
    expected_fracs = np.array([
        weights[bins == b].sum() / w_sum for b in range(n_bins)
    ])

    print(f"\n{SEP}")
    print(f"  AFTER REWEIGHTING  targets={[f'{t:.2f}' for t in targets]}")
    print(SEP)
    for b in range(n_bins):
        lo, hi  = edge_strs[b], edge_strs[b + 1]
        frac    = expected_fracs[b]
        target  = targets[b]
        delta   = frac - target
        arrow   = "^" if delta > 0.005 else ("v" if delta < -0.005 else "=")
        eff_n   = int(frac * total)
        bar     = "#" * int(40 * frac)
        print(f"  Bin {b}  [{lo:>5}, {hi:<5})  eff~{eff_n:6,}  got={100*frac:5.1f}%  want={100*target:.1f}%  {arrow}  {bar}")

    # --- Accel breakdown ----------------------------------------------------
    braking      = (accels < -0.1).sum()
    coasting     = (np.abs(accels) < 0.05).sum()
    accelerating = total - braking - coasting

    print(f"\n{SEP}")
    print("  ACCEL BREAKDOWN")
    print(SEP)
    print(f"  accelerating (>+0.05):  {accelerating:7,}  ({100*accelerating/total:5.1f}%)")
    print(f"  coasting (|a|<0.05):    {coasting:7,}  ({100*coasting/total:5.1f}%)")
    print(f"  braking (<-0.1):        {braking:7,}  ({100*braking/total:5.1f}%)  [3x loss weight]")

    # --- Steer percentiles --------------------------------------------------
    pcts = [1, 5, 25, 50, 75, 95, 99]
    vals = np.percentile(steers, pcts)
    print(f"\n{SEP}")
    print("  STEER PERCENTILES")
    print(SEP)
    print("  " + "  ".join(f"p{p}={v:+.3f}" for p, v in zip(pcts, vals)))
    print(f"  mean={steers.mean():+.4f}  std={steers.std():.4f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--split",    default="train")
    parser.add_argument("--edges",   nargs="+", type=float, default=DEFAULT_EDGES,
                        help="|steer| bin edges (ascending). N edges -> N+1 bins.")
    parser.add_argument("--targets", nargs="+", type=float, default=DEFAULT_TARGETS,
                        help="Target proportion per bin (must sum to 1, one per bin).")
    args = parser.parse_args()

    actions = load_actions(args.data_dir, args.split)
    analyze(actions, args.edges, args.targets)


if __name__ == "__main__":
    main()
