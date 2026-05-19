"""Apply a fitted Glauber-MC + NBD to SMASH events to produce centrality predictions.

For each energy, given the fit_glauber.py output:
  1. Load the Glauber event set (b, N_part, N_coll, sampled mult).
  2. Sort Glauber events by multiplicity to define centrality percentile bins.
  3. For each SMASH event, look up the centrality bin that its multiplicity
     falls into (same percentile-thresholds the Glauber set would assign).
  4. The predicted b is the mean truth-b of *Glauber* events in that bin.
     The Glauber-internal b is used because that is what STAR analyses report:
     the Glauber model's geometric impact parameter associated with each
     multiplicity bin, not the SMASH transport's b. This is the standard
     STAR-style Glauber baseline.
  5. We also report a predicted-b STD per bin (resolution proxy) and the
     fraction of SMASH events at the multiplicity floor where Glauber has no
     resolving power.

Output schema mirrors run_truth.py so the evaluator can treat both baselines
uniformly.

Usage:
    python scripts/run_glauber.py \\
        --smash-inputs data/processed/auau_{3p2,3p5,3p9,4p5}GeV.h5 \\
        --glauber-dir  data/processed/glauber \\
        --output-dir   data/processed/glauber
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines.truth import DEFAULT_BIN_EDGES, DEFAULT_BIN_LABELS  # noqa: E402


def process_energy(mult_smash: np.ndarray, b_true: np.ndarray, sqrtsNN: float,
                   n_events: int, glauber_events_path: Path, out_path: Path) -> dict:

    with h5py.File(glauber_events_path, "r") as h:
        b_glauber = h["b"][:]
        mult_glauber = h["mult"][:]

    # Define centrality bin thresholds in Glauber multiplicity space using the same
    # percentile edges as the truth-tuned baseline. Higher multiplicity → more central → smaller bin index.
    edges = DEFAULT_BIN_EDGES
    labels = list(DEFAULT_BIN_LABELS)
    lower_quantiles = 1.0 - edges[1:]
    mult_thresholds = np.quantile(mult_glauber, lower_quantiles)

    # Map each Glauber event to a centrality bin → compute the per-bin mean and std of
    # the Glauber model b. These are the "predicted b" and its in-bin spread for that
    # centrality class. By construction this is what the Glauber baseline outputs.
    bins_glauber = _assign_bins(mult_glauber, mult_thresholds)
    n_bins = len(labels)
    b_means = np.full(n_bins, np.nan, dtype=np.float64)
    b_stds = np.full(n_bins, np.nan, dtype=np.float64)
    counts_g = np.zeros(n_bins, dtype=np.int64)
    for k in range(n_bins):
        sel = bins_glauber == k
        counts_g[k] = int(sel.sum())
        if sel.sum() > 0:
            b_means[k] = float(b_glauber[sel].mean())
            b_stds[k] = float(b_glauber[sel].std())

    # Apply to SMASH events.
    bins_smash = _assign_bins(mult_smash, mult_thresholds)
    b_pred = np.where(np.isnan(b_means[bins_smash]),
                      np.nan, b_means[bins_smash]).astype(np.float32)

    # Quality diagnostics on SMASH side: residual stats per bin.
    counts_s = np.zeros(n_bins, dtype=np.int64)
    res_std = np.full(n_bins, np.nan, dtype=np.float64)
    res_mae = np.full(n_bins, np.nan, dtype=np.float64)
    for k in range(n_bins):
        sel = bins_smash == k
        counts_s[k] = int(sel.sum())
        if sel.sum() > 0 and np.isfinite(b_means[k]):
            res = b_pred[sel] - b_true[sel]
            res_std[k] = float(res.std())
            res_mae[k] = float(np.abs(res).mean())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h:
        h.create_dataset("centrality_bin", data=bins_smash)
        h.create_dataset("b_pred", data=b_pred)
        h.create_dataset("mult_thresholds", data=mult_thresholds)
        h.create_dataset("b_means", data=b_means)
        h.create_dataset("b_stds", data=b_stds)
        h.attrs["sqrtsNN"] = sqrtsNN
        h.attrs["n_events"] = n_events
        h.attrs["bin_edges"] = list(edges)
        h.attrs["bin_labels"] = labels
        h.attrs["source_cache"] = str(out_path)
        h.attrs["source_glauber_events"] = str(glauber_events_path)

    return {
        "sqrtsNN": sqrtsNN,
        "bin_labels": labels,
        "mult_thresholds": mult_thresholds.tolist(),
        "b_means": b_means.tolist(),
        "b_residual_mae_per_bin": res_mae.tolist(),
        "smash_counts_per_bin": counts_s.tolist(),
        "glauber_counts_per_bin": counts_g.tolist(),
    }


def _assign_bins(mult: np.ndarray, mult_thresholds: np.ndarray) -> np.ndarray:
    """Vectorized assignment — matches the truth-baseline convention so the two baselines
    can be compared bin-by-bin downstream."""
    asc = -mult_thresholds
    idx = np.searchsorted(asc, -mult, side="right") - 1
    return np.clip(idx, 0, len(mult_thresholds) - 1).astype(np.int16)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, type=Path,
                   help="Lab-frame cache (provides mult_lab and truth b for each event)")
    p.add_argument("--glauber-dir", required=True, type=Path,
                   help="Directory holding glauber_events_<tag>GeV.h5 files from fit_glauber.py")
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.cache, "r") as h:
        mult = h["mult_lab"][:]
        b_all = h["b"][:]
        sqrts_per_energy = list(h.attrs["sqrtsNN_per_energy"])
        n_per_energy = list(h.attrs["n_events_per_energy"])

    cum = np.cumsum([0] + n_per_energy)
    summary = {}
    for e_id, (sqrtsNN, n_e) in enumerate(zip(sqrts_per_energy, n_per_energy)):
        lo, hi = int(cum[e_id]), int(cum[e_id + 1])
        tag = f"{sqrtsNN:.1f}".replace(".", "p")
        gev_path = args.glauber_dir / f"glauber_events_{tag}GeV.h5"
        out_path = args.output_dir / f"glauber_pred_auau_{tag}GeV.h5"
        s = process_energy(mult[lo:hi], b_all[lo:hi], sqrtsNN, int(n_e), gev_path, out_path)
        summary[f"auau_{tag}GeV"] = s
        print(f"auau_{tag}GeV: √sNN={sqrtsNN} GeV, "
              f"MAE per bin = {[f'{v:.2f}' if np.isfinite(v) else 'nan' for v in s['b_residual_mae_per_bin']]} fm")

    with open(args.output_dir / "predictions_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.output_dir / 'predictions_summary.json'}")


if __name__ == "__main__":
    main()
