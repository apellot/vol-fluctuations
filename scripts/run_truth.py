"""Calibrate and apply the truth-tuned percentile baseline using mult_lab from the cache.

This is the *oracle classical baseline* — bin events by lab-frame charged
multiplicity in 0 < η_lab < 2, then assign each bin the SMASH-truth mean b.
It sets the upper bound on what any single-observable classical method can
achieve. The realistic STAR-style baseline (Glauber-tuned-to-RefMult) is in
scripts/run_glauber.py.

Reads from the padded cache (lab-frame, charged-only) built by
build_padded_cache.py. Writes one prediction HDF5 per energy slice so the
evaluator and ML training scripts can pull centrality_bin labels.

Usage:
    python scripts/run_truth.py \\
        --cache data/processed/cached/all_lab_truth.h5 \\
        --output-dir data/processed/truth_lab
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines.truth import (  # noqa: E402
    fit, predict, train_test_split_indices,
    DEFAULT_BIN_EDGES, DEFAULT_BIN_LABELS,
)


def process_energy_slice(mult: np.ndarray, b: np.ndarray, *, seed: int) -> dict:
    """Calibrate truth-tuned binning on a 50/50 split of one energy."""
    n_events = len(b)
    train_idx, test_idx = train_test_split_indices(n_events, train_frac=0.5, seed=seed)
    calib = fit(mult[train_idx], b[train_idx])

    bins_train, b_pred_train = predict(mult[train_idx], calib)
    bins_test, b_pred_test = predict(mult[test_idx], calib)

    # Per-bin diagnostics on the held-out half.
    n_bins = len(calib.bin_labels)
    test_res_std = np.full(n_bins, np.nan, dtype=np.float64)
    test_res_mae = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    b_test = b[test_idx]
    for k in range(n_bins):
        sel = bins_test == k
        counts[k] = int(sel.sum())
        if counts[k] > 0:
            res = b_pred_test[sel] - b_test[sel]
            test_res_std[k] = float(res.std())
            test_res_mae[k] = float(np.abs(res).mean())

    # Recompose into per-event arrays in the original event order.
    bin_full = np.empty(n_events, dtype=np.int16)
    b_pred_full = np.empty(n_events, dtype=np.float32)
    is_train = np.zeros(n_events, dtype=bool)
    bin_full[train_idx] = bins_train; bin_full[test_idx] = bins_test
    b_pred_full[train_idx] = b_pred_train; b_pred_full[test_idx] = b_pred_test
    is_train[train_idx] = True

    return {
        "calib": calib,
        "bin_full": bin_full,
        "b_pred_full": b_pred_full,
        "is_train": is_train,
        "n_events": n_events,
        "n_train": int(len(train_idx)),
        "counts_per_bin": counts.tolist(),
        "test_res_mae_per_bin": test_res_mae.tolist(),
        "test_res_std_per_bin": test_res_std.tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.cache, "r") as h:
        mult = h["mult_lab"][:]
        b = h["b"][:]
        eid = h["energy_id"][:]
        sqrts_per_energy = list(h.attrs["sqrtsNN_per_energy"])
        n_per_energy = list(h.attrs["n_events_per_energy"])

    cum = np.cumsum([0] + n_per_energy)
    report = {}
    for e_id, (sqrtsNN, n_e) in enumerate(zip(sqrts_per_energy, n_per_energy)):
        lo, hi = int(cum[e_id]), int(cum[e_id + 1])
        result = process_energy_slice(mult[lo:hi], b[lo:hi], seed=args.seed)
        tag = f"{sqrtsNN:.1f}".replace(".", "p")
        out_path = args.output_dir / f"truth_auau_{tag}GeV.h5"
        with h5py.File(out_path, "w") as h:
            h.create_dataset("centrality_bin", data=result["bin_full"])
            h.create_dataset("b_pred", data=result["b_pred_full"])
            h.create_dataset("is_train", data=result["is_train"])
            h.create_dataset("mult_thresholds", data=result["calib"].mult_thresholds)
            h.create_dataset("b_means", data=result["calib"].b_means)
            h.create_dataset("b_stds", data=result["calib"].b_stds)
            h.attrs["sqrtsNN"] = sqrtsNN
            h.attrs["n_events"] = result["n_events"]
            h.attrs["n_train"] = result["n_train"]
            h.attrs["bin_edges"] = list(DEFAULT_BIN_EDGES)
            h.attrs["bin_labels"] = list(DEFAULT_BIN_LABELS)
            h.attrs["source_cache"] = str(args.cache)
            h.attrs["observable"] = "mult_lab (0<eta_lab<2 charged)"
            h.attrs["split_seed"] = args.seed
        report[f"auau_{tag}GeV"] = {
            "sqrtsNN": sqrtsNN,
            "n_events": result["n_events"],
            "test_res_mae_per_bin": result["test_res_mae_per_bin"],
            "test_counts_per_bin": result["counts_per_bin"],
        }
        print(f"  {sqrtsNN} GeV: MAE per bin = {[f'{v:.2f}' if np.isfinite(v) else 'nan' for v in result['test_res_mae_per_bin']]} fm")

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote summary to {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
