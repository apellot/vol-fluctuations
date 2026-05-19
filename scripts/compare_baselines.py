"""Head-to-head: MAE vs true b for truth, Glauber, and MLP baselines, per energy.

Both classical baselines emit a single constant predicted-b per centrality bin, so
their MAE-vs-true-b traces have the saw-tooth shape of a bin classifier. The MLP
emits a continuous μ per event, so its trace is smooth. Comparing them on the
same axes (same true-b binning, same test events) shows where ML wins — most
visibly in the peripheral region where classical thresholds collapse.

Usage:
    python scripts/compare_baselines.py \\
        --truth-inputs data/processed/auau_{3p2,3p5,3p9,4p5}GeV.h5 \\
        --truth-preds  data/processed/truth/truth_auau_{3p2,3p5,3p9,4p5}GeV.h5 \\
        --glauber-preds data/processed/glauber/glauber_pred_auau_{3p2,3p5,3p9,4p5}GeV.h5 \\
        --mlp-preds    data/processed/mlp/baseline_v1/mlp_pred_auau_{3p2,3p5,3p9,4p5}GeV.h5 \\
        --output-dir   figures/compare/baseline_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _profile_mae(b_true: np.ndarray, b_pred: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    n = len(centers)
    mae = np.full(n, np.nan)
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        sel = (b_true >= edges[i]) & (b_true < edges[i + 1]) & np.isfinite(b_pred)
        counts[i] = sel.sum()
        if counts[i] > 5:
            mae[i] = float(np.abs(b_pred[sel] - b_true[sel]).mean())
    return centers, mae, counts


def make_plot_for_energy(truth_h5: Path, t_pred: Path, g_pred: Path, m_pred: Path, out_dir: Path) -> dict:
    with h5py.File(truth_h5, "r") as h:
        b_true_full = h["b"][:]
        sqrtsNN = float(h.attrs["sqrtsNN"])
    with h5py.File(t_pred, "r") as h:
        b_pred_truth = h["b_pred"][:]
        is_train_truth = h["is_train"][:]
    with h5py.File(g_pred, "r") as h:
        b_pred_glauber = h["b_pred"][:]
    with h5py.File(m_pred, "r") as h:
        b_pred_mlp = h["b_pred"][:]
        is_test_mlp = h["is_test"][:]

    # Use the MLP's test split as the common evaluation set so the comparison is
    # apples-to-apples (the MLP never saw these events, and the classical baselines
    # are calibrated independently).
    eval_mask = is_test_mlp
    b_true = b_true_full[eval_mask]
    b_t = b_pred_truth[eval_mask]
    b_g = b_pred_glauber[eval_mask]
    b_m = b_pred_mlp[eval_mask]

    edges = np.linspace(0, float(b_true.max()), 26)
    cx, mae_t, n_t = _profile_mae(b_true, b_t, edges)
    cx, mae_g, n_g = _profile_mae(b_true, b_g, edges)
    cx, mae_m, n_m = _profile_mae(b_true, b_m, edges)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(cx, mae_t, "o-", color="C0", ms=4, lw=1.5, label="Truth-tuned percentile")
    ax.plot(cx, mae_g, "s-", color="C1", ms=4, lw=1.5, label="Glauber → RefMult")
    ax.plot(cx, mae_m, "^-", color="C3", ms=4, lw=1.5, label="MLP (evidential)")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("true impact parameter b (fm)")
    ax.set_ylabel(r"$\langle |b_{\rm pred} - b_{\rm true}| \rangle$ per b-bin (fm)")
    ax.set_title(f"Baseline comparison — {sqrtsNN} GeV   (N_test = {eval_mask.sum():,})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    tag = f"{sqrtsNN:.1f}".replace(".", "p")
    fig.savefig(out_dir / f"compare_{tag}GeV.png", dpi=140)
    fig.savefig(out_dir / f"compare_{tag}GeV.pdf")
    plt.close(fig)

    overall = {
        "sqrtsNN": sqrtsNN,
        "n_test": int(eval_mask.sum()),
        "mae_overall": {
            "truth": float(np.abs(b_t - b_true).mean()),
            "glauber": float(np.abs(np.where(np.isfinite(b_g), b_g, b_true) - b_true).mean()),  # Glauber NaN events count as their own b? simpler: drop nan
            "mlp": float(np.abs(b_m - b_true).mean()),
        },
    }
    # Recompute Glauber overall MAE excluding NaN predictions for honesty.
    finite_g = np.isfinite(b_g)
    overall["mae_overall"]["glauber"] = float(np.abs(b_g[finite_g] - b_true[finite_g]).mean())
    overall["n_glauber_finite"] = int(finite_g.sum())
    return overall


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--truth-inputs", required=True, nargs="+", type=Path)
    p.add_argument("--truth-preds", required=True, nargs="+", type=Path)
    p.add_argument("--glauber-preds", required=True, nargs="+", type=Path)
    p.add_argument("--mlp-preds", required=True, nargs="+", type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    n = len(args.truth_inputs)
    assert all(len(x) == n for x in [args.truth_preds, args.glauber_preds, args.mlp_preds])

    summaries = []
    for ti, tp, gp, mp in zip(args.truth_inputs, args.truth_preds, args.glauber_preds, args.mlp_preds):
        s = make_plot_for_energy(ti, tp, gp, mp, args.output_dir)
        summaries.append(s)
        m = s["mae_overall"]
        print(f"  {s['sqrtsNN']} GeV  (N={s['n_test']:,}):  "
              f"truth {m['truth']:.3f}  Glauber {m['glauber']:.3f}  MLP {m['mlp']:.3f}  fm")

    print("\nPer-energy overall test-set b-MAE (fm)")


if __name__ == "__main__":
    main()
