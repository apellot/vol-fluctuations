"""Resolution-vs-true-b plot for a centrality baseline at each energy.

Works for either the truth-tuned or the Glauber baseline — both produce the same
output schema (centrality_bin, b_pred, bin_labels). Pass --truth-inputs and
--pred-inputs in matching order, and use --label to control the figure title.

For each event in the held-out test set, plot |b_pred − b_true| as a function
of true b. The predicted b is constant within a centrality bin, so the residual
is exactly the offset between the bin's mean truth-b and the event's actual
truth-b. The "unresolvable peripheral" region — where adjacent percentile bins
collapse to the same multiplicity threshold of zero, an artefact of the
spectator-dominated FXT regime — is shaded explicitly.

Usage:
    python scripts/plot_baseline_resolution.py \\
        --truth-inputs data/processed/auau_{3p2,3p5,3p9,4p5}GeV.h5 \\
        --pred-inputs  data/processed/truth/smash/truth_auau_{3p2,3p5,3p9,4p5}GeV.h5 \\
        --label Truth \\
        --output-dir   figures/baselines/truth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def plot_one_energy(truth_path: Path, pred_path: Path, out_dir: Path, label: str) -> None:
    with h5py.File(truth_path, "r") as h:
        b_true = h["b"][:]
    with h5py.File(pred_path, "r") as h:
        b_pred = h["b_pred"][:]
        bins = h["centrality_bin"][:]
        sqrtsNN = float(h.attrs["sqrtsNN"])
        labels = list(h.attrs["bin_labels"])
        # is_train is only present for the truth-tuned baseline (which has a train/test split). Glauber
        # is calibrated entirely on Glauber-MC events, so every SMASH event counts as test.
        is_train = h["is_train"][:] if "is_train" in h else np.zeros(len(b_pred), dtype=bool)

    # Use the held-out test half only.
    test = ~is_train
    b_t = b_true[test]
    b_p = b_pred[test]
    bn = bins[test]
    # Glauber may produce NaN predictions in empty bins — drop those for the diagnostic.
    finite = np.isfinite(b_p)
    b_t = b_t[finite]; b_p = b_p[finite]; bn = bn[finite]

    # Per-bin profile of the absolute residual vs true b.
    b_grid = np.linspace(0, b_t.max(), 30)
    cx = 0.5 * (b_grid[:-1] + b_grid[1:])
    mae_profile = np.full(len(cx), np.nan)
    std_profile = np.full(len(cx), np.nan)
    for i in range(len(cx)):
        sel = (b_t >= b_grid[i]) & (b_t < b_grid[i + 1])
        if sel.sum() > 5:
            res = b_p[sel] - b_t[sel]
            mae_profile[i] = float(np.abs(res).mean())
            std_profile[i] = float(res.std())

    # Identify the "unresolvable peripheral" region: events that fell into the
    # last bin (70-80% by percentile labeling) because their multiplicity was
    # at or below the collapsed-zero threshold.
    last_bin = len(labels) - 1
    last_bin_b_min = float(b_t[bn == last_bin].min()) if (bn == last_bin).any() else np.nan
    frac_last_bin = float((bn == last_bin).mean())

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Scatter of |b_pred − b_true| vs b_true.
    ax.scatter(b_t, np.abs(b_p - b_t), s=2, alpha=0.15, color="C0", rasterized=True, label="event-by-event")

    # Profile of MAE.
    ax.plot(cx, mae_profile, "o-", color="C3", ms=3, lw=1.5, label="MAE profile")
    # 1σ residual envelope.
    ax.fill_between(cx,
                    np.where(np.isfinite(std_profile), mae_profile - std_profile, np.nan),
                    np.where(np.isfinite(std_profile), mae_profile + std_profile, np.nan),
                    color="C3", alpha=0.15, label="±1σ residual")

    # Shade the "unresolvable peripheral" region — events in the last centrality bin
    # span a wide range of true b because the multiplicity is at the floor and gives
    # no information. The shading is the visual evidence that the classical baseline
    # has no resolving power in that region.
    if np.isfinite(last_bin_b_min):
        ax.axvspan(last_bin_b_min, b_t.max(), color="gray", alpha=0.15,
                   label=f"unresolved peripheral ({100*frac_last_bin:.0f}% of events)")

    ax.set_xlabel("true impact parameter b (fm)")
    ax.set_ylabel(r"$|b_{\rm pred} - b_{\rm true}|$ (fm)")
    ax.set_title(f"{label} resolution vs true b   ({sqrtsNN} GeV)")
    ax.set_ylim(0, max(8, 1.2 * float(np.nanmax(mae_profile + (std_profile if np.any(np.isfinite(std_profile)) else 0)))))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    tag = f"{sqrtsNN:.1f}".replace(".", "p")
    file_stem = label.lower().replace(" ", "_")
    fig.savefig(out_dir / f"{file_stem}_resolution_{tag}GeV.png", dpi=140)
    fig.savefig(out_dir / f"{file_stem}_resolution_{tag}GeV.pdf")
    plt.close(fig)
    print(f"  {sqrtsNN} GeV: {100*frac_last_bin:.0f}% events in unresolved peripheral "
          f"(b >= {last_bin_b_min:.1f} fm)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--truth-inputs", required=True, nargs="+", type=Path)
    p.add_argument("--pred-inputs", required=True, nargs="+", type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--label", default="Truth", help="Display name (e.g., 'Truth', 'Glauber')")
    args = p.parse_args()
    if len(args.truth_inputs) != len(args.pred_inputs):
        raise SystemExit("--truth-inputs and --pred-inputs must have the same length")
    for tp, pp in zip(args.truth_inputs, args.pred_inputs):
        plot_one_energy(tp.expanduser(), pp.expanduser(), args.output_dir.expanduser(), args.label)


if __name__ == "__main__":
    main()
