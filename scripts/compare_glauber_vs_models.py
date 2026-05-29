"""Glauber-vs-models comparison: event-by-event impact-parameter prediction.

STALE AFTER 2026-05-28 RESTRUCTURE. The hardcoded FULL_CACHE/LAB_CACHE and the
MODELS lineup below point at the old SMASH, truth-level, b<11 study with its
full-vs-lab cache split — none of which is the headline pipeline anymore
(UrQMD + detector + b<14, single common cache). Do NOT repoint these paths
piecemeal: the full/lab-split design is obsolete and the UrQMD+detector model
predictions do not exist until the retraining step. For the paper-grade,
common-split comparison use scripts/validate_models.py against
data/processed/cached/urqmd_padded_det.h5. This script is retained only as the
historical "quick look" on the old truth artifacts and should be rewritten (or
deleted) once the UrQMD+detector models are trained.

Answers two questions directly:
  1. How well does each trained model predict b on an event-by-event basis?
     -> b_pred-vs-b_true 2D histograms with MAE / RMSE / Pearson r.
  2. How do the ML models compare to the Glauber baseline (and the Truth
     oracle) as a function of true centrality?
     -> resolution sigma(b_pred - b_true) and bias vs true b, all overlaid.

IMPORTANT CAVEAT (this is the "quick, with caveats" comparison, by design).
The trained models on disk come from three different pipelines and are NOT on a
common event set or split:

  * MLP, Glauber, Truth  -> full event set (CM-frame), per-energy original order.
  * DeepSets, MLP-pool   -> lab-filtered subset (all_lab_truth.h5): events with
                            no charged track in 0 < eta_lab < 1.5 are removed
                            (~40% of events, the ultra-peripheral tail). These
                            models therefore never see the regime where the
                            classical baselines fail hardest, so their apparent
                            advantage here is *conservative*.
  * Each ML model uses its own independently-drawn test split.

Every figure is stamped with this caveat so the plots can't be misread. For a
paper-grade comparison the models must be retrained on one common cache/split
(all_padded.h5) -- see CLAUDE.md Task 4/5. This script is the fast look using
what is already trained.

Usage:
    python scripts/compare_glauber_vs_models.py --output-dir figures/comparison
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Energies and their filename tags, in the order the caches were built.
ENERGIES = [(3.2, "3p2"), (3.5, "3p5"), (3.9, "3p9"), (4.5, "4p5")]

# Per-event truth b lives in the two caches, concatenated in energy order.
FULL_CACHE = Path("data/processed/cached/all_padded.h5")      # 50k/100k/100k/100k
LAB_CACHE = Path("data/processed/cached/all_lab_truth.h5")    # lab-filtered subset


@dataclass
class ModelSpec:
    name: str          # short key, also the prediction-file prefix
    label: str         # display name
    pred_tmpl: str     # prediction-file path template, {tag} -> e.g. 3p5
    space: str         # "full" or "lab" -- which cache supplies b_true
    split: str         # "test", "not_train", or "all" -- which events to score
    color: str
    ls: str            # line style for the overlay plots


# Order matters only for legend/zorder; Glauber + Truth drawn last so they sit on top.
MODELS = [
    ModelSpec("mlp",      "MLP",          "data/processed/mlp/baseline_v1/mlp_pred_auau_{tag}GeV.h5",        "full", "test",      "tab:blue",       "--"),
    ModelSpec("deepsets", "DeepSets",     "data/processed/deepsets/v1_truth/deepsets_pred_auau_{tag}GeV.h5", "lab",  "test",      "mediumseagreen", "--"),
    ModelSpec("mlp_pool", "MLP-pool",     "data/processed/mlp_pool/v1_truth/mlp_pool_pred_auau_{tag}GeV.h5", "lab",  "test",      "darkorange",     "--"),
    ModelSpec("glauber",  "Glauber-NBD",  "data/processed/glauber/smash/glauber_pred_auau_{tag}GeV.h5",     "full", "all",       "steelblue",      "-"),
    ModelSpec("truth",    "Truth oracle", "data/processed/truth/smash/truth_auau_{tag}GeV.h5",              "full", "not_train", "black",          "-"),
]

CAVEAT = ("DeepSets/MLP-pool: lab-filtered subset (no track in 0<eta_lab<1.5 removed); "
          "MLP/Glauber/Truth: full set. Each ML model on its own test split. "
          "Not apples-to-apples — see script docstring.")


def load_truth_b_per_energy() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return {tag: b_true} for the full and lab event spaces, sliced by energy.

    Both caches store one flat b array concatenated in the energy order above;
    we cut it back into per-energy pieces using n_events_per_energy.
    """
    def slice_by_energy(cache_path: Path) -> dict[str, np.ndarray]:
        with h5py.File(cache_path, "r") as h:
            b = h["b"][:].astype(np.float32)
            n_per = list(h.attrs["n_events_per_energy"])
        edges = np.cumsum([0] + n_per)
        return {tag: b[edges[i]:edges[i + 1]] for i, (_, tag) in enumerate(ENERGIES)}

    return slice_by_energy(FULL_CACHE), slice_by_energy(LAB_CACHE)


def load_model_eval(spec: ModelSpec, tag: str,
                    b_full: np.ndarray, b_lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (b_true, b_pred) for one model at one energy, on its scored events.

    The split flag selects which events to score:
      * "test"      -> events with is_test == True (ML models).
      * "not_train" -> events with is_train == False (Truth oracle: never score
                       the half its bin-means were fit on).
      * "all"       -> every event (Glauber has no fit/score split in this sense).
    """
    b_true_full = b_full if spec.space == "full" else b_lab
    with h5py.File(spec.pred_tmpl.format(tag=tag), "r") as h:
        b_pred = h["b_pred"][:].astype(np.float32)
        if spec.split == "test":
            mask = h["is_test"][:]
        elif spec.split == "not_train":
            mask = ~h["is_train"][:]
        else:  # "all"
            mask = np.ones(b_pred.shape[0], dtype=bool)

    if b_pred.shape[0] != b_true_full.shape[0]:
        raise ValueError(f"{spec.name} {tag}: {b_pred.shape[0]} preds vs "
                         f"{b_true_full.shape[0]} truth events -- cache/pred mismatch")

    b_true = b_true_full[mask]
    b_pred = b_pred[mask]
    # Glauber can emit NaN in collapsed/empty peripheral bins; drop them.
    finite = np.isfinite(b_pred)
    return b_true[finite], b_pred[finite]


def metrics(b_true: np.ndarray, b_pred: np.ndarray) -> dict[str, float]:
    res = b_pred - b_true
    return {
        "n": int(b_true.size),
        "mae": float(np.abs(res).mean()),
        "rmse": float(np.sqrt((res ** 2).mean())),
        "bias": float(res.mean()),
        "corr": float(np.corrcoef(b_pred, b_true)[0, 1]),
    }


def profile_vs_true_b(b_true: np.ndarray, b_pred: np.ndarray,
                      edges: np.ndarray, min_count: int = 25):
    """Per-true-b-bin resolution (std of residual) and bias (mean of residual)."""
    res = b_pred - b_true
    centers = 0.5 * (edges[:-1] + edges[1:])
    std = np.full(centers.size, np.nan)
    bias = np.full(centers.size, np.nan)
    for i in range(centers.size):
        sel = (b_true >= edges[i]) & (b_true < edges[i + 1])
        if sel.sum() >= min_count:
            std[i] = float(res[sel].std())
            bias[i] = float(res[sel].mean())
    return centers, std, bias


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_scatter_grid(evals: dict, out_dir: Path) -> None:
    """Rows = models, cols = energies. 2D histogram of b_pred vs b_true."""
    nrow, ncol = len(MODELS), len(ENERGIES)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow),
                             sharex=True, sharey=True)
    for r, spec in enumerate(MODELS):
        for c, (sqrtsNN, tag) in enumerate(ENERGIES):
            ax = axes[r, c]
            b_true, b_pred = evals[spec.name][tag]
            m = metrics(b_true, b_pred)
            lim = 16.0
            ax.hist2d(b_true, b_pred, bins=60, range=[[0, lim], [0, lim]],
                      cmap="viridis", cmin=1)
            ax.plot([0, lim], [0, lim], "r--", lw=1)
            ax.set_xlim(0, lim); ax.set_ylim(0, lim)
            ax.text(0.04, 0.96, f"MAE={m['mae']:.2f}\nr={m['corr']:.3f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8,
                    color="white", bbox=dict(fc="black", alpha=0.4, lw=0))
            if r == 0:
                ax.set_title(f"{sqrtsNN} GeV", fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{spec.label}\n$b_{{\\rm pred}}$ (fm)", fontsize=10)
            if r == nrow - 1:
                ax.set_xlabel(r"$b_{\rm true}$ (fm)", fontsize=10)
    fig.suptitle("Event-by-event impact-parameter prediction", fontsize=14, y=0.995)
    fig.text(0.5, 0.005, CAVEAT, ha="center", fontsize=7, style="italic", wrap=True)
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    _save(fig, out_dir, "scatter_grid")


def plot_resolution_overlay(evals: dict, out_dir: Path) -> None:
    """sigma(b_pred - b_true) vs true b, one panel per energy, all models overlaid."""
    edges = np.linspace(0, 16, 17)
    fig, axes = plt.subplots(1, len(ENERGIES), figsize=(4.2 * len(ENERGIES), 4.0),
                             sharey=True)
    for ax, (sqrtsNN, tag) in zip(axes, ENERGIES):
        for spec in MODELS:
            b_true, b_pred = evals[spec.name][tag]
            centers, std, _ = profile_vs_true_b(b_true, b_pred, edges)
            lw = 2.4 if spec.name in ("glauber", "truth") else 1.6
            ax.plot(centers, std, marker="o", ms=3, lw=lw, color=spec.color,
                    ls=spec.ls, label=spec.label)
        ax.set_title(f"{sqrtsNN} GeV")
        ax.set_xlabel(r"$b_{\rm true}$ (fm)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"resolution  $\sigma(b_{\rm pred}-b_{\rm true})$ (fm)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Centrality resolution vs true b  (lower is better)", fontsize=13)
    fig.text(0.5, 0.005, CAVEAT, ha="center", fontsize=7, style="italic", wrap=True)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    _save(fig, out_dir, "resolution_vs_b")


def plot_bias_overlay(evals: dict, out_dir: Path) -> None:
    """mean(b_pred - b_true) vs true b: shows systematic pull toward the mean b."""
    edges = np.linspace(0, 16, 17)
    fig, axes = plt.subplots(1, len(ENERGIES), figsize=(4.2 * len(ENERGIES), 4.0),
                             sharey=True)
    for ax, (sqrtsNN, tag) in zip(axes, ENERGIES):
        for spec in MODELS:
            b_true, b_pred = evals[spec.name][tag]
            centers, _, bias = profile_vs_true_b(b_true, b_pred, edges)
            lw = 2.4 if spec.name in ("glauber", "truth") else 1.6
            ax.plot(centers, bias, marker="o", ms=3, lw=lw, color=spec.color,
                    ls=spec.ls, label=spec.label)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(f"{sqrtsNN} GeV")
        ax.set_xlabel(r"$b_{\rm true}$ (fm)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"bias  $\langle b_{\rm pred}-b_{\rm true}\rangle$ (fm)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Centrality bias vs true b  (regression-to-mean shows as a tilt)", fontsize=13)
    fig.text(0.5, 0.005, CAVEAT, ha="center", fontsize=7, style="italic", wrap=True)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    _save(fig, out_dir, "bias_vs_b")


def _save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=150)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"  saved {out_dir / stem}.png/.pdf")


def print_table(evals: dict) -> None:
    print("\n" + "=" * 92)
    print("Event-by-event b prediction  (scored events per model; see caveat)")
    print(f"{'Model':>13s} {'energy':>7s} {'N':>8s} {'MAE':>7s} {'RMSE':>7s} {'bias':>7s} {'corr':>7s}")
    print("-" * 92)
    for spec in MODELS:
        for sqrtsNN, tag in ENERGIES:
            b_true, b_pred = evals[spec.name][tag]
            m = metrics(b_true, b_pred)
            print(f"{spec.label:>13s} {sqrtsNN:>6.1f}G {m['n']:>8d} "
                  f"{m['mae']:>6.3f} {m['rmse']:>6.3f} {m['bias']:>+6.3f} {m['corr']:>6.3f}")
        print("-" * 92)
    print("=" * 92)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("figures/comparison"))
    args = p.parse_args()

    b_full, b_lab = load_truth_b_per_energy()

    # Load every (model, energy) once into memory; the arrays are small (just b).
    evals: dict[str, dict[str, tuple]] = {}
    for spec in MODELS:
        evals[spec.name] = {}
        for _, tag in ENERGIES:
            evals[spec.name][tag] = load_model_eval(spec, tag, b_full[tag], b_lab[tag])

    print_table(evals)
    print("\nGenerating figures...")
    plot_scatter_grid(evals, args.output_dir)
    plot_resolution_overlay(evals, args.output_dir)
    plot_bias_overlay(evals, args.output_dir)
    print(f"\nDone. Figures in {args.output_dir}/")


if __name__ == "__main__":
    main()
