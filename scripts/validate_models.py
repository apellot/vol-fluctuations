# -*- coding: utf-8 -*-
"""Model validation and diagnostic plots.

Produces four sets of diagnostics for each trained model + Glauber baseline:
    1. b_pred vs b_true scatter (one panel per energy)
    2. MAE vs centrality bin (resolution profile)
    3. Uncertainty calibration (NIG 68% coverage) -- ML models only
    4. Pull distribution -- ML models only

When --urqmd-cache / --urqmd-pred-dirs are supplied the same four plot sets
are regenerated for the UrQMD cross-transport evaluation (files suffixed
_urqmd) and a side-by-side SMASH-vs-UrQMD MAE comparison table is printed.

Usage:
    python scripts/validate_models.py \
        --cache data/processed/cached/all_padded.h5 \
        --truth-dir data/processed/truth/ \
        --glauber-dir data/processed/glauber/ \
        --pred-dirs efn:data/processed/efn/baseline_v1 \
                    pfn:data/processed/pfn/baseline_v1 \
                    deepsets:data/processed/deepsets/baseline_v1 \
                    gnn:data/processed/gnn/baseline_v1 \
        --urqmd-cache data/processed/cached/urqmd_padded.h5 \
        --urqmd-truth-dir data/processed/urqmd_truth/ \
        --urqmd-pred-dirs efn:data/processed/efn/urqmd_v1 \
                          pfn:data/processed/pfn/urqmd_v1 \
                          deepsets:data/processed/deepsets/urqmd_v1 \
                          gnn:data/processed/gnn/urqmd_v1 \
        --output-dir figures/validation/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── style ─────────────────────────────────────────────────────────────────────

COLORS = {
    "glauber":  "steelblue",
    "efn":      "crimson",
    "pfn":      "darkorange",
    "deepsets": "mediumseagreen",
    "gnn":      "mediumpurple",
}
LABELS = {
    "glauber":  "Glauber-NBD",
    "efn":      "EFN",
    "pfn":      "PFN",
    "deepsets": "DeepSets",
    "gnn":      "GNN",
}
LINESTYLES = {
    "glauber":  "-",
    "efn":      "--",
    "pfn":      "--",
    "deepsets": "--",
    "gnn":      "--",
}

# ── data loading ──────────────────────────────────────────────────────────────

def load_cache(cache_path: Path) -> dict:
    with h5py.File(cache_path, "r") as h:
        return {
            "b":       h["b"][:].astype(np.float32),
            "sqrts":   list(h.attrs["sqrtsNN_per_energy"]),
            "n_per_e": list(h.attrs["n_events_per_energy"]),
        }


def load_truth_bins(truth_dir: Path, sqrtsNN: float) -> np.ndarray:
    tag = f"{sqrtsNN:.1f}".replace(".", "p")
    with h5py.File(truth_dir / f"truth_auau_{tag}GeV.h5", "r") as h:
        return h["centrality_bin"][:].astype(np.int16)


def load_glauber(glauber_dir: Path, sqrtsNN: float, n_events: int) -> dict:
    tag = f"{sqrtsNN:.1f}".replace(".", "p")
    p = glauber_dir / f"glauber_pred_auau_{tag}GeV.h5"
    with h5py.File(p, "r") as h:
        b_pred = h["b_pred"][:].astype(np.float32)
        cent   = h["centrality_bin"][:].astype(np.int16)
    # Glauber has no train/test split -- use all events as test
    is_test = np.ones(n_events, dtype=bool)
    return {
        "b_pred":    b_pred,
        "is_test":   is_test,
        "cent_bins": cent,
        "has_uq":    False,
    }


def load_pred(pred_dir: Path, arch: str, sqrtsNN: float) -> dict:
    tag = f"{sqrtsNN:.1f}".replace(".", "p")
    p = pred_dir / f"{arch}_pred_auau_{tag}GeV.h5"
    with h5py.File(p, "r") as h:
        return {
            "b_pred":        h["b_pred"][:].astype(np.float32),
            "total_var":     h["total_var"][:].astype(np.float32),
            "aleatoric_var": h["aleatoric_var"][:].astype(np.float32),
            "epistemic_var": h["epistemic_var"][:].astype(np.float32),
            "is_test":       h["is_test"][:],
            "cent_bins":     h["centrality_bin"][:].astype(np.int16),
            "has_uq":        True,
        }


def load_urqmd_pred(pred_dir: Path, arch: str, sqrtsNN: float) -> dict:
    """Load cross-transport (UrQMD) predictions.

    The model was trained on SMASH; all UrQMD events are unseen, so we
    use every event as the evaluation set (is_test = all True).
    """
    tag = f"{sqrtsNN:.1f}".replace(".", "p")
    p = pred_dir / f"{arch}_pred_urqmd_auau_{tag}GeV.h5"
    with h5py.File(p, "r") as h:
        n = h["b_pred"].shape[0]
        return {
            "b_pred":        h["b_pred"][:].astype(np.float32),
            "total_var":     h["total_var"][:].astype(np.float32),
            "aleatoric_var": h["aleatoric_var"][:].astype(np.float32),
            "epistemic_var": h["epistemic_var"][:].astype(np.float32),
            # All UrQMD events are cross-transport test events
            "is_test":       np.ones(n, dtype=bool),
            "cent_bins":     h["centrality_bin"][:].astype(np.int16),
            "has_uq":        True,
        }


# ── plot 1: scatter b_pred vs b_true ─────────────────────────────────────────

def plot_scatter(data_by_energy: dict, all_archs: list, sqrts_list: list,
                 output_dir: Path) -> None:
    for arch in all_archs:
        fig, axes = plt.subplots(1, len(sqrts_list),
                                 figsize=(5 * len(sqrts_list), 5), sharey=True)
        fig.suptitle(f"{LABELS[arch]} -- b_pred vs b_true (test set)", fontsize=13)

        for ax, sqrtsNN in zip(axes, sqrts_list):
            d      = data_by_energy[sqrtsNN][arch]
            mask   = d["is_test"]
            b_true = d["b_true"][mask]
            b_pred = d["b_pred"][mask]
            mae    = float(np.abs(b_pred - b_true).mean())
            corr   = float(np.corrcoef(b_pred, b_true)[0, 1])

            h2, xe, ye = np.histogram2d(b_true, b_pred, bins=60)
            ax.pcolormesh(xe, ye, h2.T, cmap="Blues")
            lim = max(b_true.max(), b_pred.max()) * 1.05
            ax.plot([0, lim], [0, lim], "r--", lw=1, label="ideal")
            ax.set_xlabel("b_true (fm)", fontsize=11)
            ax.set_ylabel("b_pred (fm)", fontsize=11)
            ax.set_title(f"sqrt(s)={sqrtsNN} GeV\nMAE={mae:.3f} fm  r={corr:.3f}")
            ax.legend(fontsize=8)

        plt.tight_layout()
        out = output_dir / f"scatter_{arch}.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(output_dir / f"scatter_{arch}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


# ── plot 2: MAE vs centrality bin ─────────────────────────────────────────────

def plot_mae_profile(data_by_energy: dict, all_archs: list, sqrts_list: list,
                     n_bins: int, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(sqrts_list),
                             figsize=(5 * len(sqrts_list), 4), sharey=False)
    fig.suptitle("MAE per centrality bin (test set)", fontsize=13)
    bin_centers = np.arange(n_bins)
    pct_labels  = [f"{int(100*i/n_bins)}-{int(100*(i+1)/n_bins)}%"
                   for i in range(n_bins)]

    for ax, sqrtsNN in zip(axes, sqrts_list):
        for arch in all_archs:
            d      = data_by_energy[sqrtsNN][arch]
            mask   = d["is_test"]
            b_true = d["b_true"][mask]
            b_pred = d["b_pred"][mask]
            bins   = d["cent_bins"][mask]

            maes = []
            for b in range(n_bins):
                sel = bins == b
                maes.append(float(np.abs(b_pred[sel] - b_true[sel]).mean())
                            if sel.sum() > 0 else np.nan)

            lw = 2.5 if arch == "glauber" else 1.5
            ax.plot(bin_centers, maes,
                    label=LABELS[arch], color=COLORS[arch],
                    ls=LINESTYLES[arch], lw=lw, marker="o", markersize=4)

        ax.set_title(f"sqrt(s) = {sqrtsNN} GeV")
        ax.set_xlabel("Centrality bin (0=central)")
        ax.set_ylabel("MAE (fm)")
        ax.set_xticks(bin_centers)
        ax.set_xticklabels(pct_labels, rotation=45, fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / "mae_profile.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(output_dir / "mae_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ── plot 3: calibration coverage (ML only) ───────────────────────────────────

def plot_calibration(data_by_energy: dict, ml_archs: list, sqrts_list: list,
                     output_dir: Path) -> None:
    cls = np.linspace(0.05, 0.95, 19)
    fig, axes = plt.subplots(1, len(sqrts_list),
                             figsize=(5 * len(sqrts_list), 4), sharey=True)
    fig.suptitle("Uncertainty calibration -- coverage vs confidence level", fontsize=13)

    for ax, sqrtsNN in zip(axes, sqrts_list):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
        for arch in ml_archs:
            d      = data_by_energy[sqrtsNN][arch]
            mask   = d["is_test"]
            b_true = d["b_true"][mask]
            b_pred = d["b_pred"][mask]
            sigma  = np.sqrt(d["total_var"][mask])

            coverages = []
            for cl in cls:
                z = stats.norm.ppf((1 + cl) / 2)
                coverages.append(float((np.abs(b_pred - b_true) <= z * sigma).mean()))

            ax.plot(cls, coverages, label=LABELS[arch],
                    color=COLORS[arch], marker="o", markersize=3, lw=1.5)

        ax.set_title(f"sqrt(s) = {sqrtsNN} GeV")
        ax.set_xlabel("Confidence level")
        ax.set_ylabel("Empirical coverage")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    plt.tight_layout()
    out = output_dir / "calibration.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(output_dir / "calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ── plot 4: pull distribution (ML only) ──────────────────────────────────────

def plot_pulls(data_by_energy: dict, ml_archs: list, sqrts_list: list,
               output_dir: Path) -> None:
    fig, axes = plt.subplots(len(ml_archs), len(sqrts_list),
                             figsize=(4 * len(sqrts_list), 3 * len(ml_archs)))
    fig.suptitle("Pull distributions (b_pred - b_true) / sigma", fontsize=13)
    x = np.linspace(-5, 5, 200)

    for i, arch in enumerate(ml_archs):
        for j, sqrtsNN in enumerate(sqrts_list):
            ax     = axes[i][j] if len(ml_archs) > 1 else axes[j]
            d      = data_by_energy[sqrtsNN][arch]
            mask   = d["is_test"]
            b_true = d["b_true"][mask]
            b_pred = d["b_pred"][mask]
            sigma  = np.sqrt(d["total_var"][mask])
            pulls  = (b_pred - b_true) / np.clip(sigma, 1e-6, None)

            ax.hist(pulls, bins=50, density=True, alpha=0.6,
                    color=COLORS[arch])
            ax.plot(x, stats.norm.pdf(x), "k--", lw=1, label="N(0,1)")
            ax.set_xlim(-5, 5)
            ax.set_title(f"{LABELS[arch]} {sqrtsNN} GeV\n"
                         f"mean={pulls.mean():.2f} std={pulls.std():.2f}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / "pulls.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(output_dir / "pulls.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ── summary tables ────────────────────────────────────────────────────────────

def _mae_table(data_by_energy: dict, all_archs: list, sqrts_list: list,
               title: str) -> dict[str, list[float]]:
    """Print a MAE summary table and return {arch: [mae_per_energy]}."""
    col_w = 12
    print("\n" + "=" * (col_w + 16 * len(sqrts_list) + 12))
    print(title)
    print(f"{'Model':>{col_w}s}", end="")
    for sqrtsNN in sqrts_list:
        print(f"  {sqrtsNN} GeV  ", end="")
    print("  Mean MAE")
    print("-" * (col_w + 16 * len(sqrts_list) + 12))

    result: dict[str, list[float]] = {}
    for arch in all_archs:
        print(f"{LABELS[arch]:>{col_w}s}", end="")
        maes: list[float] = []
        for sqrtsNN in sqrts_list:
            d    = data_by_energy[sqrtsNN][arch]
            mask = d["is_test"]
            mae  = float(np.abs(d["b_pred"][mask] - d["b_true"][mask]).mean())
            maes.append(mae)
            print(f"  {mae:.3f} fm  ", end="")
        print(f"  {np.mean(maes):.3f} fm")
        result[arch] = maes

    print("=" * (col_w + 16 * len(sqrts_list) + 12))
    return result


def print_summary(data_by_energy: dict, all_archs: list, sqrts_list: list) -> None:
    _mae_table(data_by_energy, all_archs, sqrts_list, "MAE summary — SMASH test set")


def print_cross_transport_summary(
    smash_data: dict, urqmd_data: dict, ml_archs: list, sqrts_list: list
) -> None:
    """Side-by-side SMASH MAE vs UrQMD MAE with per-energy delta."""
    smash_maes = _mae_table(smash_data, ml_archs, sqrts_list,
                            "MAE summary — SMASH test set (ML models only)")
    urqmd_maes = _mae_table(urqmd_data, ml_archs, sqrts_list,
                            "MAE summary — UrQMD cross-transport (all events)")

    col_w = 12
    n_col = len(sqrts_list)
    width = col_w + 18 * n_col + 14
    print("\n" + "=" * width)
    print("Cross-transport MAE degradation  (UrQMD − SMASH, fm)")
    print(f"{'Model':>{col_w}s}", end="")
    for sqrtsNN in sqrts_list:
        print(f"  {sqrtsNN} GeV  ", end="")
    print("  Mean Δ")
    print("-" * width)
    for arch in ml_archs:
        print(f"{LABELS[arch]:>{col_w}s}", end="")
        deltas: list[float] = []
        for i in range(n_col):
            delta = urqmd_maes[arch][i] - smash_maes[arch][i]
            deltas.append(delta)
            sign = "+" if delta >= 0 else ""
            print(f"  {sign}{delta:.3f} fm  ", end="")
        mean_d = float(np.mean(deltas))
        sign   = "+" if mean_d >= 0 else ""
        print(f"  {sign}{mean_d:.3f} fm")
    print("=" * width)


# ── main ─────────────────────────────────────────────────────────────────────

def _parse_pred_dirs(items: list[str]) -> dict[str, Path]:
    out = {}
    for item in items:
        name, path = item.split(":", 1)
        out[name] = Path(path)
    return out


def _build_smash_data(cache: dict, truth_dir: Path, glauber_dir: Path,
                      pred_dirs: dict[str, Path], ml_archs: list[str]) -> dict:
    sqrts_list = cache["sqrts"]
    n_per_e    = cache["n_per_e"]
    cum        = np.cumsum([0] + list(n_per_e))

    data_by_energy: dict = {}
    for e_idx, sqrtsNN in enumerate(sqrts_list):
        lo, hi   = int(cum[e_idx]), int(cum[e_idx + 1])
        b_true_e = cache["b"][lo:hi]
        bins_e   = load_truth_bins(truth_dir, sqrtsNN)
        print(f"\n[SMASH] {sqrtsNN} GeV ({hi-lo:,} events)")

        data_by_energy[sqrtsNN] = {}

        # Glauber
        g = load_glauber(glauber_dir, sqrtsNN, hi - lo)
        g["b_true"]    = b_true_e
        g["cent_bins"] = bins_e
        data_by_energy[sqrtsNN]["glauber"] = g
        mae_g = float(np.abs(g["b_pred"] - b_true_e).mean())
        print(f"  {'glauber':12s} MAE={mae_g:.3f} fm")

        # ML models
        for arch in ml_archs:
            d = load_pred(pred_dirs[arch], arch, sqrtsNN)
            d["b_true"]    = b_true_e
            d["cent_bins"] = bins_e
            data_by_energy[sqrtsNN][arch] = d
            mask = d["is_test"]
            mae  = float(np.abs(d["b_pred"][mask] - b_true_e[mask]).mean())
            corr = float(np.corrcoef(d["b_pred"][mask], b_true_e[mask])[0, 1])
            print(f"  {arch:12s} MAE={mae:.3f} fm  corr={corr:.3f}")

    return data_by_energy


def _build_urqmd_data(urqmd_cache: dict, urqmd_truth_dir: Path,
                      urqmd_pred_dirs: dict[str, Path],
                      ml_archs: list[str]) -> dict:
    sqrts_list = urqmd_cache["sqrts"]
    n_per_e    = urqmd_cache["n_per_e"]
    cum        = np.cumsum([0] + list(n_per_e))

    data_by_energy: dict = {}
    for e_idx, sqrtsNN in enumerate(sqrts_list):
        lo, hi   = int(cum[e_idx]), int(cum[e_idx + 1])
        b_true_e = urqmd_cache["b"][lo:hi]
        bins_e   = load_truth_bins(urqmd_truth_dir, sqrtsNN)
        print(f"\n[UrQMD] {sqrtsNN} GeV ({hi-lo:,} events)")

        data_by_energy[sqrtsNN] = {}

        for arch in ml_archs:
            d = load_urqmd_pred(urqmd_pred_dirs[arch], arch, sqrtsNN)
            d["b_true"]    = b_true_e
            d["cent_bins"] = bins_e
            data_by_energy[sqrtsNN][arch] = d
            mae  = float(np.abs(d["b_pred"] - b_true_e).mean())
            corr = float(np.corrcoef(d["b_pred"], b_true_e)[0, 1])
            print(f"  {arch:12s} MAE={mae:.3f} fm  corr={corr:.3f}")

    return data_by_energy


def main() -> None:
    p = argparse.ArgumentParser()
    # SMASH arguments
    p.add_argument("--cache",       required=True,  type=Path)
    p.add_argument("--truth-dir",   required=True,  type=Path)
    p.add_argument("--glauber-dir", required=True,  type=Path)
    p.add_argument("--pred-dirs",   nargs="+", required=True)
    # UrQMD cross-transport arguments (all optional)
    p.add_argument("--urqmd-cache",      default=None, type=Path,
                   help="Padded UrQMD cache HDF5 (e.g. data/processed/cached/urqmd_padded.h5)")
    p.add_argument("--urqmd-truth-dir",  default=None, type=Path,
                   help="Dir with urqmd truth centrality bins (e.g. data/processed/urqmd_truth/)")
    p.add_argument("--urqmd-pred-dirs",  nargs="+", default=None,
                   help="arch:path pairs for UrQMD prediction dirs (e.g. efn:data/processed/efn/urqmd_v1)")
    # Shared
    p.add_argument("--output-dir",  required=True,  type=Path)
    p.add_argument("--n-bins",      type=int, default=7)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred_dirs = _parse_pred_dirs(args.pred_dirs)
    ml_archs  = list(pred_dirs.keys())
    all_archs = ["glauber"] + ml_archs

    # ── SMASH evaluation ──────────────────────────────────────────────────────
    print("\n━━━ SMASH evaluation ━━━")
    smash_cache = load_cache(args.cache)
    sqrts_list  = smash_cache["sqrts"]
    smash_data  = _build_smash_data(smash_cache, args.truth_dir, args.glauber_dir,
                                    pred_dirs, ml_archs)

    print("\nGenerating SMASH plots...")
    plot_scatter(smash_data, all_archs, sqrts_list, args.output_dir)
    plot_mae_profile(smash_data, all_archs, sqrts_list, args.n_bins, args.output_dir)
    plot_calibration(smash_data, ml_archs, sqrts_list, args.output_dir)
    plot_pulls(smash_data, ml_archs, sqrts_list, args.output_dir)
    print_summary(smash_data, all_archs, sqrts_list)

    # ── UrQMD cross-transport evaluation (optional) ───────────────────────────
    have_urqmd = (args.urqmd_cache is not None
                  and args.urqmd_truth_dir is not None
                  and args.urqmd_pred_dirs is not None)

    if have_urqmd:
        urqmd_pred_dirs = _parse_pred_dirs(args.urqmd_pred_dirs)
        # Only evaluate archs present in both SMASH and UrQMD pred dirs
        urqmd_archs = [a for a in ml_archs if a in urqmd_pred_dirs]
        if missing := set(ml_archs) - set(urqmd_archs):
            print(f"\n[UrQMD] Warning: no UrQMD pred dir for {missing}; skipping them.")

        print("\n━━━ UrQMD cross-transport evaluation ━━━")
        urqmd_cache = load_cache(args.urqmd_cache)
        urqmd_data  = _build_urqmd_data(urqmd_cache, args.urqmd_truth_dir,
                                        urqmd_pred_dirs, urqmd_archs)

        urqmd_out = args.output_dir / "urqmd"
        urqmd_out.mkdir(parents=True, exist_ok=True)
        print("\nGenerating UrQMD plots...")
        plot_scatter(urqmd_data, urqmd_archs, sqrts_list, urqmd_out)
        plot_mae_profile(urqmd_data, urqmd_archs, sqrts_list, args.n_bins, urqmd_out)
        plot_calibration(urqmd_data, urqmd_archs, sqrts_list, urqmd_out)
        plot_pulls(urqmd_data, urqmd_archs, sqrts_list, urqmd_out)

        # Cross-transport comparison: need SMASH data restricted to same archs
        smash_ml_only = {
            sq: {a: smash_data[sq][a] for a in urqmd_archs}
            for sq in sqrts_list
        }
        print_cross_transport_summary(smash_ml_only, urqmd_data,
                                      urqmd_archs, sqrts_list)
        print(f"\nUrQMD figures saved to {urqmd_out}/")
    else:
        if any([args.urqmd_cache, args.urqmd_truth_dir, args.urqmd_pred_dirs]):
            print("\n[UrQMD] Warning: supply all three of --urqmd-cache, "
                  "--urqmd-truth-dir, --urqmd-pred-dirs to enable UrQMD evaluation.")

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()