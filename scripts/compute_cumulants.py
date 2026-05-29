"""Compute net-proton cumulants C1-C4 and ratios per centrality bin for each
centrality method and compare to the truth-b oracle.

This is the headline physics figure for the paper — it shows whether the ML
centrality improvement actually matters for the QCD critical point search via
BES-II net-proton cumulants.

Method:
    For each centrality method (truth, glauber, efn, pfn, deepsets, gnn):
        1. Bin events by predicted centrality_bin
        2. Within each bin, compute net-proton cumulants C1-C4 from mult_lab
           (we use mult_lab as a proxy for net-proton number since the cache
           doesn't store per-particle PID — a note for the paper)
        3. Compute ratios C2/C1, C3/C2, C4/C2
        4. Compare to oracle (truth-b binning)

Primary study is UrQMD + detector (2026-05-28 restructure).

Usage:
    python scripts/compute_cumulants.py \\
        --cache data/processed/cached/urqmd_padded_det.h5 \\
        --truth-dir data/processed/truth/urqmd/ \\
        --glauber-dir data/processed/glauber/urqmd/ \\
        --pred-dirs efn:data/processed/efn/urqmd_v1 \\
                    pfn:data/processed/pfn/urqmd_v1 \\
                    deepsets:data/processed/deepsets/urqmd_v1 \\
                    gnn:data/processed/gnn/urqmd_v1 \\
        --output-dir figures/cumulants/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── cumulant math ────────────────────────────────────────────────────────────

def cumulants(x: np.ndarray) -> dict[str, float]:
    """Compute C1-C4 (cumulants) from a sample x.
    C1 = mean, C2 = variance, C3 = third cumulant, C4 = fourth cumulant.
    Returns dict with C1,C2,C3,C4 and ratios C2/C1, C3/C2, C4/C2.
    """
    n = len(x)
    if n < 4:
        return {k: np.nan for k in ["C1","C2","C3","C4","C2/C1","C3/C2","C4/C2","n"]}
    x = x.astype(np.float64)
    m1 = np.mean(x)
    dx = x - m1
    m2 = np.mean(dx**2)
    m3 = np.mean(dx**3)
    m4 = np.mean(dx**4)
    C1 = m1
    C2 = m2
    C3 = m3
    C4 = m4 - 3 * m2**2
    return {
        "C1": C1, "C2": C2, "C3": C3, "C4": C4,
        "C2/C1": C2 / C1 if C1 != 0 else np.nan,
        "C3/C2": C3 / C2 if C2 != 0 else np.nan,
        "C4/C2": C4 / C2 if C2 != 0 else np.nan,
        "n": n,
    }


def bootstrap_cumulants(x: np.ndarray, n_boot: int = 200, seed: int = 0) -> dict[str, float]:
    """Bootstrap uncertainty on cumulant ratios."""
    rng = np.random.default_rng(seed)
    results = {k: [] for k in ["C1","C2","C3","C4","C2/C1","C3/C2","C4/C2"]}
    for _ in range(n_boot):
        sample = rng.choice(x, size=len(x), replace=True)
        c = cumulants(sample)
        for k in results:
            results[k].append(c[k])
    return {k: float(np.nanstd(v)) for k, v in results.items()}


# ── data loading ─────────────────────────────────────────────────────────────

def load_cache(cache_path: Path) -> dict:
    with h5py.File(cache_path, "r") as h:
        return {
            "mult":      h["mult_lab"][:].astype(np.float32),
            "b":         h["b"][:].astype(np.float32),
            "energy_id": h["energy_id"][:].astype(np.int8),
            "sqrts":     list(h.attrs["sqrtsNN_per_energy"]),
            "n_per_e":   list(h.attrs["n_events_per_energy"]),
        }


def load_centrality_bins(method: str, energy_tag: str,
                         truth_dir: Path, glauber_dir: Path,
                         pred_dirs: dict[str, Path],
                         n_events: int) -> np.ndarray:
    """Load centrality_bin array for a given method and energy."""
    if method == "truth":
        tag_p = energy_tag.replace(".", "p")
        p = truth_dir / f"truth_auau_{tag_p}GeV.h5"
        with h5py.File(p, "r") as h:
            return h["centrality_bin"][:].astype(np.int16)
    elif method == "glauber":
        p = glauber_dir / f"glauber_pred_auau_{tag_p}GeV.h5"
        with h5py.File(p, "r") as h:
            return h["centrality_bin"][:].astype(np.int16)
    else:
        base = pred_dirs[method]
        # find the right file for this energy
        tag = energy_tag.replace(".", "p")
        p = base / f"{method}_pred_auau_{tag}GeV.h5"
        with h5py.File(p, "r") as h:
            return h["centrality_bin"][:].astype(np.int16)


# ── per-energy cumulant computation ──────────────────────────────────────────

def compute_for_energy(mult_e: np.ndarray, bins_dict: dict[str, np.ndarray],
                       n_bins: int, n_boot: int) -> dict:
    """For one energy slice, compute cumulants per centrality bin for each method."""
    results = {}
    for method, bins in bins_dict.items():
        per_bin = []
        for b in range(n_bins):
            sel = bins == b
            x = mult_e[sel]
            c = cumulants(x)
            err = bootstrap_cumulants(x, n_boot=n_boot) if len(x) >= 10 else \
                  {k: np.nan for k in ["C1","C2","C3","C4","C2/C1","C3/C2","C4/C2"]}
            per_bin.append({"vals": c, "errs": err})
        results[method] = per_bin
    return results


# ── plotting ─────────────────────────────────────────────────────────────────

COLORS = {
    "truth":    "black",
    "glauber":  "steelblue",
    "efn":      "crimson",
    "pfn":      "darkorange",
    "deepsets": "mediumseagreen",
    "gnn":      "mediumpurple",
}

LABELS = {
    "truth":    "Truth-b oracle",
    "glauber":  "Glauber-NBD",
    "efn":      "EFN",
    "pfn":      "PFN",
    "deepsets": "DeepSets",
    "gnn":      "GNN",
}


def plot_ratios(results_by_energy: dict, methods: list[str],
                sqrts_list: list[float], n_bins: int, output_dir: Path) -> None:
    """Plot C2/C1, C3/C2, C4/C2 vs centrality bin for each energy."""
    ratios = ["C2/C1", "C3/C2", "C4/C2"]
    ratio_labels = [r"$C_2/C_1$", r"$C_3/C_2$", r"$C_4/C_2$"]
    bin_centers = np.arange(n_bins)
    centrality_labels = [f"{int(100*i/n_bins)}–{int(100*(i+1)/n_bins)}%" 
                         for i in range(n_bins)]

    for e_idx, sqrtsNN in enumerate(sqrts_list):
        tag = f"{sqrtsNN:.1f}".replace(".", "p")
        res = results_by_energy[tag]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Net-proton cumulant ratios — Au+Au √sNN = {sqrtsNN} GeV", fontsize=13)

        for ax, ratio, rlabel in zip(axes, ratios, ratio_labels):
            for method in methods:
                vals = [res[method][b]["vals"][ratio] for b in range(n_bins)]
                errs = [res[method][b]["errs"][ratio] for b in range(n_bins)]
                vals = np.array(vals, dtype=float)
                errs = np.array(errs, dtype=float)

                lw = 2.5 if method == "truth" else 1.5
                ls = "-" if method == "truth" else "--"
                ax.errorbar(bin_centers, vals, yerr=errs,
                            label=LABELS.get(method, method),
                            color=COLORS.get(method, "gray"),
                            lw=lw, ls=ls, marker="o", markersize=4, capsize=3)

            ax.set_xlabel("Centrality bin (0=central)", fontsize=11)
            ax.set_ylabel(rlabel, fontsize=12)
            ax.set_xticks(bin_centers)
            ax.set_xticklabels(centrality_labels, rotation=45, fontsize=8)
            ax.axhline(0, color="gray", lw=0.5, ls=":")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = output_dir / f"cumulant_ratios_{tag}GeV.pdf"
        fig.savefig(out, bbox_inches="tight")
        out_png = output_dir / f"cumulant_ratios_{tag}GeV.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


def plot_c1(results_by_energy: dict, methods: list[str],
            sqrts_list: list[float], n_bins: int, output_dir: Path) -> None:
    """Plot C1 (mean multiplicity) vs centrality bin — sanity check."""
    bin_centers = np.arange(n_bins)
    fig, axes = plt.subplots(1, len(sqrts_list), figsize=(5*len(sqrts_list), 4), sharey=False)
    if len(sqrts_list) == 1:
        axes = [axes]
    fig.suptitle("Mean multiplicity (C1) per centrality bin", fontsize=12)

    for ax, sqrtsNN in zip(axes, sqrts_list):
        tag = f"{sqrtsNN:.1f}".replace(".", "p")
        res = results_by_energy[tag]
        for method in methods:
            vals = [res[method][b]["vals"]["C1"] for b in range(n_bins)]
            ax.plot(bin_centers, vals, label=LABELS.get(method, method),
                    color=COLORS.get(method, "gray"),
                    lw=2 if method == "truth" else 1.5,
                    ls="-" if method == "truth" else "--",
                    marker="o", markersize=4)
        ax.set_title(f"√sNN = {sqrtsNN} GeV")
        ax.set_xlabel("Centrality bin")
        ax.set_ylabel(r"$\langle N_{ch} \rangle$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / "c1_mean_mult.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(output_dir / "c1_mean_mult.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--truth-dir", required=True, type=Path)
    p.add_argument("--glauber-dir", required=True, type=Path)
    p.add_argument("--pred-dirs", nargs="+", default=[],
                   help="name:path pairs e.g. efn:data/processed/efn/baseline_v1")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--n-bins", type=int, default=7)
    p.add_argument("--n-boot", type=int, default=200)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parse pred-dirs
    pred_dirs = {}
    for item in args.pred_dirs:
        name, path = item.split(":", 1)
        pred_dirs[name] = Path(path)

    methods = ["truth", "glauber"] + list(pred_dirs.keys())
    print(f"Methods: {methods}")

    # Load cache
    cache = load_cache(args.cache)
    sqrts_list = cache["sqrts"]
    n_per_e = cache["n_per_e"]
    cum = np.cumsum([0] + list(n_per_e))

    results_by_energy = {}
    for e_idx, sqrtsNN in enumerate(sqrts_list):
        lo, hi = int(cum[e_idx]), int(cum[e_idx + 1])
        mult_e = cache["mult"][lo:hi]
        tag = f"{sqrtsNN:.1f}".replace(".", "p")
        energy_tag = f"{sqrtsNN:.1f}"

        print(f"\n{sqrtsNN} GeV ({hi-lo:,} events)")

        # Load centrality bins for each method
        bins_dict = {}
        for method in methods:
            try:
                bins_dict[method] = load_centrality_bins(
                    method, energy_tag,
                    args.truth_dir, args.glauber_dir, pred_dirs,
                    hi - lo,
                )
            except Exception as e:
                print(f"  WARNING: could not load {method} for {sqrtsNN} GeV: {e}")

        res = compute_for_energy(mult_e, bins_dict, args.n_bins, args.n_boot)
        results_by_energy[tag] = res

        # Print C2/C1 per bin for each method
        for method in bins_dict:
            vals = [f"{res[method][b]['vals']['C2/C1']:.3f}" for b in range(args.n_bins)]
            print(f"  {method:12s} C2/C1: {vals}")

    # Plots
    print("\nGenerating plots...")
    plot_ratios(results_by_energy, list(bins_dict.keys()), sqrts_list, args.n_bins, args.output_dir)
    plot_c1(results_by_energy, list(bins_dict.keys()), sqrts_list, args.n_bins, args.output_dir)

    # Save JSON summary
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    with open(args.output_dir / "cumulants.json", "w") as f:
        json.dump(_clean(results_by_energy), f, indent=2)
    print(f"\nWrote summary to {args.output_dir}/cumulants.json")
    print(f"Figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()