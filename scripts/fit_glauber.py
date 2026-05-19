"""Fit two-component NBD Glauber to SMASH multiplicity at each energy.

For each energy:
  1. Generate N_GLAUBER Glauber-MC events (b sampled ∝ b on [0, b_max]).
  2. Grid-search (n_pp, k, x) to minimize χ² between the Glauber-sampled
     multiplicity distribution and the SMASH multiplicity distribution
     (observable: mult_eta05).
  3. Save fitted parameters + a diagnostic comparison plot.

Acceptance criterion (from CLAUDE.md): residuals below ~5 % across the bulk of
the multiplicity distribution. The strict mid-rapidity observable has a large
zero-multiplicity floor at FXT (≈47 % of events); we fit on the non-zero tail
where multiplicity is informative, then document the floor explicitly.

Usage:
    python scripts/fit_glauber.py \\
        --inputs data/processed/auau_{3p2,3p5,3p9,4p5}GeV.h5 \\
        --output-dir data/processed/glauber \\
        [--n-glauber 200000]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines.glauber import (  # noqa: E402
    SIGMA_NN_MB, GlauberParams,
    glauber_events, nbd_multiplicity, fit_two_component_nbd,
)


def fit_one_energy(mult_smash: np.ndarray, sqrtsNN: float, out_dir: Path,
                   n_glauber: int, seed: int, fit_mult_min: int) -> dict:
    rng = np.random.default_rng(seed)
    glauber = glauber_events(n_glauber, sqrtsNN=sqrtsNN, rng=rng)
    params, diag = fit_two_component_nbd(mult_smash, glauber, rng=rng,
                                         fit_mult_min=fit_mult_min)

    # Apply best-fit params to the Glauber events and generate the diagnostic comparison.
    mult_g = nbd_multiplicity(glauber["n_part"], glauber["n_coll"],
                              params.n_pp, params.k, params.x,
                              d=params.d, d_norm=params.d_norm, rng=rng)

    # Compare distributions over the well-populated region. Defining "bulk" as bins
    # carrying ≥1 % of the density excludes the lowest-stats tail bins where a single
    # statistical fluctuation can dominate the max-residual metric. We also report
    # the median absolute residual so a single outlier bin is not the only signal.
    edges = diag["edges"]
    target = diag["target"]
    hist_g, _ = np.histogram(mult_g, bins=edges)
    density_g = hist_g / max(1, hist_g.sum())
    mask = target > 0.01
    rel_residual = (density_g[mask] - target[mask]) / target[mask]
    bulk_max_rel = float(np.max(np.abs(rel_residual)))
    bulk_med_rel = float(np.median(np.abs(rel_residual)))

    # Save params + diagnostics.
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{sqrtsNN:.1f}".replace(".", "p")
    with open(out_dir / f"glauber_params_{tag}GeV.json", "w") as f:
        json.dump(asdict(params), f, indent=2)

    # Also save the Glauber event set for use by run_glauber.py — avoids regenerating.
    with h5py.File(out_dir / f"glauber_events_{tag}GeV.h5", "w") as h:
        h.create_dataset("b", data=glauber["b"])
        h.create_dataset("n_part", data=glauber["n_part"])
        h.create_dataset("n_coll", data=glauber["n_coll"])
        h.create_dataset("mult", data=mult_g)
        h.attrs["sqrtsNN"] = sqrtsNN
        h.attrs["sigma_NN_mb"] = params.sigma_NN
        h.attrs["n_pp"] = params.n_pp
        h.attrs["k"] = params.k
        h.attrs["x"] = params.x
        h.attrs["d"] = params.d
        h.attrs["d_norm"] = params.d_norm
        h.attrs["n_events"] = n_glauber
        h.attrs["seed"] = seed

    # Diagnostic plot — SMASH vs Glauber multiplicity distributions.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    centers = 0.5 * (edges[:-1] + edges[1:])
    fit_min = diag.get("fit_mult_min", 0)
    # Renormalise both inside the fit window so the visual comparison matches the
    # χ² metric (which is computed on the inside-window-normalised distributions).
    in_window = centers >= fit_min
    t_in = target[in_window]; g_in = density_g[in_window]
    if t_in.sum() > 0:
        t_show = target.copy(); t_show[in_window] = t_in / t_in.sum()
    else:
        t_show = target
    if g_in.sum() > 0:
        g_show = density_g.copy(); g_show[in_window] = g_in / g_in.sum()
    else:
        g_show = density_g
    ax1.step(centers, t_show, where="mid", color="C0", lw=1.5,
             label=f"SMASH (N = {len(mult_smash):,})")
    ax1.step(centers, g_show, where="mid", color="C3", lw=1.5,
             label=f"Glauber+NBD  (n_pp={params.n_pp:.2f}, k={params.k:.1f}, x={params.x:.2f}, d={params.d:.2f})")
    if fit_min > 0:
        ax1.axvspan(0, fit_min, color="gray", alpha=0.15, label=f"excluded (mult < {fit_min})")
    ax1.set_yscale("log")
    ax1.set_ylabel("normalised density (inside fit window)")
    ax1.set_title(f"Glauber NBD fit at √sNN = {sqrtsNN} GeV   χ² = {diag['chi2']:.2g}")
    ax1.legend(loc="upper right", fontsize=8)

    rel = np.where(target > 0, (density_g - target) / np.maximum(target, 1e-12), np.nan)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.fill_between(centers, -0.05, 0.05, color="gray", alpha=0.15, label="±5% acceptance")
    ax2.step(centers, rel, where="mid", color="C3")
    ax2.set_xlabel("charged multiplicity 0 < η_lab < 2")
    ax2.set_ylabel("(Gl − SMASH)/SMASH")
    ax2.set_ylim(-1, 1)
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig_dir = Path("figures/baselines/glauber")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"glauber_fit_{tag}GeV.png", dpi=140)
    fig.savefig(fig_dir / f"glauber_fit_{tag}GeV.pdf")
    plt.close(fig)

    print(f"  {sqrtsNN} GeV: n_pp={params.n_pp:.3f}, k={params.k:.2f}, x={params.x:.2f}, "
          f"d={params.d:.3f}, χ²={diag['chi2']:.3g}, bulk med-|rel|={bulk_med_rel:.1%}, max={bulk_max_rel:.1%}")
    return {
        "sqrtsNN": sqrtsNN,
        "n_pp": params.n_pp,
        "k": params.k,
        "x": params.x,
        "d": params.d,
        "d_norm": params.d_norm,
        "sigma_NN_mb": params.sigma_NN,
        "chi2": diag["chi2"],
        "bulk_max_rel_residual": bulk_max_rel,
        "bulk_med_rel_residual": bulk_med_rel,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, type=Path,
                   help="Lab-frame cache built by scripts/build_padded_cache.py")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--n-glauber", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fit-mult-min", type=int, default=0,
                   help="Lower edge of the multiplicity range used in the χ² fit. Default 0 "
                        "(no cut) because the cache already excludes spectator nucleons, so "
                        "there is no peripheral floor to mask. A positive value would mimic "
                        "the trigger-inefficiency cut a real experiment uses (Kimelman thesis).")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.cache, "r") as h:
        mult = h["mult_lab"][:]
        sqrts_per_energy = list(h.attrs["sqrtsNN_per_energy"])
        n_per_energy = list(h.attrs["n_events_per_energy"])

    cum = np.cumsum([0] + n_per_energy)
    summary = {}
    for e_id, (sqrtsNN, n_e) in enumerate(zip(sqrts_per_energy, n_per_energy)):
        lo, hi = int(cum[e_id]), int(cum[e_id + 1])
        s = fit_one_energy(mult[lo:hi], sqrtsNN, args.output_dir,
                           args.n_glauber, args.seed, args.fit_mult_min)
        tag = f"{sqrtsNN:.1f}".replace(".", "p")
        summary[f"auau_{tag}GeV"] = s
    with open(args.output_dir / "fit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.output_dir / 'fit_summary.json'}")


if __name__ == "__main__":
    main()
