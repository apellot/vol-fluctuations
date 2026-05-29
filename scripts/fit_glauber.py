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
        --transport smash \\
        [--cache data/processed/cached/smash_padded_v2.h5] \\
        [--output-dir data/processed/glauber/smash] \\
        [--n-glauber 200000]

    # UrQMD variant — Glauber is re-tuned per transport, σ_NN is unchanged.
    python scripts/fit_glauber.py --transport urqmd
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
from src.data.cuts import B_MAX_FM  # noqa: E402
from src.baselines.truth import DEFAULT_BIN_EDGES, DEFAULT_BIN_LABELS, assign_bins  # noqa: E402


def centrality_table(mult_g: np.ndarray, b_g: np.ndarray, npart_g: np.ndarray) -> list[dict]:
    """⟨b⟩ and ⟨Npart⟩ per centrality class for the fitted Glauber events.

    Centrality is defined by Glauber-multiplicity percentile using the standard
    project edges (`DEFAULT_BIN_EDGES`); within each class we report the mean and
    std of the Glauber-model b and Npart. By construction these are the b/Npart a
    Glauber centrality call assigns to each class.
    """
    edges = DEFAULT_BIN_EDGES
    labels = list(DEFAULT_BIN_LABELS)
    # Same canonical centrality binning the baselines use: multiplicity-percentile
    # thresholds + assign_bins (which returns -1 for events beyond the 80% edge,
    # out of the 0–80% scope the labels cover).
    thr = np.quantile(mult_g, 1.0 - edges[1:])
    cls = assign_bins(mult_g, thr)
    rows = []
    for k, label in enumerate(labels):
        sel = cls == k
        n = int(sel.sum())
        rows.append({
            "centrality": label,
            "n_events": n,
            "mean_b": float(b_g[sel].mean()) if n else float("nan"),
            "std_b": float(b_g[sel].std()) if n else float("nan"),
            "mean_Npart": float(npart_g[sel].mean()) if n else float("nan"),
            "std_Npart": float(npart_g[sel].std()) if n else float("nan"),
        })
    return rows


def fit_one_energy(mult_smash: np.ndarray, sqrtsNN: float, out_dir: Path,
                   n_glauber: int, seed: int, fit_mult_min: int,
                   data_label: str = "data", acc_label: str = "η_lab acceptance") -> dict:
    rng = np.random.default_rng(seed)
    glauber = glauber_events(n_glauber, sqrtsNN=sqrtsNN, rng=rng)
    # NOTE: do NOT clip the Glauber sample to b<B_MAX_FM here. _chi2_for_params
    # masks bins where target == 0, so Glauber events landing below the SMASH
    # mult support are already excluded from the χ². Clipping Glauber to b<11
    # instead narrows its Npart spread, biasing the optimizer toward x→0
    # (Ncoll-dominated, too wide) and breaking the fit. Confirmed empirically:
    # the b<11 clip drove x: 1.0 → 0.0 and bulk-median residual: 4% → 24%
    # at 3.5 GeV. See conversation 2026-05-28.
    params, diag = fit_two_component_nbd(mult_smash, glauber, rng=rng,
                                         fit_mult_min=fit_mult_min)

    # Apply best-fit params to the Glauber events and generate the diagnostic comparison.
    mult_g = nbd_multiplicity(glauber["n_part"], glauber["n_coll"],
                              params.n_pp, params.k, params.x,
                              d=params.d, d_norm=params.d_norm, rng=rng)

    # Bulk residual over the FITTED region only (mult >= fit_mult_min), with both
    # distributions renormalised inside that window so the metric matches what the
    # χ² and the diagnostic plot show. "Bulk" = bins carrying ≥10 % of the
    # IN-WINDOW peak density (≥10 % of peak makes it invariant to how broad the
    # distribution is). Anchoring on the global peak instead would sit on the
    # excluded low-multiplicity spike and badly inflate the number — that was the
    # old behaviour (reported ~18 % where the true fitted-region residual is ~6 %).
    edges = diag["edges"]
    target = diag["target"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist_g, _ = np.histogram(mult_g, bins=edges)
    density_g = hist_g / max(1, hist_g.sum())
    in_window = (centers >= fit_mult_min) & (target > 0)
    if in_window.any():
        t_w = target[in_window] / target[in_window].sum()
        g_w = density_g[in_window] / max(density_g[in_window].sum(), 1e-12)
        bulk = t_w > 0.1 * float(t_w.max())
        rel_residual = (g_w[bulk] - t_w[bulk]) / t_w[bulk]
        bulk_max_rel = float(np.max(np.abs(rel_residual)))
        bulk_med_rel = float(np.median(np.abs(rel_residual)))
    else:
        bulk_max_rel = float("nan")
        bulk_med_rel = float("nan")

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

    # Diagnostic plot — data vs Glauber multiplicity over the FITTED region only.
    # Single panel (no residual sub-plot); the x-axis starts at the fit cut so the
    # excluded low-multiplicity region is not shown.
    fig, ax1 = plt.subplots(figsize=(5.5, 4.2))
    centers = 0.5 * (edges[:-1] + edges[1:])
    fit_min = diag.get("fit_mult_min", 0)
    # Renormalise both inside the fit window so the visual comparison matches the
    # χ² metric (which is computed on the inside-window-normalised distributions).
    in_window = centers >= fit_min
    t_in = target[in_window]; g_in = density_g[in_window]
    t_show = t_in / t_in.sum() if t_in.sum() > 0 else t_in
    g_show = g_in / g_in.sum() if g_in.sum() > 0 else g_in
    c_show = centers[in_window]
    ax1.step(c_show, t_show, where="mid", color="C0", lw=1.5,
             label=f"{data_label} (N = {len(mult_smash):,})")
    ax1.step(c_show, g_show, where="mid", color="C3", lw=1.5,
             label=f"Glauber+NBD  (n_pp={params.n_pp:.2f}, k={params.k:.1f}, x={params.x:.2f}, d={params.d:.2f})")
    ax1.set_yscale("log")
    ax1.set_xlim(left=fit_min)
    ax1.set_ylabel("normalised density (inside fit window)")
    ax1.set_xlabel(f"charged multiplicity ({acc_label}); fit: mult ≥ {fit_min}")
    ax1.set_title(f"Glauber NBD fit at √sNN = {sqrtsNN} GeV   χ² = {diag['chi2']:.2g}")
    ax1.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig_dir = Path("figures/baselines/glauber")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"glauber_fit_{tag}GeV.png", dpi=140)
    fig.savefig(fig_dir / f"glauber_fit_{tag}GeV.pdf")
    plt.close(fig)

    # Per-centrality-class ⟨b⟩, ⟨Npart⟩ table from the fitted Glauber events.
    cent_rows = centrality_table(mult_g, glauber["b"], glauber["n_part"])
    print(f"  centrality table (Glauber model, √sNN = {sqrtsNN} GeV):")
    print(f"    {'class':>8} {'n':>8} {'<b> fm':>8} {'σb':>6} {'<Npart>':>9} {'σNpart':>8}")
    for r in cent_rows:
        print(f"    {r['centrality']:>8} {r['n_events']:>8} {r['mean_b']:>8.2f} "
              f"{r['std_b']:>6.2f} {r['mean_Npart']:>9.1f} {r['std_Npart']:>8.1f}")

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
        "fit_mult_min": fit_min,
        "centrality_table": cent_rows,
    }


# Detector-emulated caches are the headline study (2026-05-28 restructure);
# UrQMD is the primary transport. Pass --transport smash for the secondary set.
DEFAULT_CACHE = {
    "smash": Path("data/processed/cached/smash_padded_v2_det.h5"),
    "urqmd": Path("data/processed/cached/urqmd_padded_det.h5"),
}
DEFAULT_OUTPUT_DIR = {
    "smash": Path("data/processed/glauber/smash"),
    "urqmd": Path("data/processed/glauber/urqmd"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["smash", "urqmd"], default="urqmd",
                   help="Transport model whose multiplicity distribution Glauber is fit to. "
                        "Glauber is re-tuned per transport (separate (n_pp, k, x) per energy); "
                        "σ_NN is fixed per √sNN and not affected by this flag.")
    p.add_argument("--cache", type=Path, default=None,
                   help="Padded cache built by scripts/build_padded_cache{,_urqmd}.py. "
                        "Defaults to the canonical cache for the chosen --transport.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory. Defaults to data/processed/glauber/{transport}/.")
    p.add_argument("--n-glauber", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--energy", type=float, default=None,
                   help="If set, fit only this √sNN value (e.g. 3.5). Default: fit all "
                        "energies present in the cache. Useful for quick iteration on the "
                        "fit window / parameters before committing to a full 4-energy refit.")
    p.add_argument("--fit-mult-min", type=int, default=0,
                   help="Lower edge of the multiplicity range used in the χ² fit. Default 0 "
                        "(no cut) because the cache already excludes spectator nucleons, so "
                        "there is no peripheral floor to mask. A positive value would mimic "
                        "the trigger-inefficiency cut a real experiment uses (Kimelman thesis).")
    args = p.parse_args()

    if args.cache is None:
        args.cache = DEFAULT_CACHE[args.transport]
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR[args.transport]
    print(f"transport={args.transport}  cache={args.cache}  output_dir={args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.cache, "r") as h:
        mult = h["mult_lab"][:]
        sqrts_per_energy = list(h.attrs["sqrtsNN_per_energy"])
        n_per_energy = list(h.attrs["n_events_per_energy"])
        # Plot labels reflect the actual data, not a hardcoded "SMASH"/"η<2".
        data_label = str(h.attrs.get("generator", args.transport.upper()))
        if bool(h.attrs.get("detector_emulation", False)):
            acc_label = (f"{h.attrs['detector_eta_min']:g} < η_lab < "
                         f"{h.attrs['detector_eta_max']:g}, detector")
        else:
            acc_label = "0 < η_lab, truth"

    cum = np.cumsum([0] + n_per_energy)
    summary = {}
    for e_id, (sqrtsNN, n_e) in enumerate(zip(sqrts_per_energy, n_per_energy)):
        if args.energy is not None and not np.isclose(sqrtsNN, args.energy):
            continue
        lo, hi = int(cum[e_id]), int(cum[e_id + 1])
        s = fit_one_energy(mult[lo:hi], sqrtsNN, args.output_dir,
                           args.n_glauber, args.seed, args.fit_mult_min,
                           data_label=data_label, acc_label=acc_label)
        tag = f"{sqrtsNN:.1f}".replace(".", "p")
        summary[f"auau_{tag}GeV"] = s
    # When fitting a single energy on the side, don't clobber the full 4-energy
    # summary file — write to a per-energy sidecar so existing summary survives.
    if args.energy is not None:
        tag = f"{args.energy:.1f}".replace(".", "p")
        out = args.output_dir / f"fit_summary_{tag}GeV.json"
    else:
        out = args.output_dir / "fit_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
