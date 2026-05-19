"""Day-1 sanity plots for ingested SMASH HDF5 files.

Two modes:
  * Single-file: detailed per-energy plots with scatter + profile.
  * Multi-file: overlaid cross-energy comparison plots (one curve per energy).

Both modes produce the same four panels:
  1. b distribution                     — should be triangular for min-bias.
  2. Charged-mult |η|<0.5 vs b          — RefMult-style observable.
  3. N_part vs b                        — geometry; corr(N_part,b) ≈ −0.94.
  4. ⟨pT⟩(charged, |η|<1.0) vs N_ch     — mild rise; mean ⟨pT⟩ grows weakly with √s.

Usage:
    # single-file detailed plots
    python scripts/sanity_plots.py --inputs data/processed/auau_3p5GeV.h5 \\
                                   --output-dir figures/sanity/3p5GeV

    # cross-energy overlay
    python scripts/sanity_plots.py \\
        --inputs data/processed/auau_3p2GeV.h5 data/processed/auau_3p5GeV.h5 \\
                 data/processed/auau_3p9GeV.h5 data/processed/auau_4p5GeV.h5 \\
        --output-dir figures/sanity/compare
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class EventSummary:
    """Per-event arrays plus the energy label, loaded once and reused by both modes."""
    sqrtsNN: float
    n_events: int
    b: np.ndarray
    npart: np.ndarray
    mult05: np.ndarray
    nparticles: np.ndarray
    mean_pt_charged: np.ndarray  # ⟨pT⟩ over charged particles in |η|<1.0; NaN if no charged particle


def _profile(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and std of y in bins of x. Returns (centers, mean, std). NaN-tolerant."""
    centers = 0.5 * (bins[:-1] + bins[1:])
    n_bins = len(bins) - 1
    mean = np.full(n_bins, np.nan)
    std = np.full(n_bins, np.nan)
    # Mask NaNs out of y so they don't poison empty-bin statistics.
    finite = np.isfinite(y)
    x_f, y_f = x[finite], y[finite]
    idx = np.digitize(x_f, bins) - 1
    for k in range(n_bins):
        sel = idx == k
        if sel.sum() > 0:
            mean[k] = y_f[sel].mean()
            std[k] = y_f[sel].std()
    return centers, mean, std


def _vectorized_mean_pt(path: Path) -> tuple[np.ndarray, int, float]:
    """Compute per-event ⟨pT⟩ over charged particles in |η|<1.0.

    Vectorized via np.add.reduceat — far faster than a Python per-event loop
    once event counts exceed ~1e4.
    """
    with h5py.File(path, "r") as h:
        offsets = h["offset"][:].astype(np.int64)
        pt = h["particles/pT"][:]
        eta = h["particles/eta"][:]
        charge = h["particles/charge"][:]
        n_events = int(h.attrs["n_events"])
        sqrtsNN = float(h.attrs["sqrtsNN"])

    # In-acceptance flag per particle: charged AND mid-rapidity.
    in_acc = (np.abs(eta) < 1.0) & (charge != 0)
    pt_in = pt * in_acc.astype(np.float32)
    w_in = in_acc.astype(np.float32)

    # reduceat([starts]) sums each slice [starts[i] : starts[i+1]]; offsets[:-1] gives the right starts.
    sum_pt = np.add.reduceat(pt_in, offsets[:-1])
    sum_w = np.add.reduceat(w_in, offsets[:-1])

    mean_pt = np.where(sum_w > 0, sum_pt / np.maximum(sum_w, 1e-12), np.nan).astype(np.float32)
    return mean_pt, n_events, sqrtsNN


def load_summary(path: Path) -> EventSummary:
    """Read one HDF5 file into the per-event arrays needed for all four panels."""
    mean_pt_charged, n_events, sqrtsNN = _vectorized_mean_pt(path)
    with h5py.File(path, "r") as h:
        return EventSummary(
            sqrtsNN=sqrtsNN,
            n_events=n_events,
            b=h["b"][:],
            npart=h["Npart"][:],
            mult05=h["mult_eta05"][:],
            nparticles=h["nparticles"][:],
            mean_pt_charged=mean_pt_charged,
        )


# ---------------------------------------------------------------------------
# Single-file detailed plots (the original behaviour).
# ---------------------------------------------------------------------------

def make_plots(summary: EventSummary, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    title_suffix = f"√sNN = {summary.sqrtsNN} GeV, N = {summary.n_events}"

    # Panel 1 — b distribution.
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.hist(summary.b, bins=40, color="C0", edgecolor="k", linewidth=0.4)
    ax.set_xlabel("impact parameter b (fm)")
    ax.set_ylabel("events / bin")
    ax.set_title(f"b distribution  ({title_suffix})")
    fig.tight_layout()
    fig.savefig(out_dir / "01_b_distribution.png", dpi=140)
    fig.savefig(out_dir / "01_b_distribution.pdf")
    plt.close(fig)

    bins = np.linspace(0, summary.b.max(), 30)

    # Panel 2 — charged mult vs b (CM-frame |η|<0.5).
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.scatter(summary.b, summary.mult05, s=2, alpha=0.25, color="C0", rasterized=True)
    cx, m, s = _profile(summary.b, summary.mult05.astype(float), bins)
    ax.errorbar(cx, m, yerr=s, fmt="o", color="C3", ms=3, lw=1, label="profile ± 1σ")
    ax.set_xlabel("impact parameter b (fm)")
    ax.set_ylabel(r"charged multiplicity $|\eta|<0.5$  (CM frame)")
    ax.set_title(f"charged mult vs b  ({title_suffix})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "02_mult_vs_b.png", dpi=140)
    fig.savefig(out_dir / "02_mult_vs_b.pdf")
    plt.close(fig)

    # Panel 3 — N_part vs b. Sanity headline.
    corr = float(np.corrcoef(summary.b, summary.npart)[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.scatter(summary.b, summary.npart, s=2, alpha=0.25, color="C0", rasterized=True)
    cx, m, s = _profile(summary.b, summary.npart.astype(float), bins)
    ax.errorbar(cx, m, yerr=s, fmt="o", color="C3", ms=3, lw=1, label="profile ± 1σ")
    ax.set_xlabel("impact parameter b (fm)")
    ax.set_ylabel(r"$N_{\rm part}$  (derived as $2A - N_{\rm spec}$)")
    ax.set_title(f"N_part vs b   corr = {corr:.3f}   ({title_suffix})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "03_Npart_vs_b.png", dpi=140)
    fig.savefig(out_dir / "03_Npart_vs_b.pdf")
    plt.close(fig)

    # Panel 4 — ⟨pT⟩ vs N_ch.
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.scatter(summary.mult05, summary.mean_pt_charged, s=2, alpha=0.25, color="C0", rasterized=True)
    mbins = np.linspace(summary.mult05.min(), summary.mult05.max(), 30)
    cx, m, s = _profile(summary.mult05.astype(float), summary.mean_pt_charged.astype(float), mbins)
    ax.errorbar(cx, m, yerr=s, fmt="o", color="C3", ms=3, lw=1, label="profile ± 1σ")
    ax.set_xlabel(r"charged multiplicity $|\eta| < 0.5$")
    ax.set_ylabel(r"$\langle p_T \rangle$  (charged, $|\eta| < 1.0$) [GeV]")
    ax.set_title(f"⟨pT⟩ vs N_ch   ({title_suffix})")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "04_meanPt_vs_mult.png", dpi=140)
    fig.savefig(out_dir / "04_meanPt_vs_mult.pdf")
    plt.close(fig)

    print(f"  {summary.sqrtsNN} GeV → {out_dir}   corr(N_part, b) = {corr:.3f}")


# ---------------------------------------------------------------------------
# Multi-energy overlay plots.
# ---------------------------------------------------------------------------

def make_overlay_plots(summaries: list[EventSummary], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = sorted(summaries, key=lambda s: s.sqrtsNN)
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    # Common b binning across all energies.
    b_max = max(float(s.b.max()) for s in summaries)
    bbins = np.linspace(0, b_max, 30)

    # Panel 1 — b distributions, normalized so the y axis is dN/db / N_events.
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for i, s in enumerate(summaries):
        ax.hist(s.b, bins=40, density=True, histtype="step", linewidth=1.5,
                color=colors[i], label=f"√sNN = {s.sqrtsNN} GeV  (N={s.n_events//1000}k)")
    ax.set_xlabel("impact parameter b (fm)")
    ax.set_ylabel("normalized density")
    ax.set_title("b distribution — all energies")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "01_b_distribution.png", dpi=140)
    fig.savefig(out_dir / "01_b_distribution.pdf")
    plt.close(fig)

    # Panel 2 — charged mult vs b, profile lines only (no scatter — too dense).
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for i, s in enumerate(summaries):
        cx, m, sd = _profile(s.b, s.mult05.astype(float), bbins)
        ax.errorbar(cx, m, yerr=sd, fmt="o-", color=colors[i], ms=3, lw=1,
                    label=f"√sNN = {s.sqrtsNN} GeV")
    ax.set_xlabel("impact parameter b (fm)")
    ax.set_ylabel(r"charged multiplicity $|\eta|<0.5$  (CM frame)")
    ax.set_title("charged mult vs b — all energies")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "02_mult_vs_b.png", dpi=140)
    fig.savefig(out_dir / "02_mult_vs_b.pdf")
    plt.close(fig)

    # Panel 3 — N_part vs b. Geometry-only; all energies should superimpose.
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for i, s in enumerate(summaries):
        corr = float(np.corrcoef(s.b, s.npart)[0, 1])
        cx, m, sd = _profile(s.b, s.npart.astype(float), bbins)
        ax.errorbar(cx, m, yerr=sd, fmt="o-", color=colors[i], ms=3, lw=1,
                    label=f"√sNN = {s.sqrtsNN} GeV  (corr = {corr:.3f})")
    ax.set_xlabel("impact parameter b (fm)")
    ax.set_ylabel(r"$N_{\rm part}$  (derived as $2A - N_{\rm spec}$)")
    ax.set_title("N_part vs b — all energies")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "03_Npart_vs_b.png", dpi=140)
    fig.savefig(out_dir / "03_Npart_vs_b.pdf")
    plt.close(fig)

    # Panel 4 — ⟨pT⟩ vs N_ch. Each energy has its own multiplicity range, so plot independently.
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for i, s in enumerate(summaries):
        mbins = np.linspace(0, max(1, s.mult05.max()), 30)
        cx, m, sd = _profile(s.mult05.astype(float), s.mean_pt_charged.astype(float), mbins)
        ax.errorbar(cx, m, yerr=sd, fmt="o-", color=colors[i], ms=3, lw=1,
                    label=f"√sNN = {s.sqrtsNN} GeV")
    ax.set_xlabel(r"charged multiplicity $|\eta| < 0.5$")
    ax.set_ylabel(r"$\langle p_T \rangle$  (charged, $|\eta| < 1.0$) [GeV]")
    ax.set_title("⟨pT⟩ vs N_ch — all energies")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "04_meanPt_vs_mult.png", dpi=140)
    fig.savefig(out_dir / "04_meanPt_vs_mult.pdf")
    plt.close(fig)

    print(f"Overlay → {out_dir}  ({len(summaries)} energies)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True, nargs="+", type=Path, help="One or more ingested HDF5 files")
    p.add_argument("--output-dir", required=True, type=Path, help="Root output directory")
    args = p.parse_args()
    inputs = [pp.expanduser() for pp in args.inputs]
    out_dir = args.output_dir.expanduser()

    summaries = [load_summary(pp) for pp in inputs]

    if len(summaries) == 1:
        make_plots(summaries[0], out_dir)
    else:
        # Per-energy detailed plots in sub-dirs, plus the cross-energy overlay.
        for s, pp in zip(summaries, inputs):
            tag = f"{s.sqrtsNN:.1f}".replace(".", "p")
            make_plots(s, out_dir / f"{tag}GeV")
        make_overlay_plots(summaries, out_dir / "compare")


if __name__ == "__main__":
    main()
