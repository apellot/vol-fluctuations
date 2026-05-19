"""Event-level scalar feature extraction for the MLP baseline.

Loads an ingested HDF5 file and produces a per-event feature matrix using
observables a real experiment could plausibly measure. **Excluded by design**:
N_part (derived from spectator information no detector has) and `nparticles`
total counts (dominated by the ~158 spectator protons SMASH dumps as final-state
particles, which would be flagged as data leakage by physics-reviewer).

Features (current set; the MLP config's n_features must match):
    1. sqrtsNN                                   — energy label
    2. mult_eta05                                — charged mult in CM-frame |η| < 0.5
    3. mult_eta10                                — charged mult in |η| < 1.0
    4. mult_eta20                                — charged mult in |η| < 2.0 (computed at load)
    5. n_proton                                  — proton count (final-state, no feed-down)
    6. n_antiproton                              — antiproton count
    7. mean_pt_charged_eta10                     — ⟨pT⟩ over charged particles in |η|<1.0
    8. n_pion_charged_eta10                      — charged pion count |η|<1.0
    9. n_kaon_charged_eta10                      — charged kaon count |η|<1.0
   10. p_over_pi_eta10                           — proton/pion ratio at mid-rapidity

The proton/pion ratio is a composition signature that varies with centrality
(more baryon stopping at central) and with energy. Useful because it adds
information orthogonal to the bulk multiplicity counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

# Feature names in the order they appear in the output matrix.
FEATURE_NAMES = [
    "sqrtsNN",
    "mult_eta05",
    "mult_eta10",
    "mult_eta20",
    "n_proton",
    "n_antiproton",
    "mean_pt_charged_eta10",
    "n_pion_charged_eta10",
    "n_kaon_charged_eta10",
    "p_over_pi_eta10",
]
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class EventDataset:
    """Per-event arrays for one energy file.  Light wrapper so callers can keep
    the bookkeeping straightforward."""
    features: np.ndarray         # (n_events, n_features) float32
    b: np.ndarray                # (n_events,) float32  — truth impact parameter
    centrality_bin: np.ndarray   # (n_events,) int16    — bin from the truth-tuned baseline
    sqrtsNN: float
    n_events: int
    feature_names: list[str]


# PDG codes used by the composition features.
PDG_PROTON = 2212
PDG_PION_PLUS = 211
PDG_KAON_PLUS = 321


def _per_event_reduce(values: np.ndarray, mask: np.ndarray,
                      offsets: np.ndarray, *, op: str) -> np.ndarray:
    """Sum or mean of `values` within `mask` per event, using offsets to delimit slices.

    Vectorised via np.add.reduceat so it scales to the 100k-event files without a
    Python per-event loop. Returns NaN for events with no in-mask particles when
    op='mean'.
    """
    if op == "sum":
        return np.add.reduceat((values * mask).astype(np.float32), offsets[:-1])
    if op == "mean":
        v_w = (values * mask).astype(np.float32)
        w = mask.astype(np.float32)
        sum_v = np.add.reduceat(v_w, offsets[:-1])
        sum_w = np.add.reduceat(w, offsets[:-1])
        return np.where(sum_w > 0, sum_v / np.maximum(sum_w, 1e-12), 0.0).astype(np.float32)
    raise ValueError(f"Unknown op: {op}")


def load_features(h5_path: Path, baseline_pred_h5: Path) -> EventDataset:
    """Build the per-event feature matrix.  Centrality bin labels come from the
    truth-tuned baseline; the MLP's classification head is trained against these.
    """
    with h5py.File(h5_path, "r") as h:
        sqrtsNN = float(h.attrs["sqrtsNN"])
        n_events = int(h.attrs["n_events"])
        b = h["b"][:]
        mult05 = h["mult_eta05"][:]
        mult10 = h["mult_eta10"][:]
        n_proton = h["n_proton"][:]
        n_antiproton = h["n_antiproton"][:]
        offsets = h["offset"][:].astype(np.int64)
        pdg = h["particles/pdg"][:]
        pt = h["particles/pT"][:]
        eta = h["particles/eta"][:]
        charge = h["particles/charge"][:]

    # Particle-level masks for the load-time-derived features.
    in_eta20 = (np.abs(eta) < 2.0)
    in_eta10 = (np.abs(eta) < 1.0)
    is_charged = (charge != 0)
    is_pion_pm = (np.abs(pdg) == PDG_PION_PLUS)
    is_kaon_pm = (np.abs(pdg) == PDG_KAON_PLUS)

    mult20 = _per_event_reduce(np.ones_like(pt), in_eta20 & is_charged, offsets, op="sum").astype(np.int32)
    mean_pt = _per_event_reduce(pt, in_eta10 & is_charged, offsets, op="mean")
    n_pion = _per_event_reduce(np.ones_like(pt), in_eta10 & is_charged & is_pion_pm, offsets, op="sum").astype(np.int32)
    n_kaon = _per_event_reduce(np.ones_like(pt), in_eta10 & is_charged & is_kaon_pm, offsets, op="sum").astype(np.int32)

    # Proton / pion ratio at mid-rapidity. Use n_proton in the full event for the
    # numerator (it is a small number for FXT events anyway) — using the
    # eta10-restricted proton count would be cleaner but is not stored separately;
    # the bulk of protons are forward at FXT so this captures composition trend
    # well enough for an MLP feature. Avoid div-by-zero with a small floor.
    p_over_pi = (n_proton.astype(np.float32) / np.maximum(n_pion.astype(np.float32), 1.0)).astype(np.float32)

    # Compose the feature matrix in the documented order.
    feats = np.column_stack([
        np.full(n_events, sqrtsNN, dtype=np.float32),
        mult05.astype(np.float32),
        mult10.astype(np.float32),
        mult20.astype(np.float32),
        n_proton.astype(np.float32),
        n_antiproton.astype(np.float32),
        mean_pt,
        n_pion.astype(np.float32),
        n_kaon.astype(np.float32),
        p_over_pi,
    ])

    # Truth-tuned baseline centrality labels (each event's bin index, computed earlier).
    with h5py.File(baseline_pred_h5, "r") as hp:
        centrality_bin = hp["centrality_bin"][:]
    if centrality_bin.shape[0] != n_events:
        raise RuntimeError(f"baseline prediction size mismatch: {centrality_bin.shape[0]} vs {n_events}")

    return EventDataset(
        features=feats, b=b, centrality_bin=centrality_bin,
        sqrtsNN=sqrtsNN, n_events=n_events, feature_names=list(FEATURE_NAMES),
    )


def stack_energies(datasets: list[EventDataset]) -> EventDataset:
    """Concatenate per-energy datasets into one. Useful for the joint training run."""
    feature_names = datasets[0].feature_names
    feats = np.concatenate([d.features for d in datasets], axis=0)
    b = np.concatenate([d.b for d in datasets], axis=0)
    bins = np.concatenate([d.centrality_bin for d in datasets], axis=0)
    return EventDataset(
        features=feats, b=b, centrality_bin=bins,
        sqrtsNN=float("nan"),     # not a single value across energies; the feature column carries it
        n_events=int(feats.shape[0]),
        feature_names=feature_names,
    )
