"""Event-level scalar feature extraction for the MLP baseline (lab frame, v2).

Aligned with `project_scope_v2.md` (2026-05-27):
  * Lab frame everywhere (CM→lab boost applied to every particle).
  * Same particle set as the per-particle networks: charged + non-spectator
    (no η window). Cuts imported from `src.data.cuts`.
  * Event-level scalars are derived from this in-acceptance set so the MLP
    and the permutation-invariant networks see the same physics.

Features (current set; the MLP config's n_features must match):
    1. sqrtsNN              — energy label (carried per-event)
    2. mult_lab             — charged non-spectator multiplicity, no η cut
    3. mean_pT_lab          — ⟨pT⟩ over the same set
    4. total_pT_lab         — Σ pT over the same set
    5. n_proton             — proton count (final-state) over the same set
    6. n_antiproton         — antiproton count over the same set
    7. n_pion_charged       — charged-pion count over the same set
    8. n_kaon_charged       — charged-kaon count over the same set
    9. p_over_pi            — proton/pion ratio (avoid div-by-zero with floor=1)

n_proton and the PID-derived counts are event-level scalars only — they are
NOT exposed as per-particle features (the per-particle networks see only pT,
η_lab, φ, charge per particle). So this scalar list is the MLP's
"informational compensation" for not having particle-level inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from src.data.cuts import (
    DEFAULT_DETECTOR_SEED,
    ETA_LAB_MAX,
    ETA_LAB_MIN,
    boost_cm_to_lab,
    detector_keep_mask,
    eta_from_pz,
    event_passes,
    particle_keep_mask,
    smear_pt,
)

# Feature names in the order they appear in the output matrix.
FEATURE_NAMES = [
    "sqrtsNN",
    "mult_lab",
    "mean_pT_lab",
    "total_pT_lab",
    "n_proton",
    "n_antiproton",
    "n_pion_charged",
    "n_kaon_charged",
    "p_over_pi",
]
N_FEATURES = len(FEATURE_NAMES)

# PDG codes used by the composition features.
PDG_PROTON = 2212
PDG_PION_PLUS = 211
PDG_KAON_PLUS = 321


@dataclass
class EventDataset:
    """Per-event arrays for one energy file. Light wrapper for callers."""
    features: np.ndarray         # (n_events, n_features) float32
    b: np.ndarray                # (n_events,) float32  — truth impact parameter
    centrality_bin: np.ndarray   # (n_events,) int16    — bin from the truth-tuned baseline
    sqrtsNN: float
    n_events: int
    feature_names: list[str]
    detector: bool = False       # True if Phase-B detector emulation was applied


def _per_event_reduce(values: np.ndarray, mask: np.ndarray,
                      offsets: np.ndarray, *, op: str) -> np.ndarray:
    """Sum or mean of `values` within `mask` per event, indexed by offsets.

    Vectorised via np.add.reduceat — scales to 100k-event files without a
    Python loop. For op='mean' returns 0 (not NaN) when an event has no
    in-mask particles.
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


def load_features(
    h5_path: Path,
    baseline_pred_h5: Path,
    *,
    detector: bool = False,
    seed: int = DEFAULT_DETECTOR_SEED,
) -> EventDataset:
    """Build the per-event feature matrix in the lab frame on the locked set.

    Event-level filter (b<11 fm AND N_part>0) is applied here so this loader
    produces only events that survive the v1 cuts — the row count matches what
    the per-particle cache holds for the same energy.

    Per-particle filter (truth mode):    charged + non-spectator, no η window.
    Per-particle filter (detector mode): charged + non-spectator + 0<η_lab<1.5,
    then pT smearing + pT_min cut + Bernoulli ε(pT). pT-derived event scalars
    (mean_pT_lab, total_pT_lab) are recomputed from the smeared pT; PID-derived
    counts (n_proton, …) use unsmeared PDG since no PID smearing is in scope.
    """
    with h5py.File(h5_path, "r") as h:
        sqrtsNN = float(h.attrs["sqrtsNN"])
        n_events_raw = int(h.attrs["n_events"])
        b_all = h["b"][:].astype(np.float32)
        npart_all = h["Npart"][:].astype(np.int32)
        offsets_all = h["offset"][:].astype(np.int64)
        pdg = h["particles/pdg"][:].astype(np.int32)
        pt = h["particles/pT"][:].astype(np.float32)
        eta_cm = h["particles/eta"][:].astype(np.float32)
        mass = h["particles/mass"][:].astype(np.float32)
        charge = h["particles/charge"][:].astype(np.int8)
        ncoll = h["particles/ncoll"][:].astype(np.int16)

    # Particle-level CM→lab boost. Reconstruct (p0, pz) from (pT, η, mass) so
    # the boost has the inputs it expects. pT and φ are boost-invariant, so we
    # only need to recompute η_lab afterwards.
    pz_cm = pt * np.sinh(eta_cm)
    p0_cm = np.sqrt(pt * pt * np.cosh(eta_cm) ** 2 + mass * mass)
    _p0_lab, pz_lab = boost_cm_to_lab(p0_cm, pz_cm, sqrtsNN)
    # eta_lab is computed but not used directly as a scalar — kept for
    # symmetry with the per-particle builders and to make sure the boost path
    # actually runs (catches silent failures during code review).
    _eta_lab = eta_from_pz(pt, pz_lab)

    # Particle filter. Detector mode composes the truth filter with the η
    # window, smearing, pT_min, and Bernoulli ε draws — same order as in the
    # padded-cache builders so MLP and per-particle nets see the same survivors.
    if detector:
        rng = np.random.default_rng(seed)
        pt_smeared = smear_pt(pt, rng)
        keep_p = (
            particle_keep_mask(charge, pdg, ncoll)
            & np.isfinite(_eta_lab)
            & (_eta_lab > ETA_LAB_MIN) & (_eta_lab < ETA_LAB_MAX)
            & detector_keep_mask(pt_smeared, _eta_lab, rng)
        )
        pt_for_reduce = pt_smeared
    else:
        keep_p = particle_keep_mask(charge, pdg, ncoll) & np.isfinite(_eta_lab)
        pt_for_reduce = pt
    is_pion_pm = np.abs(pdg) == PDG_PION_PLUS
    is_kaon_pm = np.abs(pdg) == PDG_KAON_PLUS
    is_proton = pdg == PDG_PROTON
    is_antiproton = pdg == -PDG_PROTON

    ones = np.ones_like(pt, dtype=np.float32)

    # All reductions are over the in-acceptance set (keep_p AND species mask).
    # pT-derived scalars use pt_for_reduce so they follow the smeared pT in
    # detector mode; multiplicities and PID counts follow keep_p naturally.
    mult_lab = _per_event_reduce(ones, keep_p, offsets_all, op="sum").astype(np.int32)
    total_pT_lab = _per_event_reduce(pt_for_reduce, keep_p, offsets_all, op="sum").astype(np.float32)
    mean_pT_lab = _per_event_reduce(pt_for_reduce, keep_p, offsets_all, op="mean").astype(np.float32)
    n_proton = _per_event_reduce(ones, keep_p & is_proton, offsets_all, op="sum").astype(np.int32)
    n_antiproton = _per_event_reduce(ones, keep_p & is_antiproton, offsets_all, op="sum").astype(np.int32)
    n_pion = _per_event_reduce(ones, keep_p & is_pion_pm, offsets_all, op="sum").astype(np.int32)
    n_kaon = _per_event_reduce(ones, keep_p & is_kaon_pm, offsets_all, op="sum").astype(np.int32)

    # Composition feature: proton/pion at the event level. Floor at 1.0 (not at
    # n_pion) so denominators of zero return n_proton (small int) rather than
    # exploding — chosen to match the original semantics from v1.
    p_over_pi = (n_proton.astype(np.float32) /
                 np.maximum(n_pion.astype(np.float32), 1.0)).astype(np.float32)

    # Build the full-row feature matrix at the *raw* size first, then filter
    # rows by the locked event filter. This keeps the index correspondence with
    # `b_all` and the baseline-prediction file simple — both arrays are
    # full-size and we drop the same rows from each.
    feats_raw = np.column_stack([
        np.full(n_events_raw, sqrtsNN, dtype=np.float32),
        mult_lab.astype(np.float32),
        mean_pT_lab,
        total_pT_lab,
        n_proton.astype(np.float32),
        n_antiproton.astype(np.float32),
        n_pion.astype(np.float32),
        n_kaon.astype(np.float32),
        p_over_pi,
    ])

    # Truth-tuned baseline centrality labels. Two layouts are accepted to ease
    # migration: a full-size file (length == n_events_raw, gets filtered here)
    # or a pre-filtered file (length == n_events_post_filter). Older truth
    # outputs may have been built against a different event-filter convention
    # (e.g. only N_part>0); we cannot silently realign those, so we reject.
    keep_e = event_passes(b_all, npart_all)
    n_events = int(keep_e.sum())
    with h5py.File(baseline_pred_h5, "r") as hp:
        centrality_bin_raw = hp["centrality_bin"][:]
    if centrality_bin_raw.shape[0] == n_events_raw:
        centrality_bin = centrality_bin_raw[keep_e]
    elif centrality_bin_raw.shape[0] == n_events:
        # Baseline already produced against the locked b<14 ∧ N_part>0 set.
        centrality_bin = centrality_bin_raw
    else:
        raise RuntimeError(
            f"baseline prediction size mismatch: got {centrality_bin_raw.shape[0]} "
            f"rows, expected either {n_events_raw} (full) or {n_events} "
            f"(post-cut). Rebuild the truth baseline against the v1 cache."
        )

    feats = feats_raw[keep_e]
    b = b_all[keep_e]

    return EventDataset(
        features=feats, b=b, centrality_bin=centrality_bin,
        sqrtsNN=sqrtsNN, n_events=n_events, feature_names=list(FEATURE_NAMES),
        detector=detector,
    )


def stack_energies(datasets: list[EventDataset]) -> EventDataset:
    """Concatenate per-energy datasets into one (for joint training)."""
    feature_names = datasets[0].feature_names
    # All inputs must share detector mode; mixing truth + detector would silently
    # train on inconsistent features.
    detector_flags = {d.detector for d in datasets}
    if len(detector_flags) > 1:
        raise ValueError("stack_energies got a mix of truth and detector datasets")
    feats = np.concatenate([d.features for d in datasets], axis=0)
    b = np.concatenate([d.b for d in datasets], axis=0)
    bins = np.concatenate([d.centrality_bin for d in datasets], axis=0)
    return EventDataset(
        features=feats, b=b, centrality_bin=bins,
        sqrtsNN=float("nan"),  # not a single value; the feature column carries it
        n_events=int(feats.shape[0]),
        feature_names=feature_names,
        detector=next(iter(detector_flags)),
    )
