"""Single source of truth for v1 event/particle cuts and the CM→lab boost.

Locked by `project_scope_v2.md`; restructured 2026-05-28 to make UrQMD +
detector + b<14 the headline study (see docs/restructure_urqmd_detector_b14.md).
The four locked cuts applied uniformly to SMASH and UrQMD events:

  1. CM → lab (target-rest) frame boost.  Both generators ship in CM frame
     (verified empirically — spectator-proton y peaks at ±y_beam_cm).
  2. Event filter:        b < 14 fm  AND  N_part > 0.
  3. Charged-only filter:  charge != 0  on per-particle inputs.
  4. Spectator removal:   drop  (|pdg| ∈ {2112, 2212})  AND  (ncoll == 0).

Npart source (2026-05-28): the UrQMD-primary study uses UrQMD's *native*
Glauber-header Npart (always ≥0), not the spectator-rule recompute below.
`npart_from_spectators` is retained for the SMASH ingest and as a diagnostic,
but is no longer the UrQMD label source — it is what produced the negative
peripheral Npart that previously forced the b<11 cut (see B_MAX_FM note).

Detector emulation (Phase B, optional — applied per-particle when the builder
is run with --detector):
  * η window:        ETA_LAB_MIN < η_lab < ETA_LAB_MAX  (default 0 < η_lab < 1.5,
                     FXT TPC acceptance per Kimelman thesis Ch.6).
  * pT minimum:      pT_smeared > PT_MIN_GEV  (default 50 MeV/c).
  * pT smearing:     Gaussian with σ(pT) = (PT_SMEAR_CONST + PT_SMEAR_SLOPE·pT)·pT,
                     applied via `smear_pt`. The stored pT in detector caches is
                     the SMEARED value (matches what an experiment measures).
  * Efficiency:      logistic ε(pT) = EFF_PLATEAU / (1 + exp(-(pT-EFF_PT50)/EFF_WIDTH))
                     via `efficiency_at_pt`; Bernoulli-drawn inside
                     `detector_keep_mask` against the smeared pT.

Order of operations per particle (locked):
  boost → η_lab → particle_keep_mask (charged + non-spectator)
        → η window → smear_pt → pT_min on smeared → Bernoulli ε(pT_smeared)

All stochastic ops take an injected `np.random.Generator` — never module state —
so detector caches are deterministic given (input files, file order,
--chunk-events, --seed). PID/mass remain non-features either way.

PID-based filters are still excluded by design — the networks see only
(pT, η_lab, φ, charge) per particle regardless of detector mode.

If any of these become contested in a paper draft, look HERE first — drift
between SMASH and UrQMD cut conventions is the obvious failure mode.

Caveat on spectator removal: SMASH's `ncoll` includes elastic scatterings
on produced mesons; UrQMD's ncoll counts NN scatterings only. Same rule
applied; slightly different spectator sets. Footnote-worthy, not a blocker.
"""

from __future__ import annotations

import numpy as np

# Physics constants.
M_N = 0.938        # GeV, nucleon mass — used for the CM→lab boost rapidity.
B_MAX_FM = 14.0    # fm. Symmetric event cut for both generators — the full
                   # UrQMD b_max. Was temporarily tightened to 11 to dodge the
                   # spectator-rule negative-Npart tail (at b > 11 fm UrQMD's
                   # ncoll does not inherit to Δ→Nπ daughters, so ~4.5% of
                   # peripheral events got negative recomputed Npart). The
                   # 2026-05-28 restructure makes UrQMD primary and uses UrQMD's
                   # native Glauber-header Npart (always ≥0) instead, so the
                   # pathology no longer applies and the full b<14 range is kept.
                   # SMASH's ncoll inherits correctly, so b<14 is safe there too.
                   # See docs/restructure_urqmd_detector_b14.md (supersedes the
                   # b<11 conclusion in docs/npart_reconciliation.md).
A_AU = 197         # Au mass number.
AU_2A = 2 * A_AU   # 394; max possible nucleon count in Au+Au.

# PDG codes for the spectator definition.
PDG_NEUTRON = 2112
PDG_PROTON = 2212

# --- Phase B detector-emulation parameters (only used when --detector) --------
#
# FXT TPC acceptance window. The boost puts target spectators at η_lab≈0 and
# beam spectators at η_lab≈+2·y_cm, so the cut is one-sided.
ETA_LAB_MIN = 0.0
ETA_LAB_MAX = 1.5

# Track-level pT threshold, applied AFTER smearing (mimics the way a real
# experiment cuts on measured pT, not on the underlying truth pT).
PT_MIN_GEV = 0.050

# Gaussian pT smearing: σ(pT) = (PT_SMEAR_CONST + PT_SMEAR_SLOPE·pT) · pT.
# Constant term ≈ 1%, pT-proportional term ≈ 0.5% per GeV/c — modest TPC-style
# resolution at FXT energies.
PT_SMEAR_CONST = 0.01
PT_SMEAR_SLOPE = 0.005

# Logistic tracking efficiency: ε(pT) = EFF_PLATEAU / (1 + exp(-(pT-EFF_PT50)/EFF_WIDTH)).
# 50% efficient at 75 MeV/c (just above the pT_min cut), plateau at 90%.
EFF_PLATEAU = 0.90
EFF_PT50 = 0.075
EFF_WIDTH = 0.015

# Default RNG seed; builders accept --seed to override but should print and
# persist the seed used so a cache is reproducible from its attrs alone.
DEFAULT_DETECTOR_SEED = 42


def y_cm_to_lab(sqrtsNN_GeV: float) -> float:
    """Rapidity of the lab/target-rest frame as seen from the CM frame.

    For a symmetric fixed-target collision A+A at √sNN, the target is at rest in
    the lab and the CM moves at y_cm = arccosh(√sNN / (2 m_N)) toward +z.  To
    transform a CM-frame 4-momentum into the target-rest lab frame we apply a
    longitudinal boost by this rapidity along +z (handled by `boost_cm_to_lab`).
    """
    return float(np.arccosh(sqrtsNN_GeV / (2.0 * M_N)))


def boost_cm_to_lab(
    p0: np.ndarray,
    pz: np.ndarray,
    sqrtsNN_GeV: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Longitudinal boost (p0, pz)_CM → (p0, pz)_lab. Vectorized.

    Lorentz transform along +z with (γ, β) = (cosh y, tanh y), where y is the
    rapidity of the lab frame relative to the CM frame (see `y_cm_to_lab`).
    Equivalently, every particle's longitudinal rapidity gets shifted by +y_cm.
    pT and φ are invariant under longitudinal boosts and are not touched here.
    """
    y_boost = y_cm_to_lab(sqrtsNN_GeV)
    gamma = np.cosh(y_boost)
    beta = np.tanh(y_boost)
    # Standard form: p' = γ(p + β·p₀), p₀' = γ(p₀ + β·p).
    pz_lab = gamma * (pz + beta * p0)
    p0_lab = gamma * (p0 + beta * pz)
    return p0_lab, pz_lab


def eta_from_pz(pT: np.ndarray, pz: np.ndarray) -> np.ndarray:
    """Pseudorapidity from (pT, pz). Small ε guards the log near the beam axis.

    η ≡ ½ ln((p+pz)/(p−pz)) where p = √(pT² + pz²). Stable for our pT range
    (≥ a few MeV/c); the ε is only there to avoid 0/0 on numerically zero pT.
    """
    p = np.sqrt(pT * pT + pz * pz)
    eps = 1e-12
    return 0.5 * np.log((p + pz + eps) / (p - pz + eps))


def event_passes(b_fm: float | np.ndarray, n_part: int | np.ndarray):
    """Locked v1 event-level filter: b < 14 fm AND N_part > 0.

    Works on scalars or arrays — returns a bool / bool-ndarray respectively.
    `N_part > 0` removes pure geometric misses (no NN overlap, no real
    collision). `b < B_MAX_FM` keeps the full UrQMD impact-parameter range
    (see B_MAX_FM docstring).
    """
    return (np.asarray(b_fm) < B_MAX_FM) & (np.asarray(n_part) > 0)


def npart_from_spectators(
    pdg: np.ndarray,
    ncoll: np.ndarray,
    offset: np.ndarray | None = None,
) -> np.ndarray | int:
    """N_part from the spectator-counting rule: N_part = 2A − N_spec.

    Single source of truth for the SMASH-style participant definition that we
    apply uniformly to both SMASH and UrQMD events. Spectator = nucleon
    (|pdg| ∈ {2112, 2212}) AND ncoll == 0. This is an upper bound vs the
    Glauber convention (it counts elastic-only participants too) and must be
    documented as such in the paper — see [[project-scope-v2]].

    Two modes:
      * `offset` is None → treat (pdg, ncoll) as a single event's particles.
        Returns a Python int.
      * `offset` is an int array of length n_events+1 → particles for event i
        live at [offset[i]:offset[i+1]]. Returns an int16 ndarray of length
        n_events. Used by the UrQMD cache builder; the SMASH ingester computes
        per-event in its event loop and goes through the scalar branch.

    Reconciles SMASH ⟨Npart⟩ ≈ 196 vs UrQMD-stored ⟨Npart⟩ ≈ 123: applying this
    rule to UrQMD raises its ⟨Npart⟩ to 185–189, closing the gap to within ~3
    units. See docs/npart_reconciliation.md.
    """
    is_nucleon = (np.abs(pdg) == PDG_NEUTRON) | (np.abs(pdg) == PDG_PROTON)
    is_spectator = is_nucleon & (ncoll == 0)

    if offset is None:
        return AU_2A - int(is_spectator.sum())

    # Vectorized per-event reduction. np.add.reduceat slices [offset[i]:offset[i+1]]
    # using offset[:-1] as the slice starts; the value at the trailing offset is
    # discarded (it represents the past-the-end sum that reduceat appends).
    spec_int = is_spectator.astype(np.int32)
    n_spec_per_event = np.add.reduceat(spec_int, offset[:-1])
    # Guard against empty events (offset[i] == offset[i+1]): reduceat returns
    # the value at offset[i] which is 0/1; for a zero-length event we want 0,
    # not the spillover from the next particle. Mask those explicitly.
    empty = np.diff(offset) == 0
    n_spec_per_event[empty] = 0
    return (AU_2A - n_spec_per_event).astype(np.int16)


def particle_keep_mask(
    charge: np.ndarray,
    pdg: np.ndarray,
    ncoll: np.ndarray,
) -> np.ndarray:
    """Locked v1 per-particle filter: charged AND not a spectator nucleon.

    Spectator = (|pdg| in {2112, 2212}) AND (ncoll == 0). At FXT energies these
    target/projectile nucleons sit at η_lab ≈ 0 or +2·y_cm (truth-level
    artifact) and would otherwise inflate the in-acceptance multiplicity in a
    way no experiment can reproduce.
    """
    is_nucleon = (np.abs(pdg) == PDG_NEUTRON) | (np.abs(pdg) == PDG_PROTON)
    is_spectator = is_nucleon & (ncoll == 0)
    return (charge != 0) & (~is_spectator)


# --- Phase B detector emulation helpers ---------------------------------------
#
# Each helper takes an injected RNG (where applicable) so callers control
# determinism; nothing here touches module-level random state.


def smear_pt(pT: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gaussian pT smearing with σ(pT)/pT = PT_SMEAR_CONST + PT_SMEAR_SLOPE·pT.

    Returns a new array of the same shape. Negative draws are clipped to a
    small positive floor (1e-6 GeV/c) so downstream log/division stays finite
    — physically equivalent to a track reconstructed at numerically zero pT.
    The smearing is symmetric in additive terms so the population-mean pT is
    preserved; the distribution RMS broadens.
    """
    sigma = (PT_SMEAR_CONST + PT_SMEAR_SLOPE * pT) * pT
    smeared = pT + rng.normal(0.0, 1.0, size=pT.shape).astype(pT.dtype) * sigma.astype(pT.dtype)
    return np.maximum(smeared, np.asarray(1e-6, dtype=pT.dtype))


def efficiency_at_pt(pT: np.ndarray) -> np.ndarray:
    """Per-track tracking efficiency probability ε(pT).

    Logistic turn-on with 50% point at EFF_PT50 and width EFF_WIDTH, asymptoting
    to EFF_PLATEAU. No randomness — returns the probability so a caller can
    Bernoulli-draw (see `detector_keep_mask`) or simply plot/inspect the curve.
    """
    x = (pT - EFF_PT50) / EFF_WIDTH
    return EFF_PLATEAU / (1.0 + np.exp(-x))


def detector_keep_mask(
    pT_smeared: np.ndarray,
    eta_lab: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Combined detector-side keep mask: η window AND pT_min AND Bernoulli ε.

    Composes the three post-smearing detector cuts so the cache builder makes
    a single call. Does NOT include the truth-level particle_keep_mask
    (charged + non-spectator) — that runs first on truth pT in the builder so
    the order of operations stays explicit at the call site.
    """
    in_eta = (eta_lab > ETA_LAB_MIN) & (eta_lab < ETA_LAB_MAX)
    above_pt_min = pT_smeared > PT_MIN_GEV
    eff = efficiency_at_pt(pT_smeared)
    bernoulli_keep = rng.random(size=pT_smeared.shape) < eff
    return in_eta & above_pt_min & bernoulli_keep
