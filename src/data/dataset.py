"""Event-level dataset utilities for SMASH ROOT → HDF5 ingestion.

The functions here are kept pure (no I/O) so they can be unit-tested and reused
by both the ingestion script and the dataloader.

Frame convention: SMASH dumps 4-momenta in the center-of-mass frame. So η and y
computed directly from raw (px,py,pz,p0) are CM-frame quantities, and |η|<0.5
is already mid-rapidity. No boost is required for FXT analyses on this output.
"""

from __future__ import annotations

import numpy as np

# Au has A = 197, so the maximum possible nucleon count in Au+Au is 2A.
# Used by N_part derivation: N_part = 2A − N_spectators.
AU_2A = 394

# PDG codes for nucleons (charge-conjugate handled via |pdg|).
PDG_NEUTRON = 2112
PDG_PROTON = 2212
PDG_NUCLEONS = (PDG_NEUTRON, PDG_PROTON)


def npart_from_event(pdg: np.ndarray, ncoll: np.ndarray) -> int:
    """N_part for one event, derived from final-state nucleon spectators.

    Defined as 2A − N_spectators, where a spectator is a final-state nucleon
    (|pdg| ∈ {2112, 2212}) that never scattered (ncoll == 0). We count spectators
    rather than participants directly because some participants are converted
    into Δ or N* resonances and decay to non-nucleon final states — those would
    silently undercount with a direct participant count, but they are by
    construction not spectators, so the spectator-based definition is exact.
    """
    # Caveat: SMASH's per-particle `ncoll` counts elastic scatterings as well as
    # inelastic. The Glauber convention defines participants as nucleons that
    # underwent at least one *inelastic* NN collision. Our derivation includes
    # elastic-only participants, so this is an upper bound vs Glauber. Document
    # in the paper; do not report this number as if it were the Glauber N_part.
    is_nucleon = (np.abs(pdg) == PDG_NEUTRON) | (np.abs(pdg) == PDG_PROTON)
    n_spec = int(np.sum(is_nucleon & (ncoll == 0)))
    return AU_2A - n_spec


def pt_eta_phi(px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transverse momentum, pseudorapidity, and azimuthal angle from Cartesian 3-momentum.

    η = ½ ln((|p|+pz)/(|p|−pz)) with a small ε to keep log finite for pz → ±|p|.
    """
    pt = np.sqrt(px * px + py * py)
    p = np.sqrt(pt * pt + pz * pz)
    eps = 1e-12
    eta = 0.5 * np.log((p + pz + eps) / (p - pz + eps))
    phi = np.arctan2(py, px)
    return pt.astype(np.float32), eta.astype(np.float32), phi.astype(np.float32)


def charged_mult_in_eta(eta: np.ndarray, charge: np.ndarray, eta_max: float) -> int:
    """Charged-particle multiplicity within |η| < eta_max."""
    return int(np.sum((np.abs(eta) < eta_max) & (charge != 0)))


def count_pdg(pdg: np.ndarray, target_pdg: int) -> int:
    """Count particles of an exact PDG code (signed; use ±code separately for anti)."""
    return int(np.sum(pdg == target_pdg))


# HDF5 schema documentation.
# Datasets at the root group:
#   sqrtsNN     (N_events,)      float32   beam-energy label (one constant per file)
#   b           (N_events,)      float32   impact parameter (fm)
#   Npart       (N_events,)      int16     derived participant count
#   nparticles  (N_events,)      int32     particle count per event
#   offset      (N_events + 1,)  int64     cumulative offsets into the flat particle arrays
#   mult_eta05  (N_events,)      int32     charged-particle multiplicity |η| < 0.5
#   mult_eta10  (N_events,)      int32     charged-particle multiplicity |η| < 1.0
#   n_proton    (N_events,)      int32     proton count (pdg == 2212)
#   n_antiproton(N_events,)      int32     antiproton count (pdg == -2212)
#   particles/pdg    (N_total,)  int32
#   particles/pT     (N_total,)  float32
#   particles/eta    (N_total,)  float32
#   particles/phi    (N_total,)  float32
#   particles/mass   (N_total,)  float32
#   particles/charge (N_total,)  int8
# File attributes: sqrtsNN, n_events, source_root, smash_version, uproot_version,
# ingest_commit, ingest_iso8601.
