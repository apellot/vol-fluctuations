"""Glauber Monte Carlo + NBD multiplicity ansatz for the classical baseline.

Standard two-component construction:

    1. Sample two Au nuclei from a Woods–Saxon density.
    2. Sample an impact parameter b uniformly in b² (i.e., dN/db ∝ b for min-bias).
    3. Count participants and binary NN collisions via the geometric overlap
       criterion: two nucleons collide if their transverse separation d satisfies
       π d² < σ_NN, with σ_NN the inelastic NN cross section at √sNN.
    4. For each Glauber event, sample a charged multiplicity from a negative
       binomial distribution whose mean is N_a × n_pp, where
           N_a = x · N_part / 2 + (1 − x) · N_coll
       (the two-component "wounded nucleon + binary collision" ansatz).
    5. Fit (n_pp, k, x) per energy by matching the Glauber-predicted multiplicity
       distribution to the SMASH multiplicity distribution.

The sum of N_a iid NBD(n_pp, k) variables is itself NBD(N_a · n_pp, N_a · k);
we exploit this so that each Glauber event needs only ONE multiplicity sample
rather than a Python loop over ancestors. Because N_a is non-integer we use the
Gamma–Poisson mixture form, which extends NBD continuously to non-integer shape.

References:
    Loizides, Kamin, d'Enterria, PRC 97 (2018) 054910 — Glauber Monte Carlo conventions
        used by STAR / ALICE / PHENIX.
    Kharzeev–Nardi, Phys. Lett. B 507 (2001) 121 — two-component NBD ansatz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Au standard Woods–Saxon parameters (PRC 97 (2018) 054910 Table 1).
AU_A = 197
AU_R = 6.38     # half-density radius (fm)
AU_a = 0.535    # surface diffuseness (fm)
AU_MAX_R = 12.0 # sampling cutoff — beyond ~2R+5a the density is negligible

# Inelastic NN cross sections at our four FXT energies.
# Kimelman thesis (p. 35) uses σ_NN = 28.2 mb at √sNN = 3 GeV, "determined by
# fitting p+p total and elastic cross section data and taking the difference".
# We use that as a reference at 3.2 GeV and let σ grow modestly with energy
# following PDG inelastic-pp parametrisations through our 4 FXT points.
SIGMA_NN_MB = {
    3.2: 28.2,
    3.5: 28.5,
    3.9: 29.5,
    4.5: 30.8,
}


@dataclass
class GlauberParams:
    """Fitted two-component NBD + efficiency parameters for one energy.

    The d parameter is the multiplicity-dependent efficiency knob from STAR's
    centrality code (Zach Sweger thesis Sec. 5.3.4.2): each event's Glauber-
    sampled multiplicity m is multiplied by ε(m) = 0.98 · (1 − d · m / d_norm)
    before comparison with data. d=0 is uniform 98% efficiency; d > 0 makes
    efficiency drop with multiplicity (modeling reduced tracker performance
    in high-occupancy events). Typical range 0 ≤ d ≤ 0.2.
    """
    sqrtsNN: float
    n_pp: float       # mean charged-mult per ancestor
    k: float          # NBD shape parameter
    x: float          # wounded-vs-binary mix, in [0, 1]
    sigma_NN: float   # cross section used (mb)
    d: float = 0.0    # multiplicity-dependent efficiency parameter
    d_norm: float = 200.0  # normalisation scale in the efficiency formula


# ---------------------------------------------------------------------------
# Nucleon sampling and Glauber geometry.
# ---------------------------------------------------------------------------

def sample_nucleus(n_events: int, A: int = AU_A, R: float = AU_R, a: float = AU_a,
                   r_max: float = AU_MAX_R, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample nucleon positions for n_events independent nuclei.

    Returns array of shape (n_events, A, 3). Uses rejection sampling on the
    Woods–Saxon density profile: propose r uniformly in [0, r_max] weighted by
    r² (the spherical volume Jacobian), accept with probability proportional to
    1 / (1 + exp((r − R)/a)). Then sample (cosθ, φ) uniformly on the sphere.
    """
    rng = rng or np.random.default_rng()
    # Oversample to allow rejection.  Acceptance fraction is well above 0.5 for
    # Au, so 4× oversample is comfortable.
    n_needed = n_events * A
    accepted = np.empty(n_needed, dtype=np.float64)
    n_have = 0
    while n_have < n_needed:
        oversample = max(4 * (n_needed - n_have), 1000)
        r_try = rng.uniform(0, r_max, size=oversample)
        # Weight: r² · WS(r); normalize the max so the acceptance probability is in [0, 1].
        ws = 1.0 / (1.0 + np.exp((r_try - R) / a))
        weight = (r_try ** 2) * ws
        # Peak of r² · WS is near r = R; bound it.
        weight_max = (R ** 2) * 1.0  # 1.0 = WS at r=0; r² peaks well above r²·WS, but use this conservative bound
        # The true bound is at r ~ R where WS ~ 0.5: peak ≈ 0.5 R²; use R² to be safe.
        u = rng.uniform(0, R ** 2, size=oversample)
        accept = u < weight
        take = r_try[accept]
        room = n_needed - n_have
        if take.size > room:
            take = take[:room]
        accepted[n_have:n_have + take.size] = take
        n_have += take.size

    r = accepted.reshape(n_events, A)
    cos_theta = rng.uniform(-1, 1, size=(n_events, A))
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    phi = rng.uniform(0, 2 * np.pi, size=(n_events, A))

    pos = np.empty((n_events, A, 3), dtype=np.float32)
    pos[..., 0] = (r * sin_theta * np.cos(phi)).astype(np.float32)
    pos[..., 1] = (r * sin_theta * np.sin(phi)).astype(np.float32)
    pos[..., 2] = (r * cos_theta).astype(np.float32)
    return pos


def compute_npart_ncoll(pos_A: np.ndarray, pos_B: np.ndarray, b: np.ndarray,
                        sigma_NN_mb: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-event N_part and N_coll given two nuclei and an impact parameter.

    Inputs:
        pos_A, pos_B: (n_events, A, 3) nucleon positions in each nucleus's own frame.
        b:           (n_events,) impact parameter along x.
        sigma_NN_mb: inelastic NN cross section in millibarns.
    The collision criterion is transverse-separation < d_max, where π d_max² = σ_NN.
    σ_NN_mb of 30 mb corresponds to d_max ≈ 0.977 fm.
    """
    # Shift nucleus B by impact parameter along x, leaving y and z untouched (z is the beam axis).
    pos_B_shifted = pos_B.copy()
    pos_B_shifted[..., 0] = pos_B_shifted[..., 0] + b[:, None]

    # Convert σ from mb (= 0.1 fm²) to fm². 1 mb = 0.1 fm².
    d_max_sq = (sigma_NN_mb * 0.1) / np.pi

    n_events, A, _ = pos_A.shape

    # Pairwise transverse distances: (n_events, A, A, 2).  This is the memory-heavy
    # operation; for A=197 and n_events=1000 it is 1.5 GB at float32 — process in chunks.
    chunk = 500
    n_part = np.zeros(n_events, dtype=np.int32)
    n_coll = np.zeros(n_events, dtype=np.int32)
    for start in range(0, n_events, chunk):
        end = min(start + chunk, n_events)
        a_xy = pos_A[start:end, :, :2]          # (chunk, A, 2)
        b_xy = pos_B_shifted[start:end, :, :2]  # (chunk, A, 2)
        # Pairwise dx, dy: (chunk, A_a, A_b)
        dx = a_xy[:, :, None, 0] - b_xy[:, None, :, 0]
        dy = a_xy[:, :, None, 1] - b_xy[:, None, :, 1]
        dist2 = dx * dx + dy * dy
        collision = dist2 < d_max_sq  # (chunk, A_a, A_b)
        # N_coll: total number of colliding pairs.
        n_coll[start:end] = collision.sum(axis=(1, 2))
        # N_part: nucleons in A that collide at least once, plus same for B.
        part_A = collision.any(axis=2).sum(axis=1)
        part_B = collision.any(axis=1).sum(axis=1)
        n_part[start:end] = part_A + part_B

    return n_part, n_coll


def glauber_events(n_events: int, sqrtsNN: float, b_max: float = 20.0,
                   rng: np.random.Generator | None = None,
                   require_collision: bool = True) -> dict:
    """Generate Glauber-MC events at one energy.

    Returns dict with keys: b (fm), n_part, n_coll — each of length ≤ n_events.
    Impact parameter is sampled from dN/db ∝ b on [0, b_max].

    require_collision: if True (default), events with N_part == 0 (pure geometric
    misses where the two Au radii never overlap) are dropped. This matches the
    SMASH data filter applied in build_padded_cache.py, so the data and model
    multiplicity distributions are compared on the same population of "actual
    collisions" rather than including the b > 2R no-collision events.
    """
    rng = rng or np.random.default_rng()
    if sqrtsNN not in SIGMA_NN_MB:
        raise ValueError(f"No σ_NN value for √sNN = {sqrtsNN}. Add to SIGMA_NN_MB.")
    sigma = SIGMA_NN_MB[sqrtsNN]

    # Oversample so we still hit n_events after dropping no-collision events.
    # Empirically ~58 % of events at b_max=20 fm have N_part>0; 1.8× oversample
    # is comfortable. The over-sampled events are independent draws so the
    # surviving sample is still an unbiased min-bias sample.
    oversample = int(n_events * 1.8) if require_collision else n_events
    b = np.sqrt(rng.uniform(0, b_max ** 2, size=oversample)).astype(np.float32)
    pos_A = sample_nucleus(oversample, rng=rng)
    pos_B = sample_nucleus(oversample, rng=rng)
    n_part, n_coll = compute_npart_ncoll(pos_A, pos_B, b, sigma)
    if require_collision:
        keep = n_part > 0
        b = b[keep]; n_part = n_part[keep]; n_coll = n_coll[keep]
        # Trim back to requested count (or fewer if oversample wasn't enough).
        if len(b) > n_events:
            b = b[:n_events]; n_part = n_part[:n_events]; n_coll = n_coll[:n_events]
    return {"b": b, "n_part": n_part, "n_coll": n_coll, "sigma_NN": sigma}


# ---------------------------------------------------------------------------
# Two-component NBD multiplicity.
# ---------------------------------------------------------------------------

def nbd_multiplicity(n_part: np.ndarray, n_coll: np.ndarray,
                     n_pp: float, k: float, x: float,
                     d: float = 0.0, d_norm: float = 200.0,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample charged-multiplicity per event under the two-component NBD ansatz,
    with optional multiplicity-dependent efficiency.

    The number of "ancestors" is N_a = x · N_part/2 + (1 − x) · N_coll. Each ancestor
    contributes NBD(n_pp, k) charged particles independently. Because sums of iid
    NBDs are NBD with shape rescaled by N_a, we sample one Gamma-Poisson mixture
    per event with continuous shape parameter N_a · k.

    The efficiency factor ε(m) = 0.98 · (1 − d · m / d_norm) is applied to the
    raw NBD-sampled multiplicity. Clamped to ε ≥ 0 in case d is large enough to
    flip the sign at high mult. d=0 reduces this to a uniform 98% efficiency.

    For events with N_a == 0 (no participants — should be rare since b_max keeps
    a finite collision probability), the multiplicity is identically 0.
    """
    rng = rng or np.random.default_rng()
    n_a = x * (n_part * 0.5) + (1.0 - x) * n_coll
    has_ancestors = n_a > 0
    mult_raw = np.zeros_like(n_a, dtype=np.int64)
    if has_ancestors.any():
        shape = n_a[has_ancestors] * k
        scale = n_pp / k
        lam = rng.gamma(shape, scale)
        mult_raw[has_ancestors] = rng.poisson(lam)
    # Binomial detector response: each particle survives with probability ε(m). This
    # is what np.random.binomial(n, p) gives us, and it produces a properly smeared
    # observed-mult distribution. Using np.round(m · ε) instead creates a comb of
    # rounding artifacts that the χ² fit can lock onto.
    eff = np.clip(0.98 * (1.0 - d * mult_raw / d_norm), 0.0, 1.0).astype(np.float64)
    mult_obs = rng.binomial(mult_raw, eff).astype(np.int64)
    return mult_obs


# ---------------------------------------------------------------------------
# Fitting (n_pp, k, x) to a target multiplicity distribution.
# ---------------------------------------------------------------------------

def _multiplicity_distribution(mult: np.ndarray, n_bins: int | None = None,
                               mult_max: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Histogram with one bin per integer multiplicity value.

    `n_bins` is accepted for API compatibility but ignored — the bin count is
    determined entirely by mult_max so each bin spans exactly one integer.
    `mult_max` defaults to the global maximum of the input array; for a fit it
    should match between the data and the model so the χ² is computed on the
    same bin grid.
    """
    if mult_max is None:
        mult_max = int(mult.max())
    # Edges at 0, 1, 2, ..., mult_max+1 → integer-wide bins covering [0, mult_max].
    edges = np.arange(0, mult_max + 2)
    hist, _ = np.histogram(mult, bins=edges)
    density = hist / max(1, hist.sum())
    return edges, density


def _chi2_for_params(n_pp: float, k: float, x: float, d: float,
                     glauber: dict, target: np.ndarray, edges: np.ndarray,
                     rng: np.random.Generator, fit_mult_min: int = 0,
                     d_norm: float = 200.0) -> float:
    """χ² of the Glauber+NBD multiplicity distribution vs target.

    fit_mult_min restricts the comparison to bins above a minimum multiplicity —
    Kimelman thesis (p. 36) normalises and compares only on [40, 195] because
    the low-mult tail in data has trigger inefficiency and the high tail has
    pileup. Same idea here: the SMASH peripheral floor at mult ~ 25–35 comes
    from target-side spectator protons in the η_lab window, which the
    two-component NBD ansatz cannot reproduce. Excluding that region from the
    fit gives well-behaved parameters; the peripheral region still gets
    centrality assigned via the percentile mapping at apply time.
    """
    mult_g = nbd_multiplicity(glauber["n_part"], glauber["n_coll"], n_pp, k, x,
                              d=d, d_norm=d_norm, rng=rng)
    mult_max = int(edges[-1] - 1)
    _, dens_g = _multiplicity_distribution(mult_g, n_bins=len(target), mult_max=mult_max)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (target > 0) & (centers >= fit_mult_min)
    if mask.sum() == 0:
        return 1e6
    # Renormalise both distributions inside the fit window (thesis convention).
    t = target[mask] / target[mask].sum()
    g = dens_g[mask] / max(dens_g[mask].sum(), 1e-12)
    return float(np.sum(((g - t) ** 2) / t))


def fit_two_component_nbd(mult_target: np.ndarray,
                          glauber: dict,
                          rng: np.random.Generator | None = None,
                          fit_mult_min: int = 40) -> tuple[GlauberParams, dict]:
    """Fit (n_pp, k, x) by χ² minimization on the [fit_mult_min, ∞) window.

    Two-stage strategy: coarse grid to find the basin of attraction (the χ² surface
    is non-convex because x and n_pp trade off — higher x lowers the effective
    ancestor count which raises n_pp), then a Nelder–Mead polish for sub-grid
    precision. Each χ² evaluation re-samples NBD on the Glauber event set, so the
    objective has irreducible Monte-Carlo noise; we use a fixed seed for the
    polish step so the optimizer sees a deterministic surface.

    Per the Kimelman thesis (Ch. 4), the fit is normalised and compared inside
    the [fit_mult_min, mult_max] window; the low-mult tail (spectator floor in
    our simulation, trigger inefficiency in real data) is intentionally excluded.
    Default 40 matches the thesis lower bound at √sNN = 3 GeV.
    """
    rng = rng or np.random.default_rng(seed=0)
    edges, target = _multiplicity_distribution(mult_target)
    # d-norm: scale used inside ε(m) = 0.98(1 − d · m / d_norm). We use the
    # observed mult_max so that d ∈ [0, 0.2] has the same interpretation
    # (efficiency drop fraction at the top of the distribution) at every energy,
    # regardless of how high mult_max happens to be in this energy slice.
    d_norm = float(edges[-1] - 1)

    # Stage 1 — coarse grid (n_pp × k × x × d).
    n_pp_grid = np.array([0.3, 0.5, 0.7, 1.0, 1.5])
    k_grid = np.array([1.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    x_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    d_grid = np.array([0.0, 0.05, 0.10, 0.15])

    best = {"chi2": np.inf, "n_pp": None, "k": None, "x": None, "d": None}
    for x in x_grid:
        for n_pp in n_pp_grid:
            for k in k_grid:
                for d in d_grid:
                    chi2 = _chi2_for_params(n_pp, k, x, d, glauber, target, edges, rng,
                                            fit_mult_min=fit_mult_min, d_norm=d_norm)
                    if chi2 < best["chi2"]:
                        best.update(chi2=chi2, n_pp=n_pp, k=k, x=x, d=d)

    # Stage 2 — Nelder–Mead polish around the grid optimum (4-D simplex).
    from scipy.optimize import minimize

    def obj(theta: np.ndarray) -> float:
        n_pp_, k_, x_, d_ = theta
        if not (0.05 < n_pp_ < 10 and 0.05 < k_ < 200 and 0.0 <= x_ <= 1.0 and 0.0 <= d_ <= 0.5):
            return 1e6
        local_rng = np.random.default_rng(seed=12345)
        return _chi2_for_params(n_pp_, k_, x_, d_, glauber, target, edges, local_rng,
                                fit_mult_min=fit_mult_min, d_norm=d_norm)

    x0 = np.array([best["n_pp"], best["k"], best["x"], best["d"]])
    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 400})
    n_pp_fit, k_fit, x_fit, d_fit = res.x
    x_fit = float(np.clip(x_fit, 0.0, 1.0))
    d_fit = float(np.clip(d_fit, 0.0, 0.5))

    sqrtsNN_val = next(kk for kk, vv in SIGMA_NN_MB.items() if vv == glauber["sigma_NN"])
    params = GlauberParams(
        sqrtsNN=sqrtsNN_val, n_pp=float(n_pp_fit), k=float(k_fit), x=x_fit,
        sigma_NN=glauber["sigma_NN"], d=d_fit, d_norm=d_norm,
    )
    diagnostics = {
        "chi2": float(res.fun),
        "chi2_grid_best": best["chi2"],
        "edges": edges,
        "target": target,
        "nfev": int(res.nfev),
        "fit_mult_min": fit_mult_min,
    }
    return params, diagnostics
