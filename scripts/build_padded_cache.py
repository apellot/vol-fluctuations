"""Build the lab-frame, charged-only, per-event padded training cache.

Per the 2026-05-17 design lock:
  * boost particles from CM (SMASH default) to the lab/target-rest frame
  * keep only charged particles with 0 < η_lab < 2
  * per-particle features stored: pT, η_lab, φ, charge   (no PDG, no mass — strict "no PID")
  * per-event features computed and stored:
        b, sqrtsNN, energy_id, length (= n_kept particles),
        mult_lab, mean_pT_lab, total_pT_lab,
        centrality_bin (truth-tuned, on mult_lab — computed in a later script)

Boost: in a fixed-target geometry the lab/target-rest frame moves at rapidity
y_cm→lab = arccosh(√sNN / 2 m_N) relative to the CM frame SMASH outputs in.
Particle 4-momenta are boosted by (γ, β) = (cosh, tanh) of that rapidity along the
beam (z) axis. η_lab is then derived from the boosted 3-momentum.

Schema (single concatenated HDF5):
    cont         (N_events, MAX_PARTICLES, N_CONT)    float32
    length       (N_events,)                          int32
    b            (N_events,)                          float32
    sqrtsNN      (N_events,)                          float32
    energy_id    (N_events,)                          int8
    mult_lab     (N_events,)                          int32
    mean_pT_lab  (N_events,)                          float32
    total_pT_lab (N_events,)                          float32

Centrality_bin is intentionally NOT written here — the truth baseline computes
its bins from mult_lab, and that output is what the model training reads as the
classification target.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

# Per-particle continuous feature names; the model code reads N_CONT from here.
CONT_FEATURE_NAMES = ["pT", "eta_lab", "phi", "charge"]
N_CONT_FEATURES = len(CONT_FEATURE_NAMES)

# Cap on per-event particle count after the (charged ∧ 0<η_lab<2) filter.
# Empirically the maximum is well below ~500 (verified at build time); 512 gives
# a comfortable safety margin without wasting much disk.
MAX_PARTICLES = 512

# Nucleon mass used in the boost rapidity calculation.
M_N = 0.938  # GeV

# η_lab acceptance window — STAR FXT TPC acceptance per Kimelman thesis Ch. 6
# (p. 47): "in fixed-target mode, the pseudorapidity (η) acceptance is limited
# to 0 ≤ η ≤ 1.5". Earlier choice of 0–2 included particles outside STAR's
# physical acceptance and inflated the spectator floor.
ETA_LAB_MIN, ETA_LAB_MAX = 0.0, 1.5


def y_cm_to_lab(sqrtsNN: float) -> float:
    """Rapidity of the lab/target-rest frame relative to the CM frame, fixed target."""
    return float(np.arccosh(sqrtsNN / (2.0 * M_N)))


def boost_to_lab(p0: np.ndarray, pz: np.ndarray, y_boost: float) -> tuple[np.ndarray, np.ndarray]:
    """Longitudinal boost of (p0, pz) by rapidity y_boost. Returns (p0_lab, pz_lab).

    Standard Lorentz boost along z: pz' = γ(pz + β p0), p0' = γ(p0 + β pz).
    With (γ, β) = (cosh y, tanh y), this is equivalent to (rapidity y) → (y + y_boost)
    for a particle's longitudinal rapidity, while p_T is invariant.
    """
    gamma = np.cosh(y_boost)
    beta = np.tanh(y_boost)
    pz_lab = gamma * (pz + beta * p0)
    p0_lab = gamma * (p0 + beta * pz)
    return p0_lab, pz_lab


def eta_from_pz(pT: np.ndarray, pz_lab: np.ndarray) -> np.ndarray:
    """Pseudorapidity from (pT, pz). Small ε guards the log near the beam axis."""
    p = np.sqrt(pT * pT + pz_lab * pz_lab)
    eps = 1e-12
    return 0.5 * np.log((p + pz_lab + eps) / (p - pz_lab + eps))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True, nargs="+", type=Path,
                   help="Ingested SMASH HDF5 files (one per energy, in increasing-energy order ideally)")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--chunk-events", type=int, default=2000)
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Pre-pass: read N_part per event and count survivors of the N_part > 0 filter.
    # SMASH was generated with b sampled up to ~20 fm, so ~37% of events are pure
    # geometric misses (b > ~16 fm, two Au radii don't overlap, N_part = 0). These
    # are not physical collisions and would dominate the mult=0 atom in the
    # centrality observable, breaking the percentile binning. We drop them here.
    sizes = []          # surviving events per energy (after N_part > 0 filter)
    raw_sizes = []      # total events per energy (before filter, for reporting)
    sqrts_list = []
    keep_masks = []     # boolean mask per energy file, length = raw_size
    for in_path in args.inputs:
        with h5py.File(in_path, "r") as h:
            npart_arr = h["Npart"][:]
            mask = npart_arr > 0
            keep_masks.append(mask)
            raw_sizes.append(int(npart_arr.size))
            sizes.append(int(mask.sum()))
            sqrts_list.append(float(h.attrs["sqrtsNN"]))
    total = sum(sizes)
    print(f"Building lab-frame cache: keeping {total:,} of {sum(raw_sizes):,} events "
          f"(dropped {sum(raw_sizes)-total:,} N_part==0 misses) across {len(args.inputs)} energies")
    print(f"  energies: {sqrts_list}")
    print(f"  per energy kept / total: {[f'{s}/{r}' for s, r in zip(sizes, raw_sizes)]}")
    print(f"  MAX_PARTICLES = {MAX_PARTICLES}")
    print(f"  output: {args.output}")

    # Pre-allocate datasets. Per-event chunking on the cont array because the
    # dataloader reads one event per __getitem__.
    with h5py.File(args.output, "w") as out:
        cont_ds = out.create_dataset(
            "cont", shape=(total, MAX_PARTICLES, N_CONT_FEATURES),
            dtype=np.float32, chunks=(1, MAX_PARTICLES, N_CONT_FEATURES),
        )
        len_ds = out.create_dataset("length", shape=(total,), dtype=np.int32)
        b_ds = out.create_dataset("b", shape=(total,), dtype=np.float32)
        e_ds = out.create_dataset("sqrtsNN", shape=(total,), dtype=np.float32)
        eid_ds = out.create_dataset("energy_id", shape=(total,), dtype=np.int8)
        mult_ds = out.create_dataset("mult_lab", shape=(total,), dtype=np.int32)
        mpt_ds = out.create_dataset("mean_pT_lab", shape=(total,), dtype=np.float32)
        tpt_ds = out.create_dataset("total_pT_lab", shape=(total,), dtype=np.float32)

        out.attrs["max_particles"] = MAX_PARTICLES
        out.attrs["n_events_per_energy"] = sizes
        out.attrs["sqrtsNN_per_energy"] = sqrts_list
        out.attrs["feature_names"] = list(CONT_FEATURE_NAMES)
        out.attrs["eta_lab_window"] = (ETA_LAB_MIN, ETA_LAB_MAX)
        out.attrs["source_files"] = [str(p) for p in args.inputs]
        out.attrs["m_N"] = M_N

        max_seen = 0
        global_idx = 0
        for e_id, in_path in enumerate(args.inputs):
            keep_mask_energy = keep_masks[e_id]   # True where N_part > 0
            n_kept_this_energy = int(keep_mask_energy.sum())
            n_written_this_energy = 0
            with h5py.File(in_path, "r") as src:
                n_src = int(src.attrs["n_events"])
                sqrtsNN = float(src.attrs["sqrtsNN"])
                y_boost = y_cm_to_lab(sqrtsNN)
                offsets = src["offset"][:].astype(np.int64)
                b_all = src["b"][:].astype(np.float32)
                # The ingested HDF5 stores derived CM-frame kinematics, not raw
                # 4-momentum components. We reconstruct pz_CM from (pT, eta_CM)
                # via pz = pT · sinh(η), and p0 from (pT, η, mass) via the
                # mass-shell identity p₀ = √(pT² cosh²η + m²). These are exact
                # for pseudorapidity by definition (cosh η = p / pT).
                pT_all = src["particles/pT"]
                eta_cm_all = src["particles/eta"]
                phi_all = src["particles/phi"]
                mass_all = src["particles/mass"]
                charge_all = src["particles/charge"]
                pdg_all = src["particles/pdg"]
                ncoll_all = src["particles/ncoll"]

                print(f"  {sqrtsNN} GeV — y_cm→lab = {y_boost:.3f}, n_events = {n_src:,}")
                t0 = time.time()
                local_max = 0

                for chunk_start in range(0, n_src, args.chunk_events):
                    chunk_end = min(chunk_start + args.chunk_events, n_src)
                    n_chunk = chunk_end - chunk_start

                    # One big HDF5 read per branch per chunk — much faster than per-event reads.
                    p_lo = int(offsets[chunk_start])
                    p_hi = int(offsets[chunk_end])
                    pT = pT_all[p_lo:p_hi].astype(np.float32)
                    eta_cm = eta_cm_all[p_lo:p_hi].astype(np.float32)
                    phi = phi_all[p_lo:p_hi].astype(np.float32)
                    mass = mass_all[p_lo:p_hi].astype(np.float32)
                    charge = charge_all[p_lo:p_hi].astype(np.int8)
                    pdg = pdg_all[p_lo:p_hi].astype(np.int32)
                    ncoll = ncoll_all[p_lo:p_hi].astype(np.int16)

                    # Reconstruct CM-frame 4-momentum components needed for the longitudinal boost.
                    # pz = pT · sinh(η)  is exact for pseudorapidity (η ≡ ½ ln((p+pz)/(p−pz))).
                    pz_cm = pT * np.sinh(eta_cm)
                    p0_cm = np.sqrt(pT * pT * np.cosh(eta_cm) ** 2 + mass * mass)
                    p0_lab, pz_lab = boost_to_lab(p0_cm, pz_cm, y_boost)
                    eta_lab = eta_from_pz(pT, pz_lab).astype(np.float32)

                    # Selection mask: charged + in η_lab window + NOT a spectator nucleon.
                    # Spectator nucleons (target protons that never scattered) have ncoll == 0.
                    # In a real STAR FXT experiment they sit inside the target material and
                    # never reach the TPC; including them in mult_lab would create a
                    # truth-level artifact that the Glauber+NBD ansatz cannot reproduce.
                    is_nucleon = (np.abs(pdg) == 2112) | (np.abs(pdg) == 2212)
                    is_spectator = is_nucleon & (ncoll == 0)
                    in_acc = (charge != 0) & (eta_lab > ETA_LAB_MIN) & (eta_lab < ETA_LAB_MAX) & (~is_spectator)

                    # Pack each event's surviving particles into the padded output.
                    cont_buf = np.zeros((n_chunk, MAX_PARTICLES, N_CONT_FEATURES), dtype=np.float32)
                    len_buf = np.zeros(n_chunk, dtype=np.int32)
                    mult_buf = np.zeros(n_chunk, dtype=np.int32)
                    mpt_buf = np.zeros(n_chunk, dtype=np.float32)
                    tpt_buf = np.zeros(n_chunk, dtype=np.float32)

                    for i, ev in enumerate(range(chunk_start, chunk_end)):
                        s = int(offsets[ev]) - p_lo
                        e = int(offsets[ev + 1]) - p_lo
                        sel = in_acc[s:e]
                        n_kept = int(sel.sum())
                        if n_kept > MAX_PARTICLES:
                            # Defensive: should not trigger at the empirical cap of 512.
                            # If it ever does, keep the highest-pT particles (least info lost).
                            pt_ev = pT[s:e][sel]
                            order = np.argsort(-pt_ev)[:MAX_PARTICLES]
                            keep_local = np.zeros(sel.sum(), dtype=bool)
                            keep_local[order] = True
                            # Build a mask in the (s:e) frame that combines acc + top-pT cap.
                            full_keep = np.zeros(e - s, dtype=bool)
                            full_keep[sel] = keep_local
                            sel = full_keep
                            n_kept = MAX_PARTICLES
                        local_max = max(local_max, n_kept)

                        cont_buf[i, :n_kept, 0] = pT[s:e][sel]
                        cont_buf[i, :n_kept, 1] = eta_lab[s:e][sel]
                        cont_buf[i, :n_kept, 2] = phi[s:e][sel]
                        cont_buf[i, :n_kept, 3] = charge[s:e][sel].astype(np.float32)
                        len_buf[i] = n_kept
                        # Event-level features computed from the in-acceptance set.
                        mult_buf[i] = n_kept
                        if n_kept > 0:
                            ev_pt = pT[s:e][sel]
                            tpt_buf[i] = float(ev_pt.sum())
                            mpt_buf[i] = float(ev_pt.mean())

                    # Compact: only events with N_part > 0 are written to the cache.
                    chunk_keep = keep_mask_energy[chunk_start:chunk_end]
                    n_kept_in_chunk = int(chunk_keep.sum())
                    if n_kept_in_chunk == 0:
                        continue
                    g_start = global_idx + n_written_this_energy
                    g_end = g_start + n_kept_in_chunk
                    cont_ds[g_start:g_end] = cont_buf[chunk_keep]
                    len_ds[g_start:g_end] = len_buf[chunk_keep]
                    b_ds[g_start:g_end] = b_all[chunk_start:chunk_end][chunk_keep]
                    e_ds[g_start:g_end] = sqrtsNN
                    eid_ds[g_start:g_end] = e_id
                    mult_ds[g_start:g_end] = mult_buf[chunk_keep]
                    mpt_ds[g_start:g_end] = mpt_buf[chunk_keep]
                    tpt_ds[g_start:g_end] = tpt_buf[chunk_keep]
                    n_written_this_energy += n_kept_in_chunk

                global_idx += n_kept_this_energy
                max_seen = max(max_seen, local_max)
                elapsed = time.time() - t0
                print(f"    done in {elapsed:.1f} s, kept {n_kept_this_energy:,}/{n_src:,} events, "
                      f"max in-acc particles = {local_max}")

        out.attrs["max_observed_particles"] = max_seen

    print(f"\nMax observed in-acc particles across all energies: {max_seen} (cap {MAX_PARTICLES})")
    out_size_mb = args.output.stat().st_size / 1024**2
    print(f"Wrote {args.output}  ({out_size_mb:,.0f} MB)")


if __name__ == "__main__":
    main()
