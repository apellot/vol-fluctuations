"""Build the lab-frame, charged-only, per-event padded UrQMD training cache.

Mirror of `scripts/build_padded_cache.py` (the SMASH builder), reading from the
ingested UrQMD HDF5 files in `data/raw/urqmd_auau_*GeV.h5`. Same cuts (imported
from `src/data/cuts.py`), same per-particle features, same per-event scalar
features, same output schema — so SMASH and UrQMD caches are interchangeable
inputs to the architectures.

UrQMD specifics:
  * Particles in a flat group with `offset` of length n_events+1; particles
    for event i live at `particles/*[offset[i]:offset[i+1]]`.
  * The file stores (pT, η, φ, mass), NOT (p0, px, py, pz). We reconstruct
    (p0_CM, pz_CM) from these and then boost — same path the SMASH builder
    already uses (the SMASH ingest also strips raw 4-momentum down to derived
    kinematics).
  * Npart is read from the raw file's stored Glauber-header `Npart` field
    (UrQMD 4.0 `Participants_Glauber`), which is always ≥0. This is the
    2026-05-28 restructure decision: with UrQMD as the primary generator we
    use its native participant count directly, since the cross-transport
    SMASH-style spectator recompute (`cuts.npart_from_spectators`) only mattered
    for joint SMASH+UrQMD training and produced a negative peripheral tail that
    had forced the old b<11 cut. See docs/restructure_urqmd_detector_b14.md
    (supersedes the recompute recommendation in docs/npart_reconciliation.md).
    CAVEAT: confirm the nucleus/A behind this header before reporting it as a
    physical Au participant count — native max ≈ 411 hints at an A=208 (Pb)
    Glauber. `b` (the headline target) is unaffected either way.
  * UrQMD's pdg field tags only nucleons/pions/kaons/η; Λ/Σ/Ξ/Δ/N* are pdg=0.
    This DOES NOT affect spectator removal, which only looks at nucleons.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.cuts import (  # noqa: E402
    B_MAX_FM,
    DEFAULT_DETECTOR_SEED,
    EFF_PLATEAU,
    EFF_PT50,
    EFF_WIDTH,
    ETA_LAB_MAX,
    ETA_LAB_MIN,
    PT_MIN_GEV,
    PT_SMEAR_CONST,
    PT_SMEAR_SLOPE,
    boost_cm_to_lab,
    detector_keep_mask,
    eta_from_pz,
    event_passes,
    particle_keep_mask,
    smear_pt,
    y_cm_to_lab,
)

CONT_FEATURE_NAMES = ["pT", "eta_lab", "phi", "charge"]
N_CONT_FEATURES = len(CONT_FEATURE_NAMES)

# Same caps as the SMASH builder so the two caches use identical fixed-shape
# tensors and a downstream model sees the same input shape in each mode.
MAX_PARTICLES_TRUTH = 1024
MAX_PARTICLES_DETECTOR = 384


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True, nargs="+", type=Path,
                   help="Ingested UrQMD HDF5 files (one per energy, in increasing-energy order ideally)")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--chunk-events", type=int, default=2000)
    p.add_argument("--detector", action="store_true",
                   help="Apply Phase-B detector emulation (η window + pT smear + pT_min + Bernoulli ε)")
    p.add_argument("--seed", type=int, default=DEFAULT_DETECTOR_SEED,
                   help="RNG seed for detector smearing + efficiency draws (only used with --detector)")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    MAX_PARTICLES = MAX_PARTICLES_DETECTOR if args.detector else MAX_PARTICLES_TRUTH
    rng = np.random.default_rng(args.seed) if args.detector else None

    # Pre-pass: event filter (b<B_MAX_FM AND N_part>0).  Npart is the raw file's
    # native Glauber-header value (always ≥0); both the filter and the stored
    # regression target use that same native convention.
    keep_masks = []
    npart_native_all = []   # one array per input file, full event count
    raw_sizes = []
    sizes = []
    sqrts_list = []
    for in_path in args.inputs:
        with h5py.File(in_path, "r") as h:
            b_arr = h["b"][:]
            npart_arr = h["Npart"][:].astype(np.int16)
            mask = event_passes(b_arr, npart_arr)
            keep_masks.append(mask)
            npart_native_all.append(npart_arr)
            raw_sizes.append(int(npart_arr.size))
            sizes.append(int(mask.sum()))
            # sqrtsNN is per-event in the UrQMD files; take it from attrs which
            # holds the single value the file was generated at.
            sqrts_list.append(float(h.attrs["sqrtsNN"]))
    total = sum(sizes)
    mode = "DETECTOR-emulated" if args.detector else "truth-level"
    print(f"Building UrQMD {mode} lab-frame cache: keeping {total:,} of {sum(raw_sizes):,} events")
    print(f"  cuts applied: b < {B_MAX_FM} fm AND N_part > 0")
    if args.detector:
        print(f"  detector: {ETA_LAB_MIN} < η_lab < {ETA_LAB_MAX}, pT_smeared > {PT_MIN_GEV} GeV/c, "
              f"σ(pT)/pT = {PT_SMEAR_CONST} + {PT_SMEAR_SLOPE}·pT, "
              f"ε plateau {EFF_PLATEAU} at pT50={EFF_PT50}±{EFF_WIDTH} GeV/c, seed={args.seed}")
    print(f"  energies: {sqrts_list}")
    print(f"  per energy kept / total: {[f'{s}/{r}' for s, r in zip(sizes, raw_sizes)]}")
    print(f"  MAX_PARTICLES = {MAX_PARTICLES}")
    print(f"  output: {args.output}")

    report = []

    with h5py.File(args.output, "w") as out:
        cont_ds = out.create_dataset(
            "cont", shape=(total, MAX_PARTICLES, N_CONT_FEATURES),
            dtype=np.float32, chunks=(1, MAX_PARTICLES, N_CONT_FEATURES),
        )
        len_ds = out.create_dataset("length", shape=(total,), dtype=np.int32)
        b_ds = out.create_dataset("b", shape=(total,), dtype=np.float32)
        npart_ds = out.create_dataset("Npart", shape=(total,), dtype=np.int16)
        e_ds = out.create_dataset("sqrtsNN", shape=(total,), dtype=np.float32)
        eid_ds = out.create_dataset("energy_id", shape=(total,), dtype=np.int8)
        mult_ds = out.create_dataset("mult_lab", shape=(total,), dtype=np.int32)
        mpt_ds = out.create_dataset("mean_pT_lab", shape=(total,), dtype=np.float32)
        tpt_ds = out.create_dataset("total_pT_lab", shape=(total,), dtype=np.float32)

        out.attrs["max_particles"] = MAX_PARTICLES
        out.attrs["n_events_per_energy"] = sizes
        out.attrs["sqrtsNN_per_energy"] = sqrts_list
        out.attrs["feature_names"] = list(CONT_FEATURE_NAMES)
        out.attrs["source_files"] = [str(p) for p in args.inputs]
        out.attrs["b_max_fm"] = B_MAX_FM
        out.attrs["generator"] = "UrQMD"
        out.attrs["npart_source"] = "urqmd_glauber_header"
        out.attrs["cuts_module"] = "src/data/cuts.py"
        out.attrs["chunk_events"] = args.chunk_events
        out.attrs["detector_emulation"] = bool(args.detector)
        if args.detector:
            out.attrs["detector_seed"] = args.seed
            out.attrs["detector_eta_min"] = ETA_LAB_MIN
            out.attrs["detector_eta_max"] = ETA_LAB_MAX
            out.attrs["detector_pT_min"] = PT_MIN_GEV
            out.attrs["detector_smear_const"] = PT_SMEAR_CONST
            out.attrs["detector_smear_slope"] = PT_SMEAR_SLOPE
            out.attrs["detector_eff_plateau"] = EFF_PLATEAU
            out.attrs["detector_eff_pT50"] = EFF_PT50
            out.attrs["detector_eff_width"] = EFF_WIDTH

        max_seen = 0
        global_idx = 0
        for e_id, in_path in enumerate(args.inputs):
            keep_mask_energy = keep_masks[e_id]
            n_kept_this_energy = int(keep_mask_energy.sum())
            n_written_this_energy = 0
            with h5py.File(in_path, "r") as src:
                n_src = int(src.attrs["n_events"])
                sqrtsNN = float(src.attrs["sqrtsNN"])
                y_boost = y_cm_to_lab(sqrtsNN)
                offsets = src["offset"][:].astype(np.int64)
                b_all = src["b"][:].astype(np.float32)
                # Native Glauber-header Npart, read in the pre-pass.
                npart_all = npart_native_all[e_id]

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
                mult_sum = 0.0
                npart_sum = 0.0

                for chunk_start in range(0, n_src, args.chunk_events):
                    chunk_end = min(chunk_start + args.chunk_events, n_src)
                    n_chunk = chunk_end - chunk_start

                    p_lo = int(offsets[chunk_start])
                    p_hi = int(offsets[chunk_end])
                    pT = pT_all[p_lo:p_hi].astype(np.float32)
                    eta_cm = eta_cm_all[p_lo:p_hi].astype(np.float32)
                    phi = phi_all[p_lo:p_hi].astype(np.float32)
                    mass = mass_all[p_lo:p_hi].astype(np.float32)
                    charge = charge_all[p_lo:p_hi].astype(np.int8)
                    pdg = pdg_all[p_lo:p_hi].astype(np.int32)
                    ncoll = ncoll_all[p_lo:p_hi].astype(np.int16)

                    # Reconstruct CM-frame (p0, pz) — same identity as in the
                    # SMASH builder (UrQMD ingest only stores derived kinematics).
                    # A handful of UrQMD particles have sub-µeV pT (effectively
                    # spectators) which gives inf in sinh(η); we silence the
                    # transient RuntimeWarnings and drop them via np.isfinite()
                    # in the particle filter below.
                    with np.errstate(invalid="ignore", over="ignore"):
                        pz_cm = pT * np.sinh(eta_cm)
                        p0_cm = np.sqrt(pT * pT * np.cosh(eta_cm) ** 2 + mass * mass)
                        p0_lab, pz_lab = boost_cm_to_lab(p0_cm, pz_cm, sqrtsNN)
                        eta_lab = eta_from_pz(pT, pz_lab).astype(np.float32)

                    # Locked particle filter — same module as SMASH builder.
                    # AND require finite η_lab — a few hundred UrQMD particles
                    # have sub-µeV pT (mostly spectator-like nucleons missed by
                    # the ncoll==0 cut); those produce inf in η. We drop them
                    # here to keep the cache numerically clean.
                    keep_p = particle_keep_mask(charge, pdg, ncoll) & np.isfinite(eta_lab)

                    # Phase-B detector emulation (same as SMASH builder).
                    if args.detector:
                        pT_smeared = smear_pt(pT, rng)
                        keep_p = keep_p & detector_keep_mask(pT_smeared, eta_lab, rng)
                        pT_for_write = pT_smeared
                    else:
                        pT_for_write = pT

                    cont_buf = np.zeros((n_chunk, MAX_PARTICLES, N_CONT_FEATURES), dtype=np.float32)
                    len_buf = np.zeros(n_chunk, dtype=np.int32)
                    mult_buf = np.zeros(n_chunk, dtype=np.int32)
                    mpt_buf = np.zeros(n_chunk, dtype=np.float32)
                    tpt_buf = np.zeros(n_chunk, dtype=np.float32)

                    for i, ev in enumerate(range(chunk_start, chunk_end)):
                        s = int(offsets[ev]) - p_lo
                        e = int(offsets[ev + 1]) - p_lo
                        sel = keep_p[s:e]
                        n_kept = int(sel.sum())
                        if n_kept > MAX_PARTICLES:
                            # Sort by the pT actually written to the cache
                            # (smeared in detector mode, truth otherwise).
                            pt_ev = pT_for_write[s:e][sel]
                            order = np.argsort(-pt_ev)[:MAX_PARTICLES]
                            keep_local = np.zeros(int(sel.sum()), dtype=bool)
                            keep_local[order] = True
                            full_keep = np.zeros(e - s, dtype=bool)
                            full_keep[sel] = keep_local
                            sel = full_keep
                            n_kept = MAX_PARTICLES
                        local_max = max(local_max, n_kept)

                        cont_buf[i, :n_kept, 0] = pT_for_write[s:e][sel]
                        cont_buf[i, :n_kept, 1] = eta_lab[s:e][sel]
                        cont_buf[i, :n_kept, 2] = phi[s:e][sel]
                        cont_buf[i, :n_kept, 3] = charge[s:e][sel].astype(np.float32)
                        len_buf[i] = n_kept
                        mult_buf[i] = n_kept
                        if n_kept > 0:
                            ev_pt = pT_for_write[s:e][sel]
                            tpt_buf[i] = float(ev_pt.sum())
                            mpt_buf[i] = float(ev_pt.mean())

                    chunk_keep = keep_mask_energy[chunk_start:chunk_end]
                    n_kept_in_chunk = int(chunk_keep.sum())
                    if n_kept_in_chunk == 0:
                        continue
                    g_start = global_idx + n_written_this_energy
                    g_end = g_start + n_kept_in_chunk
                    cont_ds[g_start:g_end] = cont_buf[chunk_keep]
                    len_ds[g_start:g_end] = len_buf[chunk_keep]
                    b_ds[g_start:g_end] = b_all[chunk_start:chunk_end][chunk_keep]
                    npart_ds[g_start:g_end] = npart_all[chunk_start:chunk_end][chunk_keep]
                    e_ds[g_start:g_end] = sqrtsNN
                    eid_ds[g_start:g_end] = e_id
                    mult_ds[g_start:g_end] = mult_buf[chunk_keep]
                    mpt_ds[g_start:g_end] = mpt_buf[chunk_keep]
                    tpt_ds[g_start:g_end] = tpt_buf[chunk_keep]
                    mult_sum += float(mult_buf[chunk_keep].sum())
                    npart_sum += float(npart_all[chunk_start:chunk_end][chunk_keep].sum())
                    n_written_this_energy += n_kept_in_chunk

                global_idx += n_kept_this_energy
                max_seen = max(max_seen, local_max)
                elapsed = time.time() - t0

                if n_kept_this_energy > 0:
                    b_kept = b_all[keep_mask_energy]
                    report.append({
                        "sqrtsNN": sqrtsNN,
                        "n_in": n_src,
                        "n_out": n_kept_this_energy,
                        "b_min": float(b_kept.min()),
                        "b_max": float(b_kept.max()),
                        "mean_Npart": npart_sum / n_kept_this_energy,
                        "mean_mult_lab": mult_sum / n_kept_this_energy,
                    })
                print(f"    done in {elapsed:.1f} s, kept {n_kept_this_energy:,}/{n_src:,} events, "
                      f"max kept particles = {local_max}")

        out.attrs["max_observed_particles"] = max_seen

    print("\n=== UrQMD cache build summary ===")
    print(f"{'sqrtsNN':>8} {'n_in':>8} {'n_out':>8} {'b_min':>7} {'b_max':>7} "
          f"{'<Npart>':>9} {'<mult_lab>':>11}")
    for r in report:
        print(f"{r['sqrtsNN']:>8.1f} {r['n_in']:>8} {r['n_out']:>8} "
              f"{r['b_min']:>7.2f} {r['b_max']:>7.2f} "
              f"{r['mean_Npart']:>9.1f} {r['mean_mult_lab']:>11.1f}")
    print(f"\nMax observed kept particles across all energies: {max_seen} (cap {MAX_PARTICLES})")
    out_size_mb = args.output.stat().st_size / 1024 ** 2
    print(f"Wrote {args.output}  ({out_size_mb:,.0f} MB)")


if __name__ == "__main__":
    main()
