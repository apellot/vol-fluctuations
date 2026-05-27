"""Ingest UrQMD f13 ASCII final-state particle lists into the SMASH-compatible HDF5 schema.

Fixes three defects in the previous UrQMD ingestion:
  1. PDG codes: derived from (mass, charge) since UrQMD's internal ityp codes
     were never converted to PDG.  Covers the species that appear at FXT
     energies; others stored as pdg=0.
  2. Neutral particles: ALL particles kept, including n, π⁰, K⁰, K̄⁰.
     The previous script filtered charge != 0, losing half the nucleons.
  3. Spectators: particles with ncoll==0 are now stored.  They are required
     for the N_part = 2A − N_spectators derivation used by ingest_smash.py
     and compute_ncoll_labels() in train_ncoll.py.

Output HDF5 schema is identical to ingest_smash.py so all downstream scripts
(build_padded_cache.py, train_cached.py, train_ncoll.py, …) work unchanged.

UrQMD f13 format (version 3.3 / 3.4):
    Event block starts with a line containing "Nr. of testparticles:".
    The following line contains collision statistics and the impact parameter b.
    Then N particle lines follow, each with 17 space-separated values:

        col  0     r0     (fm)       freeze-out time
        col  1     rx     (fm)       x position
        col  2     ry     (fm)       y position
        col  3     rz     (fm)       z position
        col  4     p0     (GeV)      energy
        col  5     px     (GeV)      x momentum (CM frame)
        col  6     py     (GeV)      y momentum
        col  7     pz     (GeV)      z momentum
        col  8     m      (GeV)      pole mass
        col  9     ityp   (int)      UrQMD internal type code
        col 10     2*I3   (int)      2 × isospin_z
        col 11     q      (int)      electric charge
        col 12     lcl#   (int)      local participant index
        col 13     ncl    (int)      number of collisions
        col 14     form_t (fm/c)     formation time
        col 15     ID     (int)      unique particle ID
        col 16     anc    (int)      ancestor particle ID

Frame convention: UrQMD outputs momenta in the nucleon-nucleon CM frame (same
as SMASH), so build_padded_cache.py's CM→lab boost applies without changes.

Usage:
    python scripts/ingest_urqmd.py \\
        --input-dir /star/data03/scratch/apellotji/urqmd_output/3p5 \\
        --output    data/processed/urqmd_auau_3p5GeV.h5 \\
        --energy    3.5 \\
        [--pattern  "*.f13"] \\
        [--max-events 5000]

    # Re-ingest all four energies:
    for E in 3.2 3.5 3.9 4.5; do
        tag=$(echo $E | tr . p)
        python scripts/ingest_urqmd.py \\
            --input-dir /star/data03/scratch/apellotji/urqmd_output/$tag \\
            --output    data/processed/urqmd_auau_${tag}GeV.h5 \\
            --energy    $E
    done
"""

from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import (  # noqa: E402
    npart_from_event,
    pt_eta_phi,
    charged_mult_in_eta,
    count_pdg,
)

# ---------------------------------------------------------------------------
# PDG derivation from (mass, charge)
# ---------------------------------------------------------------------------

# Nucleon mass: proton 0.93827 GeV, neutron 0.93957 GeV.  We use a window
# that covers both, plus moderate smearing from UrQMD's mass-shell handling.
_NUCLEON_MASS_LO, _NUCLEON_MASS_HI = 0.930, 0.945

# Pion masses: π± 0.13957 GeV, π⁰ 0.13498 GeV.
_PION_MASS_LO, _PION_MASS_HI = 0.130, 0.145

# Kaon masses: K± 0.49368 GeV, K⁰/K̄⁰ 0.49765 GeV.
_KAON_MASS_LO, _KAON_MASS_HI = 0.488, 0.502

# Eta meson: 0.54785 GeV (neutral only; never charged).
_ETA_MASS_LO, _ETA_MASS_HI = 0.543, 0.553

# Omega meson: 0.78266 GeV.
_OMEGA_MESON_MASS_LO, _OMEGA_MESON_MASS_HI = 0.778, 0.787

# Rho meson: 0.7755 GeV (broad width; overlaps omega).
# (Not assigned a PDG here — rho can't be distinguished from omega by mass
# at this precision level; both get pdg=0.)


def mass_charge_to_pdg(mass: np.ndarray, charge: np.ndarray) -> np.ndarray:
    """Derive approximate PDG codes from pole mass and electric charge.

    Covers the particle species that dominate UrQMD final states at FXT
    energies (√sNN ≤ 5 GeV): nucleons, pions, kaons, eta.  Heavier species
    and broad resonances get pdg=0, which is sufficient for all downstream
    N_coll / N_part computations (those only need nucleon identification).

    Neutron / antineutron distinction is impossible from charge alone (both
    have q=0 at nucleon mass).  Both are assigned pdg=2112 — abs(pdg)=2112
    matches the nucleon filter in npart_from_event(), so Npart/N_coll are
    unaffected.  Antinucleon abundance at FXT energies is negligible.

    Args:
        mass:   (N,) float array — pole mass in GeV
        charge: (N,) int array  — electric charge

    Returns:
        pdg: (N,) int32 array
    """
    pdg = np.zeros(len(mass), dtype=np.int32)

    # --- Nucleons ---
    n_mask = (mass >= _NUCLEON_MASS_LO) & (mass <= _NUCLEON_MASS_HI)
    pdg[n_mask & (charge == +1)] = 2212    # proton
    pdg[n_mask & (charge == -1)] = -2212   # antiproton
    pdg[n_mask & (charge == 0)]  = 2112    # neutron (or antineutron — see docstring)

    # --- Pions ---
    pi_mask = (mass >= _PION_MASS_LO) & (mass <= _PION_MASS_HI)
    pdg[pi_mask & (charge == +1)] = 211    # π+
    pdg[pi_mask & (charge == -1)] = -211   # π-
    pdg[pi_mask & (charge == 0)]  = 111    # π⁰

    # --- Kaons ---
    k_mask = (mass >= _KAON_MASS_LO) & (mass <= _KAON_MASS_HI)
    pdg[k_mask & (charge == +1)] = 321     # K+
    pdg[k_mask & (charge == -1)] = -321    # K-
    pdg[k_mask & (charge == 0)]  = 311     # K⁰ / K̄⁰ (unsigned; abs(pdg)=311 for both)

    # --- Eta meson (neutral) ---
    eta_mask = (mass >= _ETA_MASS_LO) & (mass <= _ETA_MASS_HI) & (charge == 0)
    pdg[eta_mask] = 221                    # η

    return pdg


# ---------------------------------------------------------------------------
# f13 event parser
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent,
        ).decode().strip()
    except Exception:
        return ""


def _parse_event_header_line(line: str) -> tuple[int, int, float]:
    """Parse the UrQMD f13 event header second line.

    Line contains: Npart_proj  Npart_targ  [many collision-count fields]  b_impact
    The impact parameter is always the LAST float on the line.
    Npart_proj is column 0, Npart_targ is column 1.

    Returns:
        (Npart_proj, Npart_targ, b_fm)
    """
    tokens = line.split()
    if len(tokens) < 3:
        raise ValueError(f"Event header line has too few fields ({len(tokens)}): {line!r}")
    npart_proj = int(tokens[0])
    npart_targ = int(tokens[1])
    b_fm       = float(tokens[-1])
    return npart_proj, npart_targ, b_fm


def ingest_f13_files(
    f13_paths: list[Path],
    sqrtsNN: float,
    max_events: int | None,
) -> dict:
    """Parse UrQMD f13 files and return accumulated event-level and particle-level data.

    Returns a dict with keys:
        b, Npart_header (from f13 header),
        nparticles (all particles, incl. neutral + spectators),
        mult_eta05, mult_eta10, n_proton, n_antiproton,
        pdg, pT, eta, phi, mass, charge, ncoll, ityp (per-particle flat arrays)
    """
    # Event-level accumulators
    b_list:            list[float] = []
    npart_header_list: list[int]   = []   # Npart_proj + Npart_targ from f13 header
    npart_derived_list:list[int]   = []   # 2A - N_spec (consistent with ingest_smash.py)
    nparticles_list:   list[int]   = []
    mult_eta05_list:   list[int]   = []
    mult_eta10_list:   list[int]   = []
    nproton_list:      list[int]   = []
    nantiproton_list:  list[int]   = []

    # Per-particle flat accumulators
    pdg_flat:    list[np.ndarray] = []
    pt_flat:     list[np.ndarray] = []
    eta_flat:    list[np.ndarray] = []
    phi_flat:    list[np.ndarray] = []
    mass_flat:   list[np.ndarray] = []
    charge_flat: list[np.ndarray] = []
    ncoll_flat:  list[np.ndarray] = []
    ityp_flat:   list[np.ndarray] = []

    n_events_kept = 0
    n_events_seen = 0
    n_parse_errors = 0

    for f13_path in sorted(f13_paths):
        print(f"  reading {f13_path.name} ...", flush=True)
        with open(f13_path, "r") as fh:
            lines = fh.readlines()

        i = 0
        n_lines = len(lines)
        while i < n_lines:
            if max_events is not None and n_events_seen >= max_events:
                break

            line = lines[i]

            # --- Detect event block start ---
            if "Nr. of testparticles:" not in line:
                i += 1
                continue

            # Parse N_particles from this header line:
            # "Nr. of testparticles:     1     Nr. of particles in event: <N>"
            try:
                n_part_in_event = int(line.split("Nr. of particles in event:")[-1].strip())
            except (ValueError, IndexError) as exc:
                print(f"    WARNING: could not parse event count on line {i}: {exc}", flush=True)
                i += 1
                continue

            # Next line: collision stats + b
            i += 1
            if i >= n_lines:
                break
            try:
                npart_proj, npart_targ, b_fm = _parse_event_header_line(lines[i])
            except ValueError as exc:
                print(f"    WARNING: bad header stats line {i}: {exc}", flush=True)
                i += n_part_in_event + 1
                n_events_seen += 1
                n_parse_errors += 1
                continue

            # Read N particle lines
            i += 1
            particle_block_start = i
            particle_block_end   = i + n_part_in_event

            if particle_block_end > n_lines:
                print(
                    f"    WARNING: event at line {particle_block_start} claims {n_part_in_event} "
                    f"particles but file ends at line {n_lines}. Truncating."
                )
                particle_block_end = n_lines

            # Parse all particle lines in this event as a 2-D float array.
            # Expected shape: (n_part_in_event, 17).
            try:
                rows = []
                for ln in lines[particle_block_start:particle_block_end]:
                    rows.append(ln.split())
                if not rows:
                    arr = np.empty((0, 17), dtype=np.float64)
                else:
                    arr = np.array(rows, dtype=np.float64)
            except ValueError as exc:
                print(f"    WARNING: could not parse particle block at line {particle_block_start}: {exc}", flush=True)
                i = particle_block_end
                n_events_seen += 1
                n_parse_errors += 1
                continue

            if arr.shape[1] != 17:
                print(
                    f"    WARNING: expected 17 columns, got {arr.shape[1]} "
                    f"at line {particle_block_start}. Skipping event."
                )
                i = particle_block_end
                n_events_seen += 1
                n_parse_errors += 1
                continue

            # --- Extract columns ---
            # Positions (not stored in HDF5 but used for nothing; skip)
            # p0, px, py, pz
            p0 = arr[:, 4].astype(np.float32)  # noqa: F841 (not stored)
            px = arr[:, 5].astype(np.float32)
            py = arr[:, 6].astype(np.float32)
            pz = arr[:, 7].astype(np.float32)
            m  = arr[:, 8].astype(np.float32)
            ityp = arr[:, 9].astype(np.int32)
            # col 10: 2*I3 — not stored
            q    = arr[:, 11].astype(np.int8)
            # col 12: lcl# — not stored
            ncl  = arr[:, 13].astype(np.int16)
            # cols 14-16 not stored

            pT_ev, eta_ev, phi_ev = pt_eta_phi(px, py, pz)
            pdg_ev = mass_charge_to_pdg(m, q)

            # --- Derived event-level quantities ---
            npart_derived = npart_from_event(pdg_ev, ncl.astype(np.int32))

            # --- Accumulate ---
            b_list.append(b_fm)
            npart_header_list.append(npart_proj + npart_targ)
            npart_derived_list.append(npart_derived)
            nparticles_list.append(len(arr))
            mult_eta05_list.append(charged_mult_in_eta(eta_ev, q, 0.5))
            mult_eta10_list.append(charged_mult_in_eta(eta_ev, q, 1.0))
            nproton_list.append(count_pdg(pdg_ev, 2212))
            nantiproton_list.append(count_pdg(pdg_ev, -2212))

            pdg_flat.append(pdg_ev)
            pt_flat.append(pT_ev)
            eta_flat.append(eta_ev)
            phi_flat.append(phi_ev)
            mass_flat.append(m)
            charge_flat.append(q)
            ncoll_flat.append(ncl)
            ityp_flat.append(ityp)

            i = particle_block_end
            n_events_kept += 1
            n_events_seen += 1

        if max_events is not None and n_events_seen >= max_events:
            break

    print(
        f"  parsed {n_events_kept:,} events "
        f"({n_parse_errors} skipped due to parse errors)"
    )
    return {
        "b":               np.asarray(b_list, dtype=np.float32),
        "Npart_header":    np.asarray(npart_header_list, dtype=np.int16),
        "Npart_derived":   np.asarray(npart_derived_list, dtype=np.int16),
        "nparticles":      np.asarray(nparticles_list, dtype=np.int32),
        "mult_eta05":      np.asarray(mult_eta05_list, dtype=np.int32),
        "mult_eta10":      np.asarray(mult_eta10_list, dtype=np.int32),
        "n_proton":        np.asarray(nproton_list, dtype=np.int32),
        "n_antiproton":    np.asarray(nantiproton_list, dtype=np.int32),
        "pdg":             np.concatenate(pdg_flat) if pdg_flat else np.empty(0, np.int32),
        "pT":              np.concatenate(pt_flat)  if pt_flat  else np.empty(0, np.float32),
        "eta":             np.concatenate(eta_flat) if eta_flat else np.empty(0, np.float32),
        "phi":             np.concatenate(phi_flat) if phi_flat else np.empty(0, np.float32),
        "mass":            np.concatenate(mass_flat)   if mass_flat   else np.empty(0, np.float32),
        "charge":          np.concatenate(charge_flat) if charge_flat else np.empty(0, np.int8),
        "ncoll":           np.concatenate(ncoll_flat)  if ncoll_flat  else np.empty(0, np.int16),
        "ityp":            np.concatenate(ityp_flat)   if ityp_flat   else np.empty(0, np.int32),
    }


# ---------------------------------------------------------------------------
# Write HDF5
# ---------------------------------------------------------------------------

def write_hdf5(out_path: Path, data: dict, sqrtsNN: float, input_dir: Path) -> None:
    n_events = len(data["b"])
    counts   = data["nparticles"]
    offsets  = np.concatenate([[0], np.cumsum(counts.astype(np.int64))])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h:
        # --- Event-level scalars ---
        h.create_dataset("sqrtsNN",      data=np.full(n_events, sqrtsNN, dtype=np.float32))
        h.create_dataset("b",            data=data["b"])
        # Primary Npart — derived from spectator count, consistent with SMASH convention.
        h.create_dataset("Npart",        data=data["Npart_derived"])
        # Also store the header value as a cross-check.
        h.create_dataset("Npart_header", data=data["Npart_header"])
        h.create_dataset("nparticles",   data=counts)
        h.create_dataset("offset",       data=offsets)
        h.create_dataset("mult_eta05",   data=data["mult_eta05"])
        h.create_dataset("mult_eta10",   data=data["mult_eta10"])
        h.create_dataset("n_proton",     data=data["n_proton"])
        h.create_dataset("n_antiproton", data=data["n_antiproton"])

        # --- Per-particle flat datasets ---
        g = h.create_group("particles")
        g.create_dataset("pdg",    data=data["pdg"])
        g.create_dataset("pT",     data=data["pT"])
        g.create_dataset("eta",    data=data["eta"])
        g.create_dataset("phi",    data=data["phi"])
        g.create_dataset("mass",   data=data["mass"])
        g.create_dataset("charge", data=data["charge"])
        g.create_dataset("ncoll",  data=data["ncoll"])
        # ityp stored for traceability; not used by downstream scripts.
        g.create_dataset("ityp",   data=data["ityp"])

        # --- Provenance ---
        h.attrs["sqrtsNN"]          = sqrtsNN
        h.attrs["n_events"]         = n_events
        h.attrs["source"]           = "urqmd_f13"
        h.attrs["input_dir"]        = str(input_dir)
        h.attrs["ingest_script"]    = "scripts/ingest_urqmd.py"
        h.attrs["ingest_commit"]    = _git_commit()
        h.attrs["ingest_iso8601"]   = _dt.datetime.now(_dt.timezone.utc).isoformat()
        h.attrs["pdg_note"]         = (
            "PDG codes derived from (mass, charge) via mass_charge_to_pdg(). "
            "Nucleons, pions, kaons, and eta assigned; other species pdg=0. "
            "Neutral particles (n, pi0, K0) and spectators (ncoll==0) ARE included."
        )
        h.attrs["npart_note"]       = (
            "Npart = 2*197 - N_spectators, where spectator = "
            "(abs(pdg) in {2112,2212}) AND (ncoll==0). "
            "Npart_header = Npart_proj + Npart_targ from f13 event header (cross-check)."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-dir", required=True, type=Path,
        help="Directory containing UrQMD f13 output files for one energy",
    )
    p.add_argument(
        "--output", required=True, type=Path,
        help="Output HDF5 file (e.g. data/processed/urqmd_auau_3p5GeV.h5)",
    )
    p.add_argument(
        "--energy", required=True, type=float,
        help="√sNN in GeV (e.g. 3.5)",
    )
    p.add_argument(
        "--pattern", default="*.f13",
        help="Glob pattern for f13 files inside --input-dir (default: *.f13)",
    )
    p.add_argument(
        "--max-events", type=int, default=None,
        help="Stop after this many events (for testing)",
    )
    args = p.parse_args()

    input_dir = args.input_dir.expanduser()
    if not input_dir.is_dir():
        sys.exit(f"ERROR: --input-dir not found: {input_dir}")

    f13_paths = sorted(input_dir.glob(args.pattern))
    if not f13_paths:
        sys.exit(
            f"ERROR: no files matching '{args.pattern}' in {input_dir}.\n"
            f"       Use --pattern to specify the correct glob (e.g. '*.dat', '*.oscar')."
        )

    print(
        f"Ingesting UrQMD f13 files — √sNN={args.energy} GeV — "
        f"{len(f13_paths)} file(s) in {input_dir}"
    )

    data = ingest_f13_files(f13_paths, sqrtsNN=args.energy, max_events=args.max_events)

    n_events = len(data["b"])
    if n_events == 0:
        sys.exit("ERROR: no events were parsed. Check --pattern and file format.")

    # --- Sanity report ---
    npart_d = data["Npart_derived"]
    npart_h = data["Npart_header"]
    print(f"\nEvent-level sanity check ({n_events:,} events):")
    print(f"  b:              min={data['b'].min():.2f}  max={data['b'].max():.2f}  mean={data['b'].mean():.2f} fm")
    print(f"  Npart_derived:  min={npart_d.min()}  max={npart_d.max()}  mean={npart_d.mean():.1f}")
    print(f"  Npart_header:   min={npart_h.min()}  max={npart_h.max()}  mean={npart_h.mean():.1f}")
    mismatch = np.abs(npart_d.astype(int) - npart_h.astype(int))
    print(f"  |Npart_derived − Npart_header|: mean={mismatch.mean():.1f}  max={mismatch.max()}")
    if mismatch.mean() > 20:
        print(
            "  WARNING: large mean Npart discrepancy. This can indicate a PDG derivation "
            "issue (e.g. nucleon mass window too narrow) or a format mismatch in the "
            "header parsing. Check --pattern and the first few lines of an f13 file."
        )

    total_particles = int(data["nparticles"].sum())
    pdg = data["pdg"]
    n_nucleon  = int(np.sum((np.abs(pdg) == 2212) | (np.abs(pdg) == 2112)))
    n_pion     = int(np.sum((np.abs(pdg) == 211)  | (pdg == 111)))
    n_kaon     = int(np.sum((np.abs(pdg) == 321)  | (np.abs(pdg) == 311)))
    n_unassigned = int(np.sum(pdg == 0))
    print(f"\nParticle-level sanity check ({total_particles:,} total particles):")
    print(f"  nucleons (pdg ∈ {{2112,2212}}): {n_nucleon:,}  ({100*n_nucleon/total_particles:.1f}%)")
    print(f"  pions    (|pdg| ∈ {{111,211}}): {n_pion:,}  ({100*n_pion/total_particles:.1f}%)")
    print(f"  kaons    (|pdg| ∈ {{311,321}}): {n_kaon:,}  ({100*n_kaon/total_particles:.1f}%)")
    print(f"  unassigned (pdg=0):             {n_unassigned:,}  ({100*n_unassigned/total_particles:.1f}%)")
    charge = data["charge"]
    print(f"  neutral particles (charge=0):   {int((charge==0).sum()):,}")
    ncoll_arr = data["ncoll"]
    print(f"  spectators (ncoll=0):           {int((ncoll_arr==0).sum()):,}")
    if int((ncoll_arr==0).sum()) == 0:
        print(
            "  WARNING: no spectators found (ncoll=0 particles). "
            "This will cause Npart_derived=394 for all events. "
            "The f13 files may only contain interacting particles — "
            "check whether spectators are written to a separate UrQMD output file."
        )

    print(f"\nWriting {args.output} ...")
    write_hdf5(args.output.expanduser(), data, sqrtsNN=args.energy, input_dir=input_dir)
    out_mb = args.output.expanduser().stat().st_size / 1024**2
    print(f"Done. Wrote {args.output} ({out_mb:,.0f} MB, {n_events:,} events, {total_particles:,} particles)")


if __name__ == "__main__":
    main()
