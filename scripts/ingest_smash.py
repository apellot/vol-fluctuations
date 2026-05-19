"""Convert one SMASH ROOT file into a standardized HDF5 event dataset.

Usage:
    python scripts/ingest_smash.py \
        --input  ~/phd/SMASH_Analysis/OUT/particle_lists_AuAu_3.5GeV.root \
        --output data/processed/auau_3p5GeV.h5 \
        --energy 3.5 \
        [--max-events 2000]

Schema is documented at the bottom of src/data/dataset.py.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import uproot

# Local imports — let the script be runnable from the repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import (  # noqa: E402
    charged_mult_in_eta,
    count_pdg,
    npart_from_event,
    pt_eta_phi,
)

# Branches we need from the SMASH TTree. Explicit list keeps memory bounded
# and makes the dependency surface obvious.
READ_BRANCHES = [
    "nparticles",
    "impact_param",
    "empty_event",
    "pdg",
    "charge",
    "mass",
    "px",
    "py",
    "pz",
    "ncoll",
]


def _git_commit() -> str:
    """Best-effort git HEAD short hash; empty string if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent,
        ).decode().strip()
    except Exception:
        return ""


def ingest(root_path: Path, out_path: Path, sqrtsNN: float, max_events: int | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Event-level scalar arrays — grown as Python lists and stacked at the end.
    b_list: list[float] = []
    npart_list: list[int] = []
    nparticles_list: list[int] = []
    mult_eta05_list: list[int] = []
    mult_eta10_list: list[int] = []
    nproton_list: list[int] = []
    nantiproton_list: list[int] = []

    # Per-particle flat arrays, concatenated across events.
    pdg_flat: list[np.ndarray] = []
    pt_flat: list[np.ndarray] = []
    eta_flat: list[np.ndarray] = []
    phi_flat: list[np.ndarray] = []
    mass_flat: list[np.ndarray] = []
    charge_flat: list[np.ndarray] = []
    # ncoll is per-particle collision count — needed to identify spectator
    # nucleons (ncoll == 0) so they can be excluded from the centrality
    # observable, matching the real-experiment situation where target
    # spectators stay in the target material and never reach the TPC.
    ncoll_flat: list[np.ndarray] = []

    n_events_kept = 0
    n_events_dropped_empty = 0

    # Chunked read: SMASH ROOT files are ~1 GB so we cannot load all branches at once.
    # step_size of 1000 events ≈ a few hundred MB per chunk, comfortable in memory.
    with uproot.open(str(root_path)) as f:
        tree = f["tree"]
        n_total = tree.num_entries
        if max_events is not None:
            n_total = min(n_total, max_events)

        entry_stop = n_total
        for batch in tree.iterate(READ_BRANCHES, entry_stop=entry_stop, step_size=1000, library="np"):
            n_in_batch = len(batch["nparticles"])
            for i in range(n_in_batch):
                if batch["empty_event"][i]:
                    n_events_dropped_empty += 1
                    continue

                px = batch["px"][i]
                py = batch["py"][i]
                pz = batch["pz"][i]
                pdg = batch["pdg"][i].astype(np.int32)
                chg = batch["charge"][i].astype(np.int8)
                mass = batch["mass"][i].astype(np.float32)
                ncoll = batch["ncoll"][i]

                pt, eta, phi = pt_eta_phi(px, py, pz)

                b_list.append(float(batch["impact_param"][i]))
                npart_list.append(npart_from_event(pdg, ncoll))
                nparticles_list.append(int(batch["nparticles"][i]))
                mult_eta05_list.append(charged_mult_in_eta(eta, chg, 0.5))
                mult_eta10_list.append(charged_mult_in_eta(eta, chg, 1.0))
                nproton_list.append(count_pdg(pdg, 2212))
                nantiproton_list.append(count_pdg(pdg, -2212))

                pdg_flat.append(pdg)
                pt_flat.append(pt)
                eta_flat.append(eta)
                phi_flat.append(phi)
                mass_flat.append(mass)
                charge_flat.append(chg)
                ncoll_flat.append(ncoll.astype(np.int16))

                n_events_kept += 1

    # Concatenate per-particle data and compute offsets.
    if n_events_kept == 0:
        raise RuntimeError(f"No non-empty events found in {root_path}")

    counts = np.asarray(nparticles_list, dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(counts.astype(np.int64))])

    with h5py.File(out_path, "w") as h:
        # Event-level scalars.
        h.create_dataset("sqrtsNN", data=np.full(n_events_kept, sqrtsNN, dtype=np.float32))
        h.create_dataset("b", data=np.asarray(b_list, dtype=np.float32))
        h.create_dataset("Npart", data=np.asarray(npart_list, dtype=np.int16))
        h.create_dataset("nparticles", data=counts)
        h.create_dataset("offset", data=offsets)
        h.create_dataset("mult_eta05", data=np.asarray(mult_eta05_list, dtype=np.int32))
        h.create_dataset("mult_eta10", data=np.asarray(mult_eta10_list, dtype=np.int32))
        h.create_dataset("n_proton", data=np.asarray(nproton_list, dtype=np.int32))
        h.create_dataset("n_antiproton", data=np.asarray(nantiproton_list, dtype=np.int32))

        # Per-particle flat datasets under a sub-group.
        g = h.create_group("particles")
        g.create_dataset("pdg", data=np.concatenate(pdg_flat))
        g.create_dataset("pT", data=np.concatenate(pt_flat))
        g.create_dataset("eta", data=np.concatenate(eta_flat))
        g.create_dataset("phi", data=np.concatenate(phi_flat))
        g.create_dataset("mass", data=np.concatenate(mass_flat))
        g.create_dataset("charge", data=np.concatenate(charge_flat))
        g.create_dataset("ncoll", data=np.concatenate(ncoll_flat))

        # Provenance attributes — these get checked by physics-reviewer.
        h.attrs["sqrtsNN"] = sqrtsNN
        h.attrs["n_events"] = n_events_kept
        h.attrs["n_events_dropped_empty"] = n_events_dropped_empty
        h.attrs["source_root"] = str(root_path)
        h.attrs["uproot_version"] = uproot.__version__
        h.attrs["ingest_commit"] = _git_commit()
        h.attrs["ingest_iso8601"] = _dt.datetime.now(_dt.timezone.utc).isoformat()

    print(
        f"Wrote {out_path}: {n_events_kept} events kept, "
        f"{n_events_dropped_empty} empty events dropped, "
        f"{offsets[-1]} particles total."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path, help="SMASH ROOT file")
    p.add_argument("--output", required=True, type=Path, help="Output HDF5 file")
    p.add_argument("--energy", required=True, type=float, help="√sNN in GeV")
    p.add_argument("--max-events", type=int, default=None, help="Cap events processed (for testing)")
    args = p.parse_args()
    ingest(args.input.expanduser(), args.output.expanduser(), args.energy, args.max_events)


if __name__ == "__main__":
    main()
