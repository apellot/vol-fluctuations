"""PyTorch Dataset for variable-length particle lists from ingested SMASH HDF5 files.

Per-particle features fed to the permutation-invariant networks are
deliberately detector-feasible: kinematics (pT, η, φ, mass), charge, and a
learned embedding for the PDG code. Truth-only fields (per-particle ncoll,
proc_type_origin, mother PDGs) are excluded — including them would inflate
performance with information no experiment has.

Batching: events have variable particle counts (~50–700 at FXT). The collate
function pads to the longest sequence in the batch and returns a Boolean mask
so the model can ignore padded positions during pooling and attention.

The HDF5 schema is the one produced by scripts/ingest_smash.py — per-event
scalar columns plus a `particles/` sub-group of flat arrays indexed by `offset`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

# PDG vocabulary covering everything SMASH emits at FXT. Mapping is closed —
# unknown PDG codes are routed to OOV_INDEX so the network never silently maps
# a new species to "neutron" via hash collision.
PDG_VOCAB = {
    0:    0,   # padding token (never used as a real PDG)
    2112: 1,   # neutron
   -2112: 2,
    2212: 3,   # proton
   -2212: 4,
    211:  5,   # pi+
   -211:  6,   # pi-
    111:  7,   # pi0
    321:  8,   # K+
   -321:  9,
    311: 10,   # K0
   -311: 11,
    221: 12,   # eta
    3122: 13,  # Lambda
   -3122: 14,
    3112: 15,  # Sigma-
    3212: 16,  # Sigma0
    3222: 17,  # Sigma+
    3312: 18,  # Xi-
    22:   19,  # photon (rare; from pi0 decay)
}
OOV_INDEX = 20
PDG_EMB_VOCAB = OOV_INDEX + 1   # +1 because we reserve OOV_INDEX as a valid slot
PDG_EMB_DIM = 8                 # default embedding dimension


def pdg_to_index(pdg: np.ndarray) -> np.ndarray:
    """Map an array of PDG codes to embedding-table indices, with OOV handling."""
    # Vectorized lookup via a small dict-backed function; fast enough for batch sizes
    # well under 1M particles (which holds — batches are a few × 10⁵ at most).
    out = np.full(pdg.shape, OOV_INDEX, dtype=np.int64)
    for code, idx in PDG_VOCAB.items():
        if code == 0:
            continue
        out[pdg == code] = idx
    return out


# Continuous per-particle features in the order they appear in the model input.
CONT_FEATURE_NAMES = ["pT", "eta", "phi", "mass", "charge"]
N_CONT_FEATURES = len(CONT_FEATURE_NAMES)


@dataclass
class BatchedParticles:
    """Container returned by the collate function — keeps the call site readable."""
    cont: Tensor          # (B, L, N_CONT_FEATURES) float32
    pdg_idx: Tensor       # (B, L) long
    mask: Tensor          # (B, L) bool — True where the position holds a real particle
    sqrtsNN: Tensor       # (B,) float32 — per-event energy label
    b: Tensor             # (B,) float32 — truth impact parameter
    centrality_bin: Tensor  # (B,) long — supervisor label for the percentile head


class ParticleEventDataset(Dataset):
    """One HDF5 file per (energy, baseline-prediction) pair; one Dataset per energy.

    Multi-energy training uses ConcatDataset over per-energy instances so each one
    can manage its own h5py handle in worker processes safely.
    """

    def __init__(self, h5_path: Path, centrality_h5_path: Path):
        self.h5_path = Path(h5_path)
        self.centrality_h5_path = Path(centrality_h5_path)
        # Load small per-event arrays once at construction; the per-particle slices
        # are read lazily per __getitem__ so we never hold the full ~5 GB of
        # particle data in memory.
        with h5py.File(self.h5_path, "r") as h:
            self.sqrtsNN = float(h.attrs["sqrtsNN"])
            self.n_events = int(h.attrs["n_events"])
            self.b = h["b"][:].astype(np.float32)
            self.offsets = h["offset"][:].astype(np.int64)
        with h5py.File(self.centrality_h5_path, "r") as hp:
            self.centrality_bin = hp["centrality_bin"][:].astype(np.int64)
        if self.centrality_bin.shape[0] != self.n_events:
            raise RuntimeError("centrality and ingest event counts disagree")
        # Lazily-opened file handle per worker. Initialised in _ensure_open().
        self._h5: h5py.File | None = None
        self._particles: dict[str, h5py.Dataset] | None = None

    def _ensure_open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
            g = self._h5["particles"]
            self._particles = {
                "pT": g["pT"], "eta": g["eta"], "phi": g["phi"],
                "mass": g["mass"], "charge": g["charge"], "pdg": g["pdg"],
            }

    def __len__(self) -> int:
        return self.n_events

    def __getitem__(self, idx: int) -> dict:
        self._ensure_open()
        s, e = int(self.offsets[idx]), int(self.offsets[idx + 1])
        pT = self._particles["pT"][s:e].astype(np.float32)
        eta = self._particles["eta"][s:e].astype(np.float32)
        phi = self._particles["phi"][s:e].astype(np.float32)
        mass = self._particles["mass"][s:e].astype(np.float32)
        charge = self._particles["charge"][s:e].astype(np.float32)
        pdg = self._particles["pdg"][s:e].astype(np.int64)

        cont = np.column_stack([pT, eta, phi, mass, charge]).astype(np.float32)
        pdg_idx = pdg_to_index(pdg)

        return {
            "cont": cont,
            "pdg_idx": pdg_idx,
            "sqrtsNN": np.float32(self.sqrtsNN),
            "b": self.b[idx],
            "centrality_bin": self.centrality_bin[idx],
        }


def collate_particles(batch: list[dict]) -> BatchedParticles:
    """Pad variable-length particle lists to the batch's longest sequence."""
    lengths = [b["cont"].shape[0] for b in batch]
    L = max(lengths) if lengths else 1
    B = len(batch)

    cont = torch.zeros(B, L, N_CONT_FEATURES, dtype=torch.float32)
    pdg_idx = torch.zeros(B, L, dtype=torch.long)            # 0 = padding token in vocab
    mask = torch.zeros(B, L, dtype=torch.bool)
    sqrtsNN = torch.zeros(B, dtype=torch.float32)
    b = torch.zeros(B, dtype=torch.float32)
    cb = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["cont"].shape[0]
        cont[i, :n] = torch.from_numpy(item["cont"])
        pdg_idx[i, :n] = torch.from_numpy(item["pdg_idx"])
        mask[i, :n] = True
        sqrtsNN[i] = float(item["sqrtsNN"])
        b[i] = float(item["b"])
        cb[i] = int(item["centrality_bin"])
    return BatchedParticles(cont=cont, pdg_idx=pdg_idx, mask=mask,
                            sqrtsNN=sqrtsNN, b=b, centrality_bin=cb)
