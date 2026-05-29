"""PyTorch Dataset wrapping the pre-padded cache built by scripts/build_padded_cache.py.

Key difference vs ParticleEventDataset: every event has exactly MAX_PARTICLES
positions on disk, so __getitem__ does no padding and the collate step becomes
a trivial stack. This is what makes training fast on the M3.

The Dataset opens one h5py handle per worker (lazily, in __getitem__) so
DataLoader's multiprocessing remains safe.

Schema (current — MAX_PARTICLES=512, 4 features, charged+spectator-removed):
    cont:         (N, MAX_PARTICLES, 4)  float32  — pT, eta_lab, phi, charge
    length:       (N,)                   int32    — number of real particles
    sqrtsNN:      (N,)                   float32
    b:            (N,)                   float32
    mult_lab:     (N,)                   int32

Targets are `b` and `Npart`, both read directly from the cache (2026-05-28: the
dual-evidential models regress both; centrality is derived downstream from
predicted Npart, so there is no longer a centrality_bin / truth_dir dependency
for training — `run_truth` is only the classical baseline now).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class CachedBatch:
    cont:    Tensor   # (B, MAX_PARTICLES, 4) float32 — pT, eta_lab, phi, charge
    mask:    Tensor   # (B, MAX_PARTICLES) bool — True where a real particle sits
    sqrtsNN: Tensor   # (B,) float32
    b:       Tensor   # (B,) float32 — impact parameter target
    Npart:   Tensor   # (B,) float32 — participant target


class CachedParticleDataset(Dataset):
    """Random-access dataset over the global concatenated cache."""

    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)

        with h5py.File(self.cache_path, "r") as h:
            self.n_events      = int(h["b"].shape[0])
            self.max_particles = int(h.attrs["max_particles"])
            self.energy_sizes  = list(h.attrs["n_events_per_energy"])

        self._h: h5py.File | None = None

    def _ensure_open(self) -> None:
        if self._h is None:
            self._h = h5py.File(self.cache_path, "r", swmr=True)

    def __len__(self) -> int:
        return self.n_events

    def __getitem__(self, idx: int) -> dict:
        self._ensure_open()
        n    = int(self._h["length"][idx])
        cont = self._h["cont"][idx]
        mask = np.zeros(self.max_particles, dtype=bool)
        mask[:n] = True
        return {
            "cont":    cont,
            "mask":    mask,
            "sqrtsNN": float(self._h["sqrtsNN"][idx]),
            "b":       float(self._h["b"][idx]),
            "Npart":   float(self._h["Npart"][idx]),
        }


def collate_cached(batch: list[dict]) -> CachedBatch:
    """Stack the per-event fixed-shape arrays into a batch tensor."""
    return CachedBatch(
        cont=torch.from_numpy(np.stack([b["cont"] for b in batch])),
        mask=torch.from_numpy(np.stack([b["mask"] for b in batch])),
        sqrtsNN=torch.tensor([b["sqrtsNN"] for b in batch], dtype=torch.float32),
        b=torch.tensor([b["b"] for b in batch], dtype=torch.float32),
        Npart=torch.tensor([b["Npart"] for b in batch], dtype=torch.float32),
    )