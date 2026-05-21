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

centrality_bin is read from the truth baseline output files produced by
run_truth.py (one per energy, in data/processed/truth/). Pass the truth_dir
argument to load them; they are concatenated in energy order to match the
cache layout.
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
    cont:           Tensor   # (B, MAX_PARTICLES, 4) float32 — pT, eta_lab, phi, charge
    mask:           Tensor   # (B, MAX_PARTICLES) bool — True where a real particle sits
    sqrtsNN:        Tensor   # (B,) float32
    b:              Tensor   # (B,) float32
    centrality_bin: Tensor   # (B,) long


class CachedParticleDataset(Dataset):
    """Random-access dataset over the global concatenated cache."""

    def __init__(self, cache_path: Path, truth_dir: Path | None = None,
                 n_centrality_bins: int = 9):
        self.cache_path = Path(cache_path)

        with h5py.File(self.cache_path, "r") as h:
            self.n_events      = int(h["b"].shape[0])
            self.max_particles = int(h.attrs["max_particles"])
            self.energy_sizes  = list(h.attrs["n_events_per_energy"])
            sqrts_per_energy   = list(h.attrs["sqrtsNN_per_energy"])
            b_all              = h["b"][:]

        if truth_dir is not None:
            # Load centrality_bin from run_truth.py output files, one per energy.
            # Files are named truth_auau_{sqrtsNN}GeV.h5 e.g. truth_auau_3p2GeV.h5
            truth_dir = Path(truth_dir)
            bins_list = []
            for sqrtsNN in sqrts_per_energy:
                tag = f"{sqrtsNN:.1f}".replace(".", "p")
                truth_path = truth_dir / f"truth_auau_{tag}GeV.h5"
                if not truth_path.exists():
                    raise FileNotFoundError(
                        f"Truth file not found: {truth_path}\n"
                        f"Run: python scripts/run_truth.py --cache {cache_path} "
                        f"--output-dir {truth_dir}"
                    )
                with h5py.File(truth_path, "r") as th:
                    bins_list.append(th["centrality_bin"][:].astype(np.int16))
            self._centrality_bin = np.concatenate(bins_list)
            print(f"Loaded centrality_bin from truth files in {truth_dir}")
        else:
            # Fallback: compute from b quantiles on the fly (matches truth logic).
            print("No truth_dir provided — computing centrality_bin from b quantiles.")
            edges = np.quantile(b_all, np.linspace(0.0, 1.0, n_centrality_bins + 1))
            self._centrality_bin = np.clip(
                np.digitize(b_all, edges[1:-1]),
                0, n_centrality_bins - 1,
            ).astype(np.int16)

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
            "cont":           cont,
            "mask":           mask,
            "sqrtsNN":        float(self._h["sqrtsNN"][idx]),
            "b":              float(self._h["b"][idx]),
            "centrality_bin": int(self._centrality_bin[idx]),
        }


def collate_cached(batch: list[dict]) -> CachedBatch:
    """Stack the per-event fixed-shape arrays into a batch tensor."""
    return CachedBatch(
        cont=torch.from_numpy(np.stack([b["cont"] for b in batch])),
        mask=torch.from_numpy(np.stack([b["mask"] for b in batch])),
        sqrtsNN=torch.tensor([b["sqrtsNN"] for b in batch], dtype=torch.float32),
        b=torch.tensor([b["b"] for b in batch], dtype=torch.float32),
        centrality_bin=torch.tensor([b["centrality_bin"] for b in batch], dtype=torch.long),
    )