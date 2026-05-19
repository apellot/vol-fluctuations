"""PyTorch Dataset wrapping the lab-frame, charged-only padded cache.

Built for the design locked on 2026-05-17:
  * fixed-shape per-event tensor cont[MAX_PARTICLES, 4]
  * length[i] gives the number of real (non-padding) particles in event i
  * event-level features pre-computed: sqrtsNN, mult_lab, mean_pT_lab, total_pT_lab
  * centrality labels read from a sibling truth-baseline prediction file

The DataLoader collate function just stacks per-event tensors — no padding work
at batch time. Each worker opens its own h5py handle lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


# Order of the per-event scalar features fed to the model after pooling.
EVENT_FEATURE_NAMES = ["sqrtsNN", "mult_lab", "mean_pT_lab", "total_pT_lab"]


@dataclass
class CachedBatch:
    cont: Tensor          # (B, L_max, 4) float32 — pT, eta_lab, phi, charge
    mask: Tensor          # (B, L_max) bool
    event_feats: Tensor   # (B, 4) float32 — event-level features in EVENT_FEATURE_NAMES order
    b: Tensor             # (B,) float32 — truth impact parameter
    centrality_bin: Tensor  # (B,) long — supervisor label
    energy_id: Tensor     # (B,) long — for diagnostic per-energy reporting


class LabCachedDataset(Dataset):
    """Reads the lab-frame cache + per-energy truth-baseline prediction files for labels.

    The cache holds concatenated events from 4 energies. The truth-baseline outputs
    are *per-energy* files; we stitch their centrality_bin arrays back into the
    cache's global event ordering using the n_events_per_energy attribute.
    """

    def __init__(self, cache_path: Path, truth_pred_paths: list[Path]) -> None:
        self.cache_path = Path(cache_path)
        with h5py.File(self.cache_path, "r") as h:
            self.n_events = int(h["b"].shape[0])
            self.max_particles = int(h.attrs["max_particles"])
            self.energy_sizes = list(h.attrs["n_events_per_energy"])
            self.sqrts_per_energy = list(h.attrs["sqrtsNN_per_energy"])

        if len(truth_pred_paths) != len(self.energy_sizes):
            raise ValueError("Need one truth-prediction file per energy in the cache")

        # Stitch centrality labels into a single flat array in cache order.
        labels = np.empty(self.n_events, dtype=np.int64)
        cum = np.cumsum([0] + self.energy_sizes)
        for i, p in enumerate(truth_pred_paths):
            with h5py.File(p, "r") as h:
                ce = h["centrality_bin"][:].astype(np.int64)
                if ce.shape[0] != self.energy_sizes[i]:
                    raise RuntimeError(f"length mismatch for {p}")
                labels[cum[i]:cum[i + 1]] = ce
        self.centrality_bin = labels

        self._h: h5py.File | None = None

    def _ensure_open(self) -> None:
        # Open lazily per worker process so DataLoader multiprocessing is safe.
        if self._h is None:
            self._h = h5py.File(self.cache_path, "r", swmr=True)

    def __len__(self) -> int:
        return self.n_events

    def __getitem__(self, idx: int) -> dict:
        self._ensure_open()
        n = int(self._h["length"][idx])
        cont = self._h["cont"][idx]                  # (L_max, 4)
        mask = np.zeros(self.max_particles, dtype=bool)
        mask[:n] = True
        event_feats = np.array([
            self._h["sqrtsNN"][idx],
            self._h["mult_lab"][idx],
            self._h["mean_pT_lab"][idx],
            self._h["total_pT_lab"][idx],
        ], dtype=np.float32)
        return {
            "cont": cont,
            "mask": mask,
            "event_feats": event_feats,
            "b": float(self._h["b"][idx]),
            "centrality_bin": int(self.centrality_bin[idx]),
            "energy_id": int(self._h["energy_id"][idx]),
        }


def collate_lab(batch: list[dict]) -> CachedBatch:
    return CachedBatch(
        cont=torch.from_numpy(np.stack([b["cont"] for b in batch])),
        mask=torch.from_numpy(np.stack([b["mask"] for b in batch])),
        event_feats=torch.from_numpy(np.stack([b["event_feats"] for b in batch])),
        b=torch.tensor([b["b"] for b in batch], dtype=torch.float32),
        centrality_bin=torch.tensor([b["centrality_bin"] for b in batch], dtype=torch.long),
        energy_id=torch.tensor([b["energy_id"] for b in batch], dtype=torch.long),
    )


def apply_phi_rotation(cont: Tensor, mask: Tensor, rng: torch.Generator | None = None) -> Tensor:
    """Per-batch random φ rotation augmentation.

    Adds a random Δφ (one draw per event) to the φ feature of every particle in
    that event. Wraps the result back into (−π, π] so the network's downstream
    embedding (which is sensitive to the raw float value) sees a consistent range.
    The change is a physical symmetry of the data: no preferred azimuthal
    direction in the lab for min-bias FXT, so the network should be φ-invariant.
    """
    B = cont.shape[0]
    if rng is None:
        delta = torch.rand(B, device=cont.device) * (2 * torch.pi) - torch.pi
    else:
        delta = (torch.rand(B, generator=rng) * (2 * torch.pi) - torch.pi).to(cont.device)
    cont = cont.clone()
    phi = cont[..., 2]
    new_phi = phi + delta.unsqueeze(-1)
    new_phi = (new_phi + torch.pi) % (2 * torch.pi) - torch.pi
    cont[..., 2] = new_phi * mask.float() + cont[..., 2] * (~mask).float()
    # The line above only updates masked positions; padded positions keep their (zero) value.
    return cont
