"""Feature-vector MLP baseline for joint centrality regression and classification.

The model has a shared trunk and two heads:

  Trunk:
    F input features  →  [Linear → ReLU → Dropout] × depth  →  hidden vector.

  Heads (shared OutputHeads, 2026-05-28): two evidential NIG heads — one for b,
    one for Npart (μ, ν, α, β each). No classification head; centrality is derived
    downstream from predicted Npart. See docs/modelling_plan.md.

Why a feature-vector MLP at all? It is the simplest ML baseline — beating it
with a per-particle permutation-invariant network is what justifies the rest of
the architectural comparison in the paper. The features should be observables
a real experiment could measure; in particular, do NOT include N_part as a
feature (it is derived from spectator information no experiment has).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .heads import OutputHeads


@dataclass
class MLPConfig:
    n_features: int          # number of scalar input features per event
    n_centrality_bins: int | None = None  # ignored (no classifier); kept for back-compat
    hidden_dim: int = 128
    depth: int = 3           # number of hidden Linear layers
    dropout: float = 0.1


class MLPHead(nn.Module):
    """Trunk + dual evidential heads (b, Npart). Returns a dict; consumes the
    event-level scalar feature vector x (B, n_features) — no per-particle list."""

    def __init__(self, cfg: MLPConfig) -> None:
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        in_dim = cfg.n_features
        for _ in range(cfg.depth):
            layers += [nn.Linear(in_dim, cfg.hidden_dim), nn.ReLU(), nn.Dropout(cfg.dropout)]
            in_dim = cfg.hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.heads = OutputHeads(cfg.hidden_dim)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        return self.heads(self.trunk(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
