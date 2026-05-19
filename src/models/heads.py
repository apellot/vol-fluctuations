"""Shared output heads and helpers for all four set-based architectures.

Centralising the NIG and softmax heads means the four model files only have to
implement the *aggregation* of per-particle inputs into an event embedding —
the loss and the head parameterisation stays identical across them.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

# Number of continuous per-particle features fed into every architecture.
# Order: pT, eta_lab, phi, charge.
N_CONT_FEATURES = 4

# Number of event-level scalar features concatenated after the per-particle aggregation.
# Order: sqrtsNN, mult_lab, mean_pT_lab, total_pT_lab.
N_EVENT_FEATURES = 4


def make_mlp(in_dim: int, hidden: int, depth: int, dropout: float) -> nn.Sequential:
    """Standard fully-connected stack ending in a hidden-dim activation (no output head)."""
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
        d = hidden
    return nn.Sequential(*layers)


class OutputHeads(nn.Module):
    """Linear NIG (4 outputs) + linear classification (n_bins outputs) on top of the trunk.

    The NIG raw outputs are mapped through softplus+offsets to enforce
    ν > 0, α > 1, β > 0 (μ unconstrained), matching src/losses/evidential.py.
    """
    def __init__(self, trunk_dim: int, n_centrality_bins: int) -> None:
        super().__init__()
        self.nig = nn.Linear(trunk_dim, 4)
        self.classifier = nn.Linear(trunk_dim, n_centrality_bins)

    def forward(self, h: Tensor) -> dict[str, Tensor]:
        raw = self.nig(h)
        mu = raw[..., 0]
        nu = nn.functional.softplus(raw[..., 1]) + 1e-6
        alpha = nn.functional.softplus(raw[..., 2]) + 1.0 + 1e-6
        beta = nn.functional.softplus(raw[..., 3]) + 1e-6
        logits = self.classifier(h)
        return {"mu": mu, "nu": nu, "alpha": alpha, "beta": beta, "logits": logits}


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """Mean of x over the second axis, ignoring positions where mask is False.

    x:    (B, L, F)
    mask: (B, L) bool
    out:  (B, F)
    """
    mask_f = mask.unsqueeze(-1).float()
    summed = (x * mask_f).sum(dim=1)
    counts = mask_f.sum(dim=1).clamp(min=1.0)
    return summed / counts
