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


def nig_from_raw(raw: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Map a (…, 4) raw linear output to NIG params (μ, ν, α, β).

    softplus+offsets enforce ν > 0, α > 1, β > 0 (μ unconstrained), matching
    src/losses/evidential.py.
    """
    mu = raw[..., 0]
    nu = nn.functional.softplus(raw[..., 1]) + 1e-6
    alpha = nn.functional.softplus(raw[..., 2]) + 1.0 + 1e-6
    beta = nn.functional.softplus(raw[..., 3]) + 1e-6
    return mu, nu, alpha, beta


class OutputHeads(nn.Module):
    """Two evidential NIG heads — one for b, one for Npart. NO classification head.

    Centrality is derived downstream from predicted Npart (rank → percentile bins),
    not learned here (see docs/modelling_plan.md, 2026-05-28). Both targets are
    trained in STANDARDIZED space; the trainer inverts μ/σ back to physical units.

    `n_centrality_bins` is accepted but ignored for back-compat with the model
    configs that still carry the field — there is no classifier any more.
    """
    def __init__(self, trunk_dim: int, n_centrality_bins: int | None = None) -> None:
        super().__init__()
        self.b_nig = nn.Linear(trunk_dim, 4)
        self.npart_nig = nn.Linear(trunk_dim, 4)

    def forward(self, h: Tensor) -> dict[str, Tensor]:
        b_mu, b_nu, b_alpha, b_beta = nig_from_raw(self.b_nig(h))
        np_mu, np_nu, np_alpha, np_beta = nig_from_raw(self.npart_nig(h))
        return {
            "b_mu": b_mu, "b_nu": b_nu, "b_alpha": b_alpha, "b_beta": b_beta,
            "np_mu": np_mu, "np_nu": np_nu, "np_alpha": np_alpha, "np_beta": np_beta,
        }


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
