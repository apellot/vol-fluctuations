"""Feature-vector MLP baseline for joint centrality regression and classification.

The model has a shared trunk and two heads:

  Trunk:
    F input features  →  [Linear → ReLU → Dropout] × depth  →  hidden vector.

  Heads:
    1. NIG head — four scalar outputs (μ_raw, ν_raw, α_raw, β_raw) mapped through
       activations so ν > 0, α > 1, β > 0 (μ is unconstrained). Trained with the
       evidential loss in src/losses/evidential.py.
    2. Centrality classification head — softmax over n_bins centrality classes,
       trained with cross-entropy against the truth-tuned bin label.

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


@dataclass
class MLPConfig:
    n_features: int          # number of scalar input features per event
    n_centrality_bins: int   # number of softmax classes (matches the baseline binning)
    hidden_dim: int = 128
    depth: int = 3           # number of hidden Linear layers
    dropout: float = 0.1


class MLPHead(nn.Module):
    """Trunk + two heads. Returns a dict to keep call sites self-documenting."""

    def __init__(self, cfg: MLPConfig) -> None:
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        in_dim = cfg.n_features
        for _ in range(cfg.depth):
            layers += [nn.Linear(in_dim, cfg.hidden_dim), nn.ReLU(), nn.Dropout(cfg.dropout)]
            in_dim = cfg.hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head_nig = nn.Linear(cfg.hidden_dim, 4)
        self.head_class = nn.Linear(cfg.hidden_dim, cfg.n_centrality_bins)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        h = self.trunk(x)
        raw = self.head_nig(h)
        # Split the four raw outputs and apply the constraints (μ, ν, α, β).
        mu = raw[..., 0]
        # softplus → strictly positive; the +1 on alpha keeps the predictive
        # Student-t with finite variance, since aleatoric ∝ 1/(α−1).
        nu = nn.functional.softplus(raw[..., 1]) + 1e-6
        alpha = nn.functional.softplus(raw[..., 2]) + 1.0 + 1e-6
        beta = nn.functional.softplus(raw[..., 3]) + 1e-6

        logits = self.head_class(h)
        return {"mu": mu, "nu": nu, "alpha": alpha, "beta": beta, "logits": logits}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
