"""Deep Sets architecture (Zaheer et al. 2017) — adapted for the lab-frame design.

Inductive bias: "each particle gets a learned embedding before being summed."
The key axis vs MLP-mean-pool is the per-particle MLP φ, which lets the model
re-weight different kinds of particles in the latent space *before* pooling.

Pipeline:
    cont (B, L, F_p), mask (B, L), event_feats (B, F_e)
        → φ(per-particle)               → (B, L, H)
        → masked mean over L            → (B, H)
        → concat event_feats            → (B, H + F_e)
        → ρ (event-level MLP)           → (B, hidden)
        → NIG head + softmax head
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .heads import N_CONT_FEATURES, N_EVENT_FEATURES, OutputHeads, make_mlp, masked_mean


@dataclass
class DeepSetsConfig:
    n_centrality_bins: int
    phi_hidden: int = 128
    phi_depth: int = 2
    rho_hidden: int = 128
    rho_depth: int = 2
    dropout: float = 0.1


class DeepSets(nn.Module):
    def __init__(self, cfg: DeepSetsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.phi = make_mlp(N_CONT_FEATURES, cfg.phi_hidden, cfg.phi_depth, cfg.dropout)
        rho_in = cfg.phi_hidden + N_EVENT_FEATURES
        self.rho = make_mlp(rho_in, cfg.rho_hidden, cfg.rho_depth, cfg.dropout)
        self.heads = OutputHeads(cfg.rho_hidden, cfg.n_centrality_bins)

    def forward(self, cont: Tensor, mask: Tensor, event_feats: Tensor) -> dict[str, Tensor]:
        phi = self.phi(cont)                       # (B, L, phi_hidden)
        pooled = masked_mean(phi, mask)            # (B, phi_hidden)
        merged = torch.cat([pooled, event_feats], dim=-1)
        h = self.rho(merged)
        return self.heads(h)
