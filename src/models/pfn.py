"""Particle Flow Network (Komiske, Metodiev, Thaler — JHEP 2019, arXiv:1810.05165).

Sister model to EFN. The key difference: the per-particle embedding network Λ
sees ALL per-particle features (pT, η_lab, φ, charge) rather than only the
angular variables. This makes PFN more expressive than EFN at the cost of
IRC safety — which is not a concern in the centrality context.

Mathematically:
    F(event) = Φ( Σ_i  Λ(x_i) )
where x_i = (pT, η_lab, φ, charge) and the sum is unweighted (contrast with
EFN where the sum is pT-weighted and Λ only sees angular variables).

Pipeline:
    cont (B, L, 4) = pT, η_lab, φ, charge    mask (B, L)    event_feats (B, F_e)
        → Λ MLP on all 4 features             → (B, L, latent)
        → mask out pads, sum over particles   → (B, latent)
        → concat event_feats                  → (B, latent + F_e)
        → Φ (event-level MLP)                 → (B, hidden)
        → NIG head + softmax head
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .heads import N_CONT_FEATURES, N_EVENT_FEATURES, OutputHeads, make_mlp


@dataclass
class PFNConfig:
    n_centrality_bins: int
    latent: int = 96
    phi_depth: int = 2
    phi_hidden: int = 96
    rho_hidden: int = 128
    rho_depth: int = 2
    dropout: float = 0.1


class PFN(nn.Module):
    def __init__(self, cfg: PFNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Λ sees all 4 per-particle features (contrast with EFN which sees 3).
        self.Lambda = make_mlp(N_CONT_FEATURES, cfg.phi_hidden, cfg.phi_depth, cfg.dropout)
        self.Lambda_out = nn.Linear(cfg.phi_hidden, cfg.latent)

        # Event-level Φ takes the pooled latent plus global event features.
        rho_in = cfg.latent + N_EVENT_FEATURES
        self.Phi = make_mlp(rho_in, cfg.rho_hidden, cfg.rho_depth, cfg.dropout)
        self.heads = OutputHeads(cfg.rho_hidden, cfg.n_centrality_bins)

    def forward(self, cont: Tensor, mask: Tensor, event_feats: Tensor) -> dict[str, Tensor]:
        lam = self.Lambda(cont)               # (B, L, phi_hidden)
        lam = self.Lambda_out(lam)            # (B, L, latent)

        # Mask out padded positions and sum over real particles.
        lam = lam * mask.unsqueeze(-1).float()
        pooled = lam.sum(dim=1)               # (B, latent)

        merged = torch.cat([pooled, event_feats], dim=-1)
        h = self.Phi(merged)
        return self.heads(h)