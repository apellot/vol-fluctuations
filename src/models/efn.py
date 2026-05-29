"""Energy Flow Network (Komiske, Metodiev, Thaler — JHEP 2019, arXiv:1810.05165).

Inductive bias: per-particle contribution to the event embedding is weighted by
*energy* (or pT in our case). Mathematically:

    F(event) = Φ( Σ_i  z_i · Λ(x_i) )

where z_i is a per-particle weight (we use pT) and Λ is a learned per-particle
embedding network applied to the remaining features (η_lab, φ, charge).

For IRC-safety (the original motivation in jet physics), Λ depends only on
angular variables (η, φ) and not on z_i. In the centrality context IRC safety
is less critical than at the LHC, but we keep the convention because (a) it
matches the canonical EFN and (b) charge is the only other safe input.

Pipeline:
    cont (B, L, 4) = pT, η_lab, φ, charge,    mask (B, L), event_feats (B, F_e)
        → split: z = cont[..., 0] (pT),  Λ_in = cont[..., 1:]    (3 features)
        → Λ MLP                                                 → (B, L, latent)
        → multiply by z, mask out pads, sum over particles      → (B, latent)
        → concat event_feats                                    → (B, latent + F_e)
        → Φ (event-level MLP)                                   → (B, hidden)
        → NIG head + softmax head

The summation rather than mean is intentional — EFNs are defined as an
*energy-weighted sum* and that is what gives the network its IRC-safety
property in the jet-physics application.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .heads import N_CONT_FEATURES, N_EVENT_FEATURES, OutputHeads, make_mlp


@dataclass
class EFNConfig:
    n_centrality_bins: int | None = None  # ignored (no classifier); kept for back-compat
    latent: int = 96
    phi_depth: int = 2          # depth of the per-particle Λ network
    phi_hidden: int = 96
    rho_hidden: int = 128       # event-level Φ network
    rho_depth: int = 2
    dropout: float = 0.1


class EFN(nn.Module):
    def __init__(self, cfg: EFNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Per-particle Λ takes (eta_lab, phi, charge) — 3 features. By
        # construction it does NOT see pT, which enters only as the summation weight.
        self.Lambda = make_mlp(N_CONT_FEATURES - 1, cfg.phi_hidden, cfg.phi_depth, cfg.dropout)
        self.Lambda_out = nn.Linear(cfg.phi_hidden, cfg.latent)

        # Event-level Φ takes the pooled latent plus the global event features.
        rho_in = cfg.latent + N_EVENT_FEATURES
        self.Phi = make_mlp(rho_in, cfg.rho_hidden, cfg.rho_depth, cfg.dropout)
        self.heads = OutputHeads(cfg.rho_hidden, cfg.n_centrality_bins)

    def forward(self, cont: Tensor, mask: Tensor, event_feats: Tensor) -> dict[str, Tensor]:
        # Split the per-particle inputs: pT acts as the weight z, the rest go through Λ.
        z = cont[..., 0]                              # (B, L) — pT
        angular = cont[..., 1:]                       # (B, L, 3) — (η_lab, φ, charge)
        lam = self.Lambda(angular)                    # (B, L, phi_hidden)
        lam = self.Lambda_out(lam)                    # (B, L, latent)

        # Mask out padded positions, multiply by per-particle weight, then sum.
        weight = (z * mask.float()).unsqueeze(-1)     # (B, L, 1)
        pooled = (lam * weight).sum(dim=1)            # (B, latent)

        merged = torch.cat([pooled, event_feats], dim=-1)
        h = self.Phi(merged)
        return self.heads(h)
