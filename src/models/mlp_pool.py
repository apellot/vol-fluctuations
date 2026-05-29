"""MLP-mean-pool architecture.

Inductive bias: "particles contribute additively and equally." This is the
simplest permutation-invariant set model — equivalent to DeepSets with the
per-particle φ replaced by the identity. It tests whether ANY learned
per-particle representation is necessary, or whether a hand-summarised event
representation suffices.

Pipeline:
    cont (B, L, F_p), mask (B, L), event_feats (B, F_e)
        → masked_mean over L           → (B, F_p)
        → concat event_feats           → (B, F_p + F_e)
        → MLP trunk                    → (B, hidden)
        → NIG head + softmax head
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .heads import N_CONT_FEATURES, N_EVENT_FEATURES, OutputHeads, make_mlp, masked_mean


@dataclass
class MLPPoolConfig:
    n_centrality_bins: int | None = None  # ignored (no classifier); kept for back-compat
    hidden: int = 128
    depth: int = 3
    dropout: float = 0.1


class MLPPool(nn.Module):
    def __init__(self, cfg: MLPPoolConfig) -> None:
        super().__init__()
        self.cfg = cfg
        trunk_in = N_CONT_FEATURES + N_EVENT_FEATURES
        self.trunk = make_mlp(trunk_in, cfg.hidden, cfg.depth, cfg.dropout)
        self.heads = OutputHeads(cfg.hidden, cfg.n_centrality_bins)

    def forward(self, cont: Tensor, mask: Tensor, event_feats: Tensor) -> dict[str, Tensor]:
        # cont: (B, L, F_p)   mask: (B, L)   event_feats: (B, F_e)
        pooled = masked_mean(cont, mask)                       # (B, F_p)
        merged = torch.cat([pooled, event_feats], dim=-1)      # (B, F_p + F_e)
        h = self.trunk(merged)                                 # (B, hidden)
        return self.heads(h)
