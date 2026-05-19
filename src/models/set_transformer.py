"""Set Transformer (Lee et al. 2019) — adapted for the lab-frame design.

Inductive bias: "particles can interact pair-wise via attention before pooling."
Same input contract as DeepSets; differs in that the per-particle embeddings
attend to each other across SAB layers before being pooled by PMA.

Building blocks from the paper:
    MAB(Q, K) — multi-head attention block with residuals and a feed-forward sublayer
    SAB(X)   = MAB(X, X)                — self-attention
    PMA(X)   = MAB(seed_vectors, X)     — k learnable seeds query the set

Pipeline:
    cont (B, L, F_p), mask (B, L), event_feats (B, F_e)
        → input proj                    → (B, L, d_model)
        → SAB × n_sab                   → (B, L, d_model)
        → PMA(k=1)                      → (B, d_model)
        → concat event_feats            → (B, d_model + F_e)
        → ρ                             → (B, hidden)
        → NIG head + softmax head
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .heads import N_CONT_FEATURES, N_EVENT_FEATURES, OutputHeads, make_mlp


@dataclass
class SetTransformerConfig:
    n_centrality_bins: int
    d_model: int = 96
    n_heads: int = 4
    n_sab: int = 2
    n_pma_seeds: int = 1
    ff_hidden: int = 192
    rho_hidden: int = 128
    rho_depth: int = 2
    dropout: float = 0.1


class MAB(nn.Module):
    """Multi-head attention block.

    Output = LN(H + FF(H)) with H = LN(Q + Attn(Q, K, K)).
    Padded keys are masked out via key_padding_mask so they don't contribute to softmax.
    """
    def __init__(self, d_model: int, n_heads: int, ff_hidden: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ff_hidden, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: Tensor, kv: Tensor, kv_mask: Tensor | None = None) -> Tensor:
        # nn.MultiheadAttention uses True-means-ignore in key_padding_mask, so invert our True-is-valid.
        kpm = (~kv_mask) if kv_mask is not None else None
        attn, _ = self.attn(q, kv, kv, key_padding_mask=kpm, need_weights=False)
        h = self.ln1(q + self.drop(attn))
        out = self.ln2(h + self.drop(self.ff(h)))
        return out


class SAB(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_hidden: int, dropout: float) -> None:
        super().__init__()
        self.mab = MAB(d_model, n_heads, ff_hidden, dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        return self.mab(x, x, mask)


class PMA(nn.Module):
    def __init__(self, d_model: int, n_seeds: int, n_heads: int, ff_hidden: int, dropout: float) -> None:
        super().__init__()
        # Small Gaussian init breaks symmetry between heads at the start of training.
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model) * 0.02)
        self.mab = MAB(d_model, n_heads, ff_hidden, dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        B = x.shape[0]
        q = self.seeds.expand(B, -1, -1)
        return self.mab(q, x, mask)


class SetTransformer(nn.Module):
    def __init__(self, cfg: SetTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(N_CONT_FEATURES, cfg.d_model)
        self.sabs = nn.ModuleList([
            SAB(cfg.d_model, cfg.n_heads, cfg.ff_hidden, cfg.dropout) for _ in range(cfg.n_sab)
        ])
        self.pma = PMA(cfg.d_model, cfg.n_pma_seeds, cfg.n_heads, cfg.ff_hidden, cfg.dropout)

        rho_in = cfg.d_model * cfg.n_pma_seeds + N_EVENT_FEATURES
        self.rho = make_mlp(rho_in, cfg.rho_hidden, cfg.rho_depth, cfg.dropout)
        self.heads = OutputHeads(cfg.rho_hidden, cfg.n_centrality_bins)

    def forward(self, cont: Tensor, mask: Tensor, event_feats: Tensor) -> dict[str, Tensor]:
        x = self.input_proj(cont)
        for sab in self.sabs:
            x = sab(x, mask)
        pooled = self.pma(x, mask)                  # (B, k, d_model)
        pooled_flat = pooled.flatten(start_dim=1)   # (B, k · d_model)
        merged = torch.cat([pooled_flat, event_feats], dim=-1)
        h = self.rho(merged)
        return self.heads(h)
