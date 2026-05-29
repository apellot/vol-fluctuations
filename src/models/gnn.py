"""Graph Neural Network for centrality estimation.

Connects each particle to its k nearest neighbours in (eta_lab, phi) space
using edge convolutions (Wang et al., SIGGRAPH 2019 — EdgeConv / DGCNN).
At low multiplicity (~5-10 particles in FXT), the graph is very small so
this is fast and captures pairwise correlations (e.g. back-to-back proton
pairs) that pure pooling architectures miss.

Pipeline:
    cont (B, L, 4) = pT, eta_lab, phi, charge    mask (B, L)    event_feats (B, F_e)
        → build kNN graph in (eta_lab, phi)       → edge index (B, L, k)
        → EdgeConv layer(s): aggregate (x_i || x_j - x_i) over neighbours → (B, L, H)
        → masked mean pool                        → (B, H)
        → concat event_feats                      → (B, H + F_e)
        → event-level MLP                         → (B, rho_hidden)
        → NIG head + softmax head

kNN is computed on the CPU in O(L^2) — totally fine for L <= 512 and typical
FXT multiplicities of 5-10 real particles (padded positions are masked out).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .heads import N_CONT_FEATURES, N_EVENT_FEATURES, OutputHeads, make_mlp, masked_mean


@dataclass
class GNNConfig:
    n_centrality_bins: int | None = None  # ignored (no classifier); kept for back-compat
    k: int = 4                   # neighbours per particle (clipped to actual multiplicity)
    edge_hidden: int = 128       # width of the EdgeConv MLP
    edge_depth: int = 2          # depth of the EdgeConv MLP
    n_edge_layers: int = 2       # number of stacked EdgeConv blocks
    rho_hidden: int = 128        # event-level MLP width
    rho_depth: int = 2
    dropout: float = 0.1


def _knn_graph(pos: Tensor, k: int, mask: Tensor) -> Tensor:
    """Build kNN graph indices in (eta, phi) space.

    Args:
        pos:  (B, L, 2) — (eta_lab, phi) coordinates
        k:    number of neighbours to find (will be clipped to real particle count)
        mask: (B, L) bool — True for real particles

    Returns:
        idx: (B, L, k) long — neighbour indices; padded particles point to 0
    """
    B, L, _ = pos.shape
    # Pairwise squared distances, shape (B, L, L).
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)   # (B, L, L, 2)
    dist2 = (diff ** 2).sum(-1)                  # (B, L, L)

    # Mask out padded positions so they never become neighbours.
    # Set their distance to a large value.
    INF = 1e9
    mask_f = mask.float()                        # (B, L)
    not_real = (1.0 - mask_f).unsqueeze(1)       # (B, 1, L)  — cols = candidates
    dist2 = dist2 + not_real * INF

    # Also mask out self-loops.
    eye = torch.eye(L, device=pos.device, dtype=torch.bool).unsqueeze(0)
    dist2 = dist2.masked_fill(eye, INF)

    # Clip k to the actual number of real neighbours available.
    n_real = mask.long().sum(dim=1)              # (B,)
    k_eff = max(1, min(k, int(n_real.min().item()) - 1))
    k_eff = max(k_eff, 1)

    # (B, L, k_eff) — indices of nearest neighbours.
    _, idx = dist2.topk(k_eff, dim=-1, largest=False, sorted=False)
    return idx                                   # (B, L, k_eff)


class EdgeConv(nn.Module):
    """Single EdgeConv block.

    For each particle i, aggregates messages from its k neighbours j:
        m_ij = MLP(x_i || x_j - x_i)
    Then max-pools over j to get the new representation of i.
    """

    def __init__(self, in_dim: int, hidden: int, depth: int, dropout: float) -> None:
        super().__init__()
        # Input to the edge MLP: (x_i, x_j - x_i) → 2 * in_dim
        self.mlp = make_mlp(2 * in_dim, hidden, depth, dropout)
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, x: Tensor, idx: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x:    (B, L, C) particle features
            idx:  (B, L, k) neighbour indices
            mask: (B, L)    bool, True = real particle

        Returns:
            (B, L, hidden)
        """
        B, L, C = x.shape
        k = idx.shape[-1]

        # Gather neighbour features: (B, L, k, C)
        idx_exp = idx.unsqueeze(-1).expand(B, L, k, C)
        x_exp = x.unsqueeze(2).expand(B, L, L, C)
        # We need to gather along dim=2 (the L dimension of candidates).
        x_j = torch.gather(x.unsqueeze(2).expand(B, L, L, C),
                           2,
                           idx.unsqueeze(-1).expand(B, L, k, C))  # (B, L, k, C)

        x_i = x.unsqueeze(2).expand(B, L, k, C)                  # (B, L, k, C)
        edge_feat = torch.cat([x_i, x_j - x_i], dim=-1)          # (B, L, k, 2C)

        # Apply MLP to each edge independently.
        edge_feat = edge_feat.view(B * L * k, 2 * C)
        h = self.mlp(edge_feat)                                    # (B*L*k, hidden)
        h = h.view(B, L, k, -1)                                   # (B, L, k, hidden)

        # Max-pool over neighbours.
        h, _ = h.max(dim=2)                                       # (B, L, hidden)
        h = self.out_proj(h)

        # Zero out padded positions.
        h = h * mask.unsqueeze(-1).float()
        return h


class GNN(nn.Module):
    def __init__(self, cfg: GNNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Input projection: map raw particle features to edge_hidden dim.
        self.input_proj = nn.Linear(N_CONT_FEATURES, cfg.edge_hidden)

        # Stack of EdgeConv blocks (residual connections after the first).
        self.edge_convs = nn.ModuleList([
            EdgeConv(cfg.edge_hidden, cfg.edge_hidden, cfg.edge_depth, cfg.dropout)
            for _ in range(cfg.n_edge_layers)
        ])

        # Event-level MLP after pooling.
        rho_in = cfg.edge_hidden + N_EVENT_FEATURES
        self.rho = make_mlp(rho_in, cfg.rho_hidden, cfg.rho_depth, cfg.dropout)
        self.heads = OutputHeads(cfg.rho_hidden, cfg.n_centrality_bins)

    def forward(self, cont: Tensor, mask: Tensor, event_feats: Tensor) -> dict[str, Tensor]:
        # Build kNN graph from (eta_lab, phi) — features at indices 1 and 2.
        pos = cont[..., 1:3]                          # (B, L, 2)
        idx = _knn_graph(pos, self.cfg.k, mask)       # (B, L, k)

        # Project raw features into the hidden space.
        h = self.input_proj(cont)                     # (B, L, edge_hidden)
        h = h * mask.unsqueeze(-1).float()

        # Apply stacked EdgeConv blocks with residual connections.
        for i, conv in enumerate(self.edge_convs):
            h_new = conv(h, idx, mask)
            h = h + h_new if i > 0 else h_new        # residual after first layer

        # Global masked mean pool.
        pooled = masked_mean(h, mask)                 # (B, edge_hidden)

        # Concat event-level scalars and run event MLP.
        merged = torch.cat([pooled, event_feats], dim=-1)
        out = self.rho(merged)
        return self.heads(out)