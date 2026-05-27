"""Graph Neural Network for centrality estimation (PyTorch Geometric).

Uses EdgeConv / DGCNN (Wang et al., SIGGRAPH 2019) with a kNN graph built in
(eta_lab, phi) space.  The padded dense input (B, L, C) + mask is stripped to
PyG's flat node representation internally; the forward signature is unchanged
so the training loop requires no modifications.

torch_cluster is not required.  knn_graph is implemented in pure PyTorch
(pairwise distance per mini-graph; cheap for FXT multiplicities of 5-10).

Pipeline:
    cont (B, L, 4)  mask (B, L)  event_feats (B, F_e)
        -> strip padding -> x (N_real, 4),  batch_idx (N_real,)
        -> _knn_graph(pos=[eta,phi], k, batch_idx) -> edge_index (2, E)
        -> stacked EdgeConv blocks with residual -> h (N_real, H)
        -> global_mean_pool -> (B, H)
        -> concat event_feats -> rho MLP -> NIG + softmax heads
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from torch_geometric.nn import EdgeConv, global_mean_pool

from .heads import N_CONT_FEATURES, N_EVENT_FEATURES, OutputHeads, make_mlp


@dataclass
class GNNConfig:
    n_centrality_bins: int
    k: int = 4
    edge_hidden: int = 50
    edge_depth: int = 2
    n_edge_layers: int = 2
    rho_hidden: int = 50
    rho_depth: int = 2
    dropout: float = 0.1


def _knn_graph(pos: Tensor, k: int, batch_idx: Tensor) -> Tensor:
    """Build a kNN graph in COO format without torch_cluster.

    For each node i, finds k nearest neighbours j (by Euclidean distance in
    pos-space, within the same batch element) and emits directed edges j->i
    (source-to-target convention matching PyG's EdgeConv default).

    Args:
        pos:       (N_real, D) coordinates used for distance computation
        k:         number of neighbours per node
        batch_idx: (N_real,) long — batch membership of each node

    Returns:
        edge_index: (2, E) long — row 0 = source, row 1 = target
    """
    device = pos.device
    src_list: list[Tensor] = []
    dst_list: list[Tensor] = []

    for b in batch_idx.unique():
        global_idx = (batch_idx == b).nonzero(as_tuple=True)[0]  # (n_b,)
        p = pos[global_idx]                                        # (n_b, D)
        n_b = p.shape[0]
        k_eff = min(k, n_b - 1)
        if k_eff <= 0:
            continue

        diff = p.unsqueeze(0) - p.unsqueeze(1)                    # (n_b, n_b, D)
        dist2 = (diff ** 2).sum(-1)                                # (n_b, n_b)
        dist2.fill_diagonal_(float("inf"))

        # nn_idx[i, :] = local indices of k nearest neighbours of node i
        _, nn_idx = dist2.topk(k_eff, dim=1, largest=False)       # (n_b, k_eff)

        targets_local = (
            torch.arange(n_b, device=device)
            .unsqueeze(1).expand(n_b, k_eff).reshape(-1)
        )
        sources_local = nn_idx.reshape(-1)

        src_list.append(global_idx[sources_local])
        dst_list.append(global_idx[targets_local])

    if not src_list:
        return torch.zeros(2, 0, dtype=torch.long, device=device)

    return torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)


def _edge_mlp(in_dim: int, hidden: int, depth: int, dropout: float) -> nn.Sequential:
    """MLP for EdgeConv: input is (x_i || x_j - x_i), so 2*in_dim."""
    trunk = make_mlp(2 * in_dim, hidden, depth, dropout)
    return nn.Sequential(*trunk, nn.Linear(hidden, hidden))


class GNN(nn.Module):
    def __init__(self, cfg: GNNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.input_proj = nn.Linear(N_CONT_FEATURES, cfg.edge_hidden)

        self.edge_convs = nn.ModuleList([
            EdgeConv(
                nn=_edge_mlp(cfg.edge_hidden, cfg.edge_hidden, cfg.edge_depth, cfg.dropout),
                aggr="max",
            )
            for _ in range(cfg.n_edge_layers)
        ])

        rho_in = cfg.edge_hidden + N_EVENT_FEATURES
        self.rho = make_mlp(rho_in, cfg.rho_hidden, cfg.rho_depth, cfg.dropout)
        self.heads = OutputHeads(cfg.rho_hidden, cfg.n_centrality_bins)

    def forward(self, cont: Tensor, mask: Tensor, event_feats: Tensor) -> dict[str, Tensor]:
        B, L, _ = cont.shape

        # Strip padding: convert (B, L, C) + mask to flat PyG node format.
        x = cont[mask]                                             # (N_real, 4)
        batch_idx = (
            torch.arange(B, device=cont.device)
            .unsqueeze(1).expand(B, L)[mask]
        )                                                          # (N_real,)

        # Project raw particle features to the hidden dimension.
        h = self.input_proj(x)                                     # (N_real, H)

        # Build kNN graph in (eta_lab, phi) space (features 1 and 2).
        pos = x[:, 1:3]
        edge_index = _knn_graph(pos.detach(), k=self.cfg.k, batch_idx=batch_idx)

        # Stacked EdgeConv blocks; residual connections from the second layer on.
        for i, conv in enumerate(self.edge_convs):
            h_new = conv(h, edge_index)
            h = h + h_new if i > 0 else h_new

        # Pool to one vector per event.  size=B ensures shape (B, H) even when
        # the last event(s) in the batch have zero real particles.
        pooled = global_mean_pool(h, batch_idx, size=B)            # (B, H)

        # Concat event-level scalars and run event MLP.
        merged = torch.cat([pooled, event_feats], dim=-1)
        out = self.rho(merged)
        return self.heads(out)
