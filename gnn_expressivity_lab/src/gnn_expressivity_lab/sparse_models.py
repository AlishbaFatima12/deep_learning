"""Sparse message-passing GNNs using PyTorch edge_index and index_add operations.

No dense N x N adjacency matrix is constructed. The implementation is intended
for transparent experiments and small/medium graphs, not as a replacement for
optimised production kernels.
"""
from dataclasses import dataclass
from typing import List, Sequence
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class GraphBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor
    y: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_graphs(self) -> int:
        return int(self.y.shape[0])

def batch_graphs(graphs: Sequence) -> GraphBatch:
    if not graphs:
        raise ValueError("Cannot batch an empty graph sequence")
    feature_dim = int(graphs[0].x.shape[1])
    xs, edges, assignments, labels = [], [], [], []
    offset = 0
    for graph_id, graph in enumerate(graphs):
        if graph.x.ndim != 2 or graph.x.shape[1] != feature_dim:
            raise ValueError("All graphs in a batch must have the same feature dimension")
        x = graph.x.float()
        xs.append(x)
        if graph.edge_index.numel():
            edges.append(graph.edge_index.long() + offset)
        assignments.append(torch.full((graph.num_nodes,), graph_id, dtype=torch.long))
        labels.append(int(graph.label))
        offset += graph.num_nodes
    edge_index = torch.cat(edges, dim=1) if edges else torch.empty((2, 0), dtype=torch.long)
    return GraphBatch(torch.cat(xs, dim=0), edge_index,
                      torch.cat(assignments, dim=0), torch.tensor(labels, dtype=torch.long))

def _add_self_loops(edge_index: torch.Tensor, num_nodes: int):
    loops = torch.arange(num_nodes, device=edge_index.device, dtype=torch.long)
    return torch.cat([edge_index, torch.stack([loops, loops])], dim=1)

def _sum_messages(messages: torch.Tensor, index: torch.Tensor, num_nodes: int):
    out = messages.new_zeros((num_nodes, messages.shape[-1]))
    if index.numel():
        out.index_add_(0, index, messages)
    return out

def _mean_neighbours(x: torch.Tensor, edge_index: torch.Tensor):
    src, dst = edge_index
    sums = _sum_messages(x[src], dst, x.shape[0])
    counts = x.new_zeros((x.shape[0], 1))
    if dst.numel():
        counts.index_add_(0, dst, x.new_ones((dst.shape[0], 1)))
    return sums / counts.clamp_min(1.0)

class SparseGCNLayer(nn.Module):
    """GCN layer with symmetric degree normalisation over sparse edges."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        edge_index = _add_self_loops(edge_index, x.shape[0])
        src, dst = edge_index
        degree = x.new_zeros((x.shape[0],))
        degree.index_add_(0, dst, x.new_ones((dst.shape[0],)))
        norm = degree[src].clamp_min(1.0).rsqrt() * degree[dst].clamp_min(1.0).rsqrt()
        transformed = self.linear(x)
        out = _sum_messages(transformed[src] * norm.unsqueeze(-1), dst, x.shape[0])
        return out + self.bias

class SparseGINLayer(nn.Module):
    """GIN sum aggregation followed by an MLP."""
    def __init__(self, in_channels: int, out_channels: int, train_eps: bool = True):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(()), requires_grad=train_eps)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        src, dst = edge_index
        neighbours = _sum_messages(x[src], dst, x.shape[0])
        return self.mlp((1.0 + self.eps) * x + neighbours)

class SparseSAGELayer(nn.Module):
    """Mean-aggregating GraphSAGE layer with separate self/neighbour maps."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.self_linear = nn.Linear(in_channels, out_channels, bias=True)
        self.neighbour_linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        neighbours = _mean_neighbours(x, edge_index)
        return self.self_linear(x) + self.neighbour_linear(neighbours)

class SparseGATLayer(nn.Module):
    """Single-head additive graph attention using stable destination-wise softmax."""
    def __init__(self, in_channels: int, out_channels: int, negative_slope: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.attn_src = nn.Parameter(torch.empty(out_channels))
        self.attn_dst = nn.Parameter(torch.empty(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.negative_slope = negative_slope
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(0))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        edge_index = _add_self_loops(edge_index, x.shape[0])
        src, dst = edge_index
        h = self.linear(x)
        scores = F.leaky_relu((h[src] * self.attn_src).sum(-1) +
                              (h[dst] * self.attn_dst).sum(-1),
                              negative_slope=self.negative_slope)
        # Stable softmax independently for each destination node.
        maxima = scores.new_full((x.shape[0],), -torch.inf)
        maxima.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)
        exp_scores = torch.exp(scores - maxima[dst])
        denom = scores.new_zeros((x.shape[0],))
        denom.index_add_(0, dst, exp_scores)
        alpha = exp_scores / denom[dst].clamp_min(torch.finfo(exp_scores.dtype).tiny)
        out = _sum_messages(h[src] * alpha.unsqueeze(-1), dst, x.shape[0])
        return out + self.bias

class SparseGraphClassifier(nn.Module):
    """Two-layer graph classifier with sum or mean graph readout."""
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 architecture: str = "gin", pooling: str = "mean", dropout: float = 0.1):
        super().__init__()
        layers = {
            "gcn": SparseGCNLayer,
            "gin": SparseGINLayer,
            "sage": SparseSAGELayer,
            "gat": SparseGATLayer,
        }
        if architecture not in layers:
            raise ValueError("architecture must be one of: gcn, gin, sage, gat")
        if pooling not in {"sum", "mean"}:
            raise ValueError("pooling must be 'sum' or 'mean'")
        layer = layers[architecture]
        self.layers = nn.ModuleList([layer(input_dim, hidden_dim), layer(hidden_dim, hidden_dim)])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes)
        )
        self.pooling = pooling

    def encode(self, batch: GraphBatch):
        h = batch.x
        for layer in self.layers:
            h = F.relu(layer(h, batch.edge_index))
        pooled = h.new_zeros((batch.num_graphs, h.shape[-1]))
        pooled.index_add_(0, batch.batch, h)
        if self.pooling == "mean":
            counts = h.new_zeros((batch.num_graphs, 1))
            counts.index_add_(0, batch.batch, h.new_ones((h.shape[0], 1)))
            pooled = pooled / counts.clamp_min(1.0)
        return pooled

    def forward(self, batch: GraphBatch):
        return self.classifier(self.encode(batch))
