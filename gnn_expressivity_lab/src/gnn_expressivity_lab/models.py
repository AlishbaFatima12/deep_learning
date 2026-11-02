"""Minimal, auditable GCN and GIN graph classifiers implemented in PyTorch."""
import torch
from torch import nn
import torch.nn.functional as F
from .graphs import Graph

def adjacency(graph: Graph, self_loops: bool = False) -> torch.Tensor:
    n = graph.num_nodes
    a = torch.zeros((n, n), dtype=graph.x.dtype, device=graph.x.device)
    if graph.edge_index.numel():
        u, v = graph.edge_index; a[u, v] = 1.0
    if self_loops: a = a + torch.eye(n, dtype=a.dtype, device=a.device)
    return a

class GCNLayer(nn.Module):
    """Kipf-Welling style symmetric-normalised neighbourhood aggregation."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(); self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a_hat = a + torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
        degree = a_hat.sum(dim=1).clamp_min(1.0); inv_sqrt = degree.rsqrt()
        norm = inv_sqrt[:, None] * a_hat * inv_sqrt[None, :]
        return norm @ self.linear(x) + self.bias

class GINLayer(nn.Module):
    """GIN update: MLP((1 + epsilon) h_v + sum of neighbour features)."""
    def __init__(self, in_features: int, out_features: int, train_eps: bool = True):
        super().__init__(); self.eps = nn.Parameter(torch.zeros(()), requires_grad=train_eps)
        self.mlp = nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU(), nn.Linear(out_features, out_features))
    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.mlp((1.0 + self.eps) * x + a @ x)

class GraphClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 architecture: str = "gin", pooling: str = "sum", dropout: float = 0.0):
        super().__init__()
        if architecture not in {"gcn", "gin"}: raise ValueError("architecture must be 'gcn' or 'gin'")
        if pooling not in {"sum", "mean"}: raise ValueError("pooling must be 'sum' or 'mean'")
        layer = GCNLayer if architecture == "gcn" else GINLayer
        self.layers = nn.ModuleList([layer(input_dim, hidden_dim), layer(hidden_dim, hidden_dim)])
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        self.pooling = pooling
    def encode(self, graph: Graph) -> torch.Tensor:
        a = adjacency(graph); h = graph.x
        for layer in self.layers: h = F.relu(layer(h, a))
        return h.sum(dim=0) if self.pooling == "sum" else h.mean(dim=0)
    def forward(self, graph: Graph) -> torch.Tensor:
        return self.classifier(self.encode(graph))
