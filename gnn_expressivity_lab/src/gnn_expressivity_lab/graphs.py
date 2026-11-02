"""Small, deterministic graph constructors for controlled expressivity studies."""
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple
import torch

@dataclass
class Graph:
    x: torch.Tensor
    edge_index: torch.Tensor
    label: int = -1
    name: str = "graph"
    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

def _edge_index(num_nodes: int, edges: Iterable[Tuple[int, int]]) -> torch.Tensor:
    directed = set()
    for u, v in edges:
        if u == v: continue
        directed.add((u, v)); directed.add((v, u))
    ordered = sorted(directed)
    if not ordered: return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(ordered, dtype=torch.long).t().contiguous()

def make_graph(num_nodes: int, edges: Iterable[Tuple[int, int]], label: int = -1,
               name: str = "graph", feature_mode: str = "constant") -> Graph:
    if num_nodes < 1: raise ValueError("num_nodes must be positive")
    edge_index = _edge_index(num_nodes, edges)
    if feature_mode == "constant": x = torch.ones((num_nodes, 1), dtype=torch.float32)
    elif feature_mode == "degree":
        degree = torch.bincount(edge_index[0], minlength=num_nodes).float()
        x = degree.unsqueeze(1)
    else: raise ValueError("feature_mode must be 'constant' or 'degree'")
    return Graph(x=x, edge_index=edge_index, label=label, name=name)

def path_graph(n: int, **kwargs) -> Graph:
    return make_graph(n, ((i, i + 1) for i in range(n - 1)), name=f"path-{n}", **kwargs)

def star_graph(n: int, **kwargs) -> Graph:
    if n < 2: raise ValueError("a star requires at least two nodes")
    return make_graph(n, ((0, i) for i in range(1, n)), name=f"star-{n}", **kwargs)

def cycle_graph(n: int, **kwargs) -> Graph:
    if n < 3: raise ValueError("a cycle requires at least three nodes")
    return make_graph(n, ((i, (i + 1) % n) for i in range(n)), name=f"cycle-{n}", **kwargs)

def wheel_graph(n: int, **kwargs) -> Graph:
    if n < 4: raise ValueError("a wheel requires at least four nodes")
    rim = [(i, (i + 1) % (n - 1)) for i in range(1, n)]
    spokes = [(0, i) for i in range(1, n)]
    return make_graph(n, rim + spokes, name=f"wheel-{n}", **kwargs)

def complete_graph(n: int, **kwargs) -> Graph:
    return make_graph(n, ((i, j) for i in range(n) for j in range(i + 1, n)), name=f"complete-{n}", **kwargs)

def complete_bipartite_graph(left: int, right: int, **kwargs) -> Graph:
    n = left + right
    return make_graph(n, ((i, j) for i in range(left) for j in range(left, n)), name=f"complete-bipartite-{left}-{right}", **kwargs)

def disjoint_cycles(lengths: Sequence[int], **kwargs) -> Graph:
    if not lengths or any(n < 3 for n in lengths): raise ValueError("provide cycle lengths >= 3")
    edges, offset = [], 0
    for n in lengths:
        edges.extend((offset + i, offset + (i + 1) % n) for i in range(n)); offset += n
    return make_graph(offset, edges, name="disjoint-cycles-" + "-".join(map(str, lengths)), **kwargs)

def benchmark_graphs(feature_mode: str = "constant"):
    """Small synthetic sanity-check set; not a substitute for recognised benchmarks."""
    constructors = [
        (0, lambda n: path_graph(n, feature_mode=feature_mode)),
        (1, lambda n: star_graph(n, feature_mode=feature_mode)),
        (2, lambda n: wheel_graph(n, feature_mode=feature_mode)),
        (3, lambda n: complete_graph(n, feature_mode=feature_mode)),
        (4, lambda n: complete_bipartite_graph(3, n - 3, feature_mode=feature_mode)),
    ]
    graphs = []
    for label, constructor in constructors:
        for n in range(7, 13):
            graph = constructor(n); graph.label = label; graphs.append(graph)
    return graphs
