"""Canonical 1-dimensional Weisfeiler-Leman colour refinement utilities."""
from collections import Counter
from hashlib import sha256
from .graphs import Graph

def _token(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()

def wl_refinement(graph: Graph, rounds: int = 3):
    """Return colour histograms at t=0..rounds using deterministic signatures."""
    if rounds < 0: raise ValueError("rounds must be non-negative")
    neighbours = [[] for _ in range(graph.num_nodes)]
    for u, v in graph.edge_index.t().tolist(): neighbours[u].append(v)
    initial = [tuple(float(v) for v in row) for row in graph.x.tolist()]
    colours = [_token(repr(value)) for value in initial]
    history = [tuple(sorted(Counter(colours).items()))]
    for _ in range(rounds):
        signatures = [colours[i] + "|" + ",".join(sorted(colours[j] for j in neighbours[i])) for i in range(graph.num_nodes)]
        colours = [_token(signature) for signature in signatures]
        history.append(tuple(sorted(Counter(colours).items())))
    return history

def wl_equivalent(left: Graph, right: Graph, rounds: int = 3) -> bool:
    """Whether two graphs have matching 1-WL colour histograms each round."""
    return wl_refinement(left, rounds) == wl_refinement(right, rounds)
