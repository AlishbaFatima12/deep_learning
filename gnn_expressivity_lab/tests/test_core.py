import torch
from gnn_expressivity_lab.graphs import cycle_graph, disjoint_cycles, path_graph, star_graph
from gnn_expressivity_lab.models import GraphClassifier, adjacency
from gnn_expressivity_lab.wl import wl_equivalent, wl_refinement

def test_undirected_adjacency_and_self_loops():
    graph = path_graph(4); a = adjacency(graph)
    assert torch.equal(a, a.T)
    assert torch.all(torch.diag(a) == 0)
    assert torch.all(torch.diag(adjacency(graph, self_loops=True)) == 1)

def test_wl_detects_degree_profile_difference():
    assert not wl_equivalent(path_graph(6), star_graph(6), rounds=2)

def test_wl_collision_pair_is_reproducible():
    assert wl_equivalent(cycle_graph(10), disjoint_cycles([5, 5]), rounds=5)
    assert wl_refinement(cycle_graph(10), 5) == wl_refinement(disjoint_cycles([5, 5]), 5)

def test_both_models_return_finite_graph_logits():
    graph = path_graph(5)
    for architecture in ("gcn", "gin"):
        logits = GraphClassifier(1, 8, 3, architecture=architecture)(graph)
        assert logits.shape == (3,)
        assert torch.isfinite(logits).all()
