import torch
import pytest
from gnn_expressivity_lab.graphs import path_graph, cycle_graph, disjoint_cycles
from gnn_expressivity_lab.sparse_models import (
    SparseGraphClassifier, batch_graphs, SparseGCNLayer, SparseGINLayer,
    SparseSAGELayer, SparseGATLayer
)
from gnn_expressivity_lab.wl import wl_equivalent

@pytest.mark.parametrize("architecture", ["gcn", "gin", "sage", "gat"])
def test_sparse_models_forward_and_backward(architecture):
    batch = batch_graphs([path_graph(5), cycle_graph(6)])
    model = SparseGraphClassifier(1, 8, 2, architecture=architecture)
    logits = model(batch)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()
    loss = logits.square().mean()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

def test_batch_offsets_edges_and_graph_assignment():
    graphs = [path_graph(3), path_graph(4)]
    batch = batch_graphs(graphs)
    assert batch.x.shape == (7, 1)
    assert batch.batch.tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert int(batch.edge_index.max()) == 6
    assert torch.all(batch.edge_index[:, :4] < 3)

def test_sparse_gcn_layer_handles_isolated_node():
    x = torch.ones((2, 1))
    edge_index = torch.empty((2, 0), dtype=torch.long)
    out = SparseGCNLayer(1, 4)(x, edge_index)
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()

def test_wl_counterexample_remains_separate_from_model_accuracy():
    assert wl_equivalent(cycle_graph(10), disjoint_cycles([5, 5]), rounds=5)


def test_degree_feature_ablation_encodes_node_degrees():
    from gnn_expressivity_lab.graphs import benchmark_graphs
    constant = benchmark_graphs(feature_mode="constant")
    degree = benchmark_graphs(feature_mode="degree")
    assert constant[0].x.shape == degree[0].x.shape
    assert torch.all(constant[0].x == 1)
    assert torch.unique(degree[0].x).numel() > 1
