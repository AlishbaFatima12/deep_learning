"""Optional real-world graph classification benchmark using PyTorch Geometric TU datasets."""
import torch
from .graphs import Graph

def load_tu_graphs(name: str, root: str):
    """Load a TU dataset and convert examples to the project's graph data class."""
    try:
        from torch_geometric.datasets import TUDataset
    except ImportError as exc:
        raise RuntimeError("Install the optional benchmark dependencies: pip install -e '.[benchmark]'") from exc
    dataset = TUDataset(root=root, name=name)
    if len(dataset) < 4:
        raise ValueError(f"Dataset {name} is too small for a stratified experiment")
    graphs = []
    raw_labels = [int(data.y.view(-1)[0].item()) for data in dataset]
    label_map = {value: index for index, value in enumerate(sorted(set(raw_labels)))}
    for i, data in enumerate(dataset):
        if data.x is None:
            raise ValueError(f"{name} has no node features; add a documented feature encoding before using it")
        x = data.x.float()
        edge_index = data.edge_index.long()
        if edge_index.numel():
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_index = torch.unique(edge_index, dim=1)
        label = label_map[int(data.y.view(-1)[0].item())]
        graphs.append(Graph(x=x, edge_index=edge_index, label=label, name=f"{name}-{i}"))
    feature_dims = {int(graph.x.shape[1]) for graph in graphs}
    if len(feature_dims) != 1:
        raise ValueError("Dataset contains inconsistent node feature dimensions")
    labels = sorted({graph.label for graph in graphs})
    counts = {label: sum(graph.label == label for graph in graphs) for label in labels}
    if len(labels) < 2 or min(counts.values()) < 2:
        raise ValueError("Each class must contain at least two graphs")
    metadata = {
        "dataset": name, "graphs": len(graphs), "classes": len(labels),
        "node_feature_dim": next(iter(feature_dims)),
        "class_counts": {str(key): value for key, value in counts.items()},
        "original_label_mapping": {str(key): value for key, value in label_map.items()},
        "source": "PyTorch Geometric TUDataset",
        "note": "Use documented repeated folds and comparable model-selection rules for benchmark-standard claims."
    }
    return graphs, len(labels), metadata
