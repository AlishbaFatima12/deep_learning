"""Optional real-world graph classification benchmark using PyTorch Geometric TU datasets.

Install with: pip install -e ".[dev,benchmark]"
Example: gnn-benchmark --dataset MUTAG --epochs 100 --output results/mutag.json
Dataset download occurs only when this command is run.
"""
import argparse
import json
from pathlib import Path
import torch
from .graphs import Graph
from .sparse_experiment import run_graph_experiment

def load_tu_graphs(name: str, root: str):
    try:
        from torch_geometric.datasets import TUDataset
    except ImportError as exc:
        raise RuntimeError("Install the optional benchmark dependencies: pip install -e '.[benchmark]'") from exc
    dataset = TUDataset(root=root, name=name)
    if len(dataset) < 4:
        raise ValueError(f"Dataset {name} is too small for a stratified experiment")
    graphs = []
    for i, data in enumerate(dataset):
        if data.x is None:
            raise ValueError(f"{name} has no node features; add a documented feature encoding before using it")
        x = data.x.float()
        edge_index = data.edge_index.long()
        if edge_index.numel():
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_index = torch.unique(edge_index, dim=1)
        label = int(data.y.view(-1)[0].item())
        graphs.append(Graph(x=x, edge_index=edge_index, label=label, name=f"{name}-{i}"))
    feature_dims = {int(g.x.shape[1]) for g in graphs}
    if len(feature_dims) != 1:
        raise ValueError("Dataset contains inconsistent node feature dimensions")
    labels = sorted({g.label for g in graphs})
    counts = {label: sum(g.label == label for g in graphs) for label in labels}
    if len(labels) < 2 or min(counts.values()) < 2:
        raise ValueError("Each class must contain at least two graphs")
    return graphs, len(labels), {
        "dataset": name, "graphs": len(graphs), "classes": len(labels),
        "node_feature_dim": next(iter(feature_dims)),
        "class_counts": {str(k): v for k, v in counts.items()},
        "source": "PyTorch Geometric TUDataset",
        "note": "Use identical seeds and settings for comparisons; this basic protocol is not a substitute for benchmark-standard repeated folds."
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="MUTAG", help="TU dataset name, e.g. MUTAG or PROTEINS")
    parser.add_argument("--root", default="data/TUDataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--output", type=Path, default=Path("results/MUTAG.json"))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    graphs, num_classes, metadata = load_tu_graphs(args.dataset, args.root)
    result = run_graph_experiment(graphs, num_classes, tuple(args.seeds), args.epochs,
                                  dataset_name=args.dataset)
    result["dataset_metadata"] = metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
