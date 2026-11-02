"""Batched sparse GNN experiments and reproducible metric reporting."""
import argparse
import json
import random
import time
from pathlib import Path
from statistics import mean, pstdev
import torch
from torch import nn
from .graphs import benchmark_graphs, cycle_graph, disjoint_cycles
from .sparse_models import SparseGraphClassifier, batch_graphs
from .wl import wl_equivalent

ARCHITECTURES = ("gcn", "gin", "sage", "gat")

def set_seed(seed: int):
    """Seed Python and PyTorch random generators."""
    random.seed(seed)
    torch.manual_seed(seed)

def stratified_split(graphs, seed: int, train_fraction: float = 0.7):
    """Create a deterministic stratified holdout split."""
    rng = random.Random(seed)
    train, test = [], []
    for label in sorted({int(g.label) for g in graphs}):
        group = [g for g in graphs if int(g.label) == label]
        if len(group) < 2:
            raise ValueError(f"Class {label} needs at least two graphs for a train/test split")
        rng.shuffle(group)
        cut = max(1, min(len(group) - 1, round(len(group) * train_fraction)))
        train.extend(group[:cut])
        test.extend(group[cut:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test

def _chunks(items, batch_size):
    """Yield consecutive mini-batches from a sequence."""
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]

def evaluate(model, graphs, batch_size=8):
    """Calculate graph classification accuracy without gradient tracking."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for group in _chunks(graphs, batch_size):
            batch = batch_graphs(group)
            predictions = model(batch).argmax(dim=-1)
            correct += int((predictions == batch.y).sum().item())
            total += len(group)
    return correct / max(total, 1)

def train_on_split(train, test, num_classes, architecture, seed, epochs=120,
                   hidden_dim=32, batch_size=8, pooling="mean"):
    """Fit one architecture on a fixed split and return descriptive metrics."""
    set_seed(seed)
    input_dim = int(train[0].x.shape[1])
    model = SparseGraphClassifier(input_dim, hidden_dim, num_classes,
                                  architecture=architecture, pooling=pooling, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    start = time.perf_counter()
    for _ in range(epochs):
        shuffled = train[:]
        random.shuffle(shuffled)
        model.train()
        for group in _chunks(shuffled, batch_size):
            batch = batch_graphs(group)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), batch.y)
            loss.backward()
            optimizer.step()
    elapsed = time.perf_counter() - start
    return {
        "seed": seed,
        "train_accuracy": evaluate(model, train, batch_size),
        "test_accuracy": evaluate(model, test, batch_size),
        "training_seconds": elapsed,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }

def train_model(graphs, num_classes, architecture, seed, epochs=120,
                hidden_dim=32, batch_size=8, pooling="mean"):
    """Create a stratified holdout split and train one model."""
    train, test = stratified_split(graphs, seed)
    return train_on_split(train, test, num_classes, architecture, seed,
                          epochs, hidden_dim, batch_size, pooling)

def run_graph_experiment(graphs, num_classes, seeds=(11, 23, 37), epochs=120,
                         architectures=ARCHITECTURES, hidden_dim=32, batch_size=8,
                         dataset_name="custom"):
    """Compare architectures over repeated stratified holdout seeds."""
    results = {}
    for architecture in architectures:
        runs = [train_model(graphs, num_classes, architecture, seed, epochs,
                            hidden_dim, batch_size) for seed in seeds]
        test_scores = [r["test_accuracy"] for r in runs]
        train_scores = [r["train_accuracy"] for r in runs]
        results[architecture] = {
            "train_accuracy_mean": mean(train_scores),
            "test_accuracy_mean": mean(test_scores),
            "test_accuracy_population_sd": pstdev(test_scores),
            "training_seconds_mean": mean(r["training_seconds"] for r in runs),
            "parameter_count": runs[0]["parameter_count"],
            "per_seed": runs,
        }
    return {
        "dataset": dataset_name,
        "protocol": {
            "seeds": list(seeds), "epochs": epochs, "hidden_dim": hidden_dim,
            "batch_size": batch_size, "split": "stratified 70/30 per seed",
            "optimizer": "Adam", "learning_rate": 0.01, "weight_decay": 0.0001,
            "pooling": "mean", "architectures": list(architectures),
            "timing_note": "Training wall-clock time measured locally; hardware and software context must accompany comparisons."
        },
        "models": results,
    }

def run_synthetic_experiment(seeds=(11, 23, 37), epochs=120, hidden_dim=32, batch_size=8):
    """Run the controlled synthetic graph-family comparison and 1-WL diagnostic."""
    graphs = benchmark_graphs(feature_mode="constant")
    result = run_graph_experiment(graphs, 5, seeds, epochs, hidden_dim=hidden_dim,
                                  batch_size=batch_size, dataset_name="synthetic-graph-families")
    c10, two_c5 = cycle_graph(10), disjoint_cycles([5, 5])
    result["expressivity_diagnostic"] = {
        "pair": [c10.name, two_c5.name],
        "non_isomorphic_by_construction": True,
        "one_wl_equivalent_through_5_rounds": wl_equivalent(c10, two_c5, rounds=5),
        "interpretation": "The pair is 2-regular with identical initial node features. Matching 1-WL colour histograms demonstrate this specific collision; this is not a general graph-isomorphism test."
    }
    result["dataset_description"] = {
        "graph_count": len(graphs), "class_count": 5,
        "graph_families": ["path", "star", "wheel", "complete", "complete_bipartite"],
        "sizes_per_family": 6, "features": "constant scalar per node",
        "limitations": ["synthetic small dataset", "not a recognised benchmark", "not evidence of universal architecture superiority"]
    }
    return result

def main():
    """Run the synthetic sparse GNN experiment and write its JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("results/sparse_synthetic.json"))
    args = parser.parse_args()
    if args.epochs < 1 or args.hidden_dim < 1 or args.batch_size < 1:
        parser.error("epochs, hidden-dim, and batch-size must be positive")
    result = run_synthetic_experiment(tuple(args.seeds), args.epochs, args.hidden_dim, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
