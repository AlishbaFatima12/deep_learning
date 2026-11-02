"""Reproducible synthetic experiment for GCN/GIN expressivity diagnostics."""
import argparse
import json
import random
from pathlib import Path
from statistics import mean, pstdev
import torch
from torch import nn
from .graphs import benchmark_graphs, cycle_graph, disjoint_cycles
from .models import GraphClassifier
from .wl import wl_equivalent

def set_seed(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed)

def stratified_split(graphs, seed: int, train_fraction: float = 0.7):
    rng = random.Random(seed); train, test = [], []
    for label in sorted({g.label for g in graphs}):
        group = [g for g in graphs if g.label == label]; rng.shuffle(group)
        cut = max(1, min(len(group) - 1, round(len(group) * train_fraction)))
        train.extend(group[:cut]); test.extend(group[cut:])
    rng.shuffle(train); rng.shuffle(test)
    return train, test

def evaluate(model, graphs):
    model.eval(); correct = 0
    with torch.no_grad():
        for graph in graphs: correct += int(model(graph).argmax().item() == graph.label)
    return correct / max(1, len(graphs))

def train_once(architecture: str, seed: int, epochs: int = 120):
    set_seed(seed); graphs = benchmark_graphs(feature_mode="constant"); train, test = stratified_split(graphs, seed)
    model = GraphClassifier(1, 32, 5, architecture=architecture, pooling="sum", dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4); loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train(); random.shuffle(train)
        for graph in train:
            optimizer.zero_grad(); loss = loss_fn(model(graph).unsqueeze(0), torch.tensor([graph.label]))
            loss.backward(); optimizer.step()
    return evaluate(model, train), evaluate(model, test)

def run_experiment(seeds=(11, 23, 37), epochs=120):
    results = {}
    for architecture in ("gcn", "gin"):
        scores = [train_once(architecture, seed, epochs) for seed in seeds]
        train_scores = [v[0] for v in scores]; test_scores = [v[1] for v in scores]
        results[architecture] = {
            "train_accuracy_mean": mean(train_scores), "test_accuracy_mean": mean(test_scores),
            "test_accuracy_population_sd": pstdev(test_scores),
            "per_seed": [{"seed": seed, "train_accuracy": tr, "test_accuracy": te} for seed, (tr, te) in zip(seeds, scores)],
        }
    c10, two_c5 = cycle_graph(10), disjoint_cycles([5, 5])
    results["expressivity_diagnostic"] = {
        "pair": [c10.name, two_c5.name], "non_isomorphic_by_construction": True,
        "one_wl_equivalent_through_5_rounds": wl_equivalent(c10, two_c5, rounds=5),
        "interpretation": "With constant initial node features, 1-WL cannot distinguish these 2-regular graphs; standard message-passing GNNs bounded by 1-WL inherit this limitation.",
    }
    results["protocol"] = {
        "dataset": "synthetic graph-family sanity check; 5 classes x 6 graph sizes",
        "features": "constant scalar per node", "split": "stratified 70/30, independently seeded",
        "seeds": list(seeds), "epochs": epochs,
        "limitations": ["synthetic data only", "small sample size", "no hyperparameter search", "not a published benchmark", "accuracy is not evidence of theoretical expressivity"],
    }
    return results

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=120); parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--output", type=Path, default=Path("results/summary.json")); args = parser.parse_args()
    if args.epochs < 1: parser.error("--epochs must be positive")
    result = run_experiment(tuple(args.seeds), args.epochs); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\\n", encoding="utf-8"); print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
