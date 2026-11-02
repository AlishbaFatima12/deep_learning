"""Repeated stratified k-fold evaluation for graph-classification datasets."""
import argparse
import json
import random
from pathlib import Path
from statistics import mean, stdev
from .sparse_experiment import ARCHITECTURES, train_on_split, environment_metadata

def stratified_kfold_indices(labels, folds: int, seed: int):
    """Return (train_indices, test_indices) for deterministic stratified folds."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    if not groups or min(len(indices) for indices in groups.values()) < folds:
        raise ValueError("Each class must contain at least as many examples as folds")
    rng = random.Random(seed)
    test_folds = [[] for _ in range(folds)]
    for label in sorted(groups):
        indices = groups[label][:]
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            test_folds[position % folds].append(index)
    all_indices = set(range(len(labels)))
    result = []
    for test in test_folds:
        test_set = set(test)
        train = sorted(all_indices - test_set)
        result.append((train, sorted(test)))
    return result

def bootstrap_confidence_interval(values, seed: int = 11, samples: int = 2000):
    """Percentile bootstrap CI for the mean of observed fold scores."""
    if not values:
        raise ValueError("values must not be empty")
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [rng.choice(values) for _ in values]
        means.append(sum(draw) / len(draw))
    means.sort()
    low = means[int(0.025 * (samples - 1))]
    high = means[int(0.975 * (samples - 1))]
    return [low, high]

def run_kfold_experiment(graphs, num_classes, folds=10, repeats=3, seed=11,
                         epochs=100, hidden_dim=32, batch_size=16,
                         architectures=ARCHITECTURES, dataset_name="dataset"):
    """Train/evaluate each architecture over repeated stratified k-fold splits."""
    labels = [int(graph.label) for graph in graphs]
    if repeats < 1:
        raise ValueError("repeats must be positive")
    results = {}
    for architecture in architectures:
        runs = []
        for repeat in range(repeats):
            split_seed = seed + repeat
            splits = stratified_kfold_indices(labels, folds, split_seed)
            for fold, (train_idx, test_idx) in enumerate(splits):
                train_graphs = [graphs[i] for i in train_idx]
                test_graphs = [graphs[i] for i in test_idx]
                run_seed = seed + repeat * folds + fold
                metrics = train_on_split(train_graphs, test_graphs, num_classes,
                                         architecture, run_seed, epochs,
                                         hidden_dim, batch_size)
                metrics.update({"repeat": repeat + 1, "fold": fold + 1})
                runs.append(metrics)
        scores = [run["test_accuracy"] for run in runs]
        results[architecture] = {
            "test_accuracy_mean": mean(scores),
            "test_accuracy_sample_sd": stdev(scores) if len(scores) > 1 else 0.0,
            "test_accuracy_bootstrap_95pct_ci": bootstrap_confidence_interval(scores, seed),
            "training_seconds_mean": mean(run["training_seconds"] for run in runs),
            "test_inference_seconds_mean": mean(run["test_inference_seconds"] for run in runs),
            "parameter_count": runs[0]["parameter_count"],
            "parameter_bytes_fp32_estimate": runs[0]["parameter_bytes_fp32_estimate"],
            "per_fold": runs,
        }
    return {
        "dataset": dataset_name,
        "protocol": {
            "evaluation": "repeated stratified k-fold",
            "folds": folds, "repeats": repeats, "seed": seed,
            "epochs": epochs, "hidden_dim": hidden_dim,
            "batch_size": batch_size, "optimizer": "Adam",
            "learning_rate": 0.01, "weight_decay": 0.0001,
            "architectures": list(architectures),
            "confidence_interval": "percentile bootstrap over observed fold scores; descriptive, not a substitute for benchmark-specific statistical testing",
            "timing_note": "Training wall-clock time is hardware- and environment-dependent."
        },
        "environment": environment_metadata(),
        "models": results,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="MUTAG")
    parser.add_argument("--root", default="data/TUDataset")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("results/MUTAG_kfold.json"))
    args = parser.parse_args()
    if min(args.folds, args.repeats, args.epochs, args.hidden_dim, args.batch_size) < 1:
        parser.error("folds, repeats, epochs, hidden-dim, and batch-size must be positive")
    from .pyg_benchmark import load_tu_graphs
    graphs, num_classes, metadata = load_tu_graphs(args.dataset, args.root)
    result = run_kfold_experiment(
        graphs, num_classes, args.folds, args.repeats, args.seed,
        args.epochs, args.hidden_dim, args.batch_size, dataset_name=args.dataset
    )
    result["dataset_metadata"] = metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
