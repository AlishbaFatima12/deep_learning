# Beyond Neighbourhood Aggregation: A Reproducible GNN Expressivity Lab

**Research portfolio project | PyTorch | Graph theory × machine learning**

> What structural information can message-passing graph neural networks preserve, and where do their expressive limits arise?

This project is a transparent research-learning implementation connecting graph-theoretic reasoning to neural models. It includes dense reference GCN/GIN models, sparse edge-index implementations of GCN/GIN/GraphSAGE/GAT, a 1-WL diagnostic, a synthetic experiment, and optional real-world graph classification using PyTorch Geometric TU datasets.

## Start here

- [VISUAL_GUIDE.md](VISUAL_GUIDE.md) — diagrams and step-by-step explanation.
- [PRESENTATION.md](PRESENTATION.md) — supervisor presentation narrative.
- [SUPERVISOR_BRIEF.md](SUPERVISOR_BRIEF.md) — concise research overview.

## Main research idea

Message-passing GNNs update node representations by aggregating information from neighbours. This creates a close connection to the 1-dimensional Weisfeiler–Leman (1-WL) colour-refinement procedure.

A controlled diagnostic compares the 10-cycle C10 with the disjoint union C5 ⊔ C5. These graphs are non-isomorphic (one connected component versus two), but every vertex has degree two and all initial node features are identical. 1-WL therefore retains matching colour histograms at every round. Standard message-passing GNNs bounded by 1-WL inherit this limitation under the same initial-feature setup.

## Implemented components

| Component | Purpose |
|---|---|
| graphs.py | Graph object and synthetic graph generators |
| wl.py | Deterministic 1-WL colour refinement |
| models.py | Dense reference GCN and GIN |
| sparse_models.py | Sparse edge-index GCN, GIN, GraphSAGE and single-head GAT |
| sparse_experiment.py | Batched synthetic training, repeated seeds, accuracy, parameter counts and timing |
| pyg_benchmark.py | Optional TU benchmark loader and experiment runner |
| tests/ | Unit tests for graph logic, sparse models, gradients and batching |

## Installation

Python 3.10+ and PyTorch 2.1+.

    cd gnn_expressivity_lab
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[dev]"
    pytest -q

For optional real-world TU datasets:

    python -m pip install -e ".[dev,benchmark]"

## Run experiments

Fast synthetic smoke test:

    gnn-sparse-experiment --epochs 5 --seeds 11 --output results/smoke.json

Default synthetic comparison:

    gnn-sparse-experiment --epochs 120 --seeds 11 23 37 --output results/sparse_synthetic.json

Optional TU dataset experiment (downloads the dataset on first use):

    gnn-benchmark --dataset MUTAG --epochs 100 --seeds 11 23 37 --output results/MUTAG.json

MUTAG and PROTEINS are examples of datasets with node features. Dataset-specific feature handling and benchmark-standard cross-validation must be checked before making published-benchmark comparisons.

## Experimental protocol

The synthetic task contains five graph families and six graph sizes per family (30 graphs), constant scalar node features, stratified 70/30 train/test splits, and repeated random seeds. Four sparse message-passing architectures are compared with a shared hidden dimension, optimiser, pooling, and training budget.

The output reports per-seed train/test accuracy, mean and population standard deviation, parameter count, and measured training wall-clock time. Runtime is hardware- and environment-dependent; only compare timings collected on the same documented setup.

The C10 versus C5 ⊔ C5 diagnostic is separate from classification accuracy. Accuracy is not proof of theoretical expressivity; the 1-WL result is a specific theoretical diagnostic, not a complete graph-isomorphism solver.

## Complexity note

The sparse layers operate on edge lists and aggregate messages with indexed reductions rather than constructing a dense N × N adjacency matrix. This avoids the dense adjacency storage bottleneck, although runtime still depends on edge count, feature width, hardware, and implementation overhead. The GAT implementation is a clear single-head reference, not a tuned high-performance kernel.

## Limitations and integrity

- The synthetic set is small and is not a recognised benchmark.
- The TU runner provides a basic stratified holdout protocol, not the repeated-fold protocol required for benchmark-standard claims.
- No novel theorem or architecture is claimed.
- Timing measurements are local and hardware-dependent.
- The implementation is a research-learning artifact; all results must be reproduced and inspected before quoting.

## Next research steps

1. Run tests and reproduce the saved outputs.
2. Use repeated stratified folds and report confidence intervals on TU datasets.
3. Add parameter-matched and depth-matched ablations.
4. Evaluate structural encodings and systematic 1-WL-indistinguishable graph pairs.
5. Document hardware, software versions, runtime and memory consistently.
6. Discuss a focused theoretical question with a graph-learning supervisor.

## References

- Kipf & Welling (2017), Semi-Supervised Classification with Graph Convolutional Networks. https://arxiv.org/abs/1609.02907
- Xu et al. (2019), How Powerful are Graph Neural Networks? https://arxiv.org/abs/1810.00826
- Morris et al. (2019), Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks. https://arxiv.org/abs/1810.02244
- Veličković et al. (2018), Graph Attention Networks. https://arxiv.org/abs/1710.10903
- Hamilton et al. (2017), Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216

