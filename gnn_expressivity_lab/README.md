# Beyond Neighbourhood Aggregation: A Reproducible GNN Expressivity Lab

**Research portfolio project | PyTorch | Graph theory × machine learning**

> What structural information can message-passing graph neural networks preserve, and where do their expressive limits arise?

This project connects graph-theoretic reasoning to practical neural implementations. It includes dense reference GCN/GIN models, sparse edge-index GCN/GIN/GraphSAGE/GAT implementations, a 1-WL diagnostic, synthetic experiments, and optional real-world TU graph classification.

## Start here

- [VISUAL_GUIDE.md](VISUAL_GUIDE.md) — diagrams and step-by-step explanation.
- [PRESENTATION.md](PRESENTATION.md) — supervisor presentation narrative.
- [SUPERVISOR_BRIEF.md](SUPERVISOR_BRIEF.md) — concise research overview.
- [FEATURE_ABLATION_RESULTS.md](FEATURE_ABLATION_RESULTS.md) — preliminary synthetic results with limitations.

## Research question

Message-passing GNNs update node representations by aggregating information from neighbours. This creates a close connection to the 1-dimensional Weisfeiler–Leman (1-WL) colour-refinement procedure.

A controlled diagnostic compares the 10-cycle C10 with the disjoint union C5 ⊔ C5. These graphs are non-isomorphic (one connected component versus two), but every vertex has degree two and all initial node features are identical. 1-WL therefore retains matching colour histograms at every round. Standard message-passing GNNs bounded by 1-WL inherit this limitation under the same initial-feature setup.

## Architecture inventory

| Architecture | Aggregation | Implementation |
|---|---|---|
| GCN | Symmetrically degree-normalised messages | Dense reference + sparse edge-index |
| GIN | Sum aggregation + MLP | Dense reference + sparse edge-index |
| GraphSAGE | Mean neighbour aggregation + self transform | Sparse edge-index |
| GAT | Learned destination-wise attention | Sparse single-head reference |

Sparse layers aggregate over edge lists and do not construct dense N × N adjacency matrices. The GAT implementation is intentionally readable and is not a tuned multi-head production implementation.

## Installation

Python 3.10+ and PyTorch 2.1+.

    cd gnn_expressivity_lab
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[dev]"
    pytest -q

For optional PyTorch Geometric TU datasets:

    python -m pip install -e ".[dev,benchmark]"

## Run experiments

Synthetic comparison:

    gnn-sparse-experiment --epochs 120 --seeds 11 23 37 --feature-mode both --output results/feature_ablation.json

Real-world graph benchmark with repeated stratified folds:

    gnn-benchmark --dataset MUTAG --folds 10 --repeats 3 --epochs 100 --output results/MUTAG_kfold.json

Other TU datasets with suitable node features, such as PROTEINS, can be selected with --dataset PROTEINS. Dataset download occurs on first run. The runner checks for node features and adequate per-class sample counts.

## Evaluation protocol and outputs

The synthetic dataset contains five graph families and six sizes per family (30 graphs), with a paired feature ablation: constant scalar features versus degree-valued node features. Both conditions use stratified 70/30 holdouts and repeated seeds. It reports per-seed train/test accuracy, mean and population standard deviation, parameter count, and measured training wall-clock time.

The optional TU runner uses repeated stratified k-fold evaluation. It reports fold scores, mean, sample standard deviation, percentile-bootstrap 95% confidence intervals, parameter counts, and training time. The confidence interval is descriptive over observed fold scores; it is not a substitute for benchmark-specific statistical testing or the standard protocol required by a paper.

Runtime is hardware- and environment-dependent. Record CPU/GPU, operating system, Python, PyTorch, and PyG versions before comparing timings. Parameter count is not a memory measurement.

The C10 versus C5 ⊔ C5 diagnostic is separate from classification accuracy. Accuracy is not proof of theoretical expressivity; the 1-WL result is a specific diagnostic, not a complete graph-isomorphism solver.

## Limitations and research integrity

- The synthetic task is small and not a recognised benchmark.
- The TU evaluation is a research-oriented repeated-fold baseline, not a claim of state-of-the-art performance.
- Hyperparameter selection must be nested or otherwise separated from final test folds for publication-grade evaluation.
- No novel theorem or architecture is claimed.
- Runtime measurements are local and hardware-dependent.
- Results must be reproduced and inspected before quoting.

## Research next steps

1. Validate the tests and reproduce experiment JSON.
2. Add parameter/depth-matched ablations.
3. Study structural encodings and systematic 1-WL-indistinguishable pairs.
4. Record memory as well as runtime and parameter count.
5. Use supervisor feedback to narrow the theoretical contribution.

## References

- Kipf & Welling (2017), Semi-Supervised Classification with Graph Convolutional Networks. https://arxiv.org/abs/1609.02907
- Xu et al. (2019), How Powerful are Graph Neural Networks? https://arxiv.org/abs/1810.00826
- Morris et al. (2019), Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks. https://arxiv.org/abs/1810.02244
- Veličković et al. (2018), Graph Attention Networks. https://arxiv.org/abs/1710.10903
- Hamilton et al. (2017), Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216
