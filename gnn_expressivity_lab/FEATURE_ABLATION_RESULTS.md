# Synthetic Pilot Results: Feature Sensitivity in Message-Passing GNNs

## Purpose

This pilot asks how the choice of initial node features changes what common message-passing architectures can learn from small graph structures. It is a controlled synthetic experiment, not a recognised graph-classification benchmark.

## Protocol

- Graphs: 30 total; five graph families (path, star, wheel, complete, complete bipartite), six graph sizes per family.
- Split: stratified 70/30 train/test.
- Architectures: sparse GCN, GIN, GraphSAGE and single-head GAT.
- Hidden width: 32; two message-passing layers; mean graph pooling.
- Optimiser: Adam, learning rate 0.01, weight decay 0.0001.
- Training: 120 epochs; batch size 8.
- Seeds: 11, 23, 37.
- Environment: Python 3.13.5; PyTorch 2.10.0+cpu; CPU; five PyTorch threads; Linux x86_64.
- Peak memory was not measured. Timing values are local and hardware dependent.

## Results

Test accuracy mean ± population standard deviation across the three seeds:

| Architecture | Constant node features | Degree node features |
|---|---:|---:|
| GCN | 46.7% ± 18.9% | 80.0% ± 0.0% |
| GIN | 86.7% ± 4.7% | 86.7% ± 4.7% |
| GraphSAGE | 20.0% ± 0.0% | 96.7% ± 4.7% |
| GAT (single-head) | 20.0% ± 0.0% | 96.7% ± 4.7% |

The constant-feature result is consistent with an important design property: when every node starts with the same representation, mean aggregation and attention can repeatedly aggregate indistinguishable messages and lose degree/multiplicity information. GIN's sum aggregation can preserve neighbourhood multiplicity. Supplying degree as an input feature changes the information available to all models, and GraphSAGE/GAT become able to classify these particular synthetic families.

A rerun on Python 3.13.7, PyTorch 2.9.1 (CPU, Windows) is committed as results/feature_ablation.json. It matched six of the eight cells above exactly; GIN with constant features gave 90.0% and GAT with degree features 100.0%, an ordinary version-to-version difference on a ten-graph test split.

## Interpretation and caution

1. The scores describe this small synthetic task and this exact training protocol only.
2. They do not establish that GIN is generally better than GraphSAGE or GAT.
3. The feature ablation is the key result: model behaviour is highly dependent on the information supplied at input.
4. The constant-feature task is deliberately restrictive and should be complemented by real graph datasets with meaningful node attributes.
5. The C10 versus C5 ⊔ C5 1-WL collision remains a separate theoretical diagnostic; classification accuracy does not prove it.

## Reproducibility command

From the gnn_expressivity_lab directory:

    gnn-sparse-experiment --epochs 120 --seeds 11 23 37 --feature-mode both --output results/feature_ablation.json

The repository's CI verifies the sparse models, tests, JSON output, and CLI wiring. The full synthetic run and the optional TU dataset evaluation should be rerun in the target research environment before quoting results in an application or supervisor meeting.

## Next experiment

Repeat the comparison on MUTAG and PROTEINS with repeated stratified folds, document dataset and software versions, and add a feature ablation that compares native node attributes, degree-only features, and constant features. Use nested model selection or a pre-registered fixed configuration to avoid tuning on evaluation folds.
