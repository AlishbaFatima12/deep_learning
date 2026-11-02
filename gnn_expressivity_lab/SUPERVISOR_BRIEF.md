# Supervisor Brief

## Title
**Beyond Neighbourhood Aggregation: A Reproducible Study of GNN Expressivity and 1-WL Limitations**

## Summary
This portfolio project investigates the relationship between message-passing graph neural networks and the 1-dimensional Weisfeiler–Leman (1-WL) colour-refinement procedure. It includes dense reference GCN/GIN models, sparse edge-index implementations of GCN, GIN, GraphSAGE and single-head GAT, synthetic graph-family experiments, and an optional PyTorch Geometric TU dataset runner with repeated stratified k-fold evaluation.

A central theoretical diagnostic compares the non-isomorphic graphs C10 and C5 ⊔ C5. With identical initial node features, both are 2-regular and 1-WL retains matching colour histograms, despite the difference in connected components.

## Motivation and questions
Message-passing GNNs learn from relational structure, but local aggregation imposes representational limits. The project asks:
1. How do common aggregation rules differ in the structural information they preserve?
2. Which non-isomorphic graph pairs remain indistinguishable under 1-WL with controlled initial features?
3. How should theoretical expressivity diagnostics be separated from empirical classification performance?
4. What computational trade-offs appear when dense reference operations are replaced by sparse edge-index message passing?

## Implementation
- Graph generators and controlled node features.
- Deterministic 1-WL colour refinement.
- Dense reference GCN and GIN.
- Sparse GCN, GIN, GraphSAGE and single-head GAT.
- Batched graph classification, repeated seeds, train/test metrics, parameter counts and training wall-clock timing.
- Optional TU dataset loading through PyTorch Geometric.
- Repeated stratified k-fold evaluation and percentile-bootstrap confidence intervals.

## Method and evidence boundaries
The synthetic dataset contains five graph families and six sizes per family. It is a pipeline sanity check, not a recognised benchmark. The C10 versus C5 ⊔ C5 result is a theoretical 1-WL diagnostic and is not inferred from model accuracy.

The optional TU runner checks node features and class counts, remaps labels to contiguous indices, and provides repeated stratified folds. This is a research-oriented baseline, not automatically a publication-standard benchmark. Model selection should be separated from final evaluation; dataset-specific protocols and statistical treatment should be agreed before making comparative claims.

## Relevance to expressivity and efficiency
The project is strongest as a learning-oriented implementation of expressivity concepts. Sparse edge-index aggregation avoids materialising dense N × N adjacency matrices, but this does not establish lower wall-clock time or memory use on every workload. A rigorous efficiency study requires matched models, graph sizes, hardware/software versions, runtime and peak-memory measurement.

## Limitations
- No new theorem or architecture is claimed.
- Synthetic graph collection is small.
- GAT is a single-head reference implementation.
- Optional TU datasets require PyTorch Geometric installation.
- No energy measurement or systematic large-graph profiling yet.
- Results must be reproduced and inspected before quoting.

## Discussion points
- Which expressivity gap is most relevant to the group's research?
- Should the next step focus on higher-order WL/GNNs, structural encodings, or aggregation analysis?
- Which benchmark suite and evaluation protocol should be adopted?
- What is the most appropriate way to quantify efficiency alongside expressivity?

The work is an independently documented portfolio prototype and evidence of active GNN learning and implementation, not evidence of years of dedicated GNN research.
