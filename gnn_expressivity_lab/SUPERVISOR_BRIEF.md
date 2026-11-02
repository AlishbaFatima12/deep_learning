# Supervisor Brief

## Title
**Beyond Neighbourhood Aggregation: A Reproducible Study of GNN Expressivity and 1-WL Limitations**

## Summary
This portfolio prototype investigates the relationship between message-passing graph neural networks and the 1-dimensional Weisfeiler–Leman (1-WL) colour-refinement procedure. It provides transparent PyTorch implementations of a GCN and a GIN, a 1-WL diagnostic, controlled synthetic graph families, and repeated-seed graph-classification experiments. A central diagnostic compares the non-isomorphic graphs C10 and C5 union C5: with identical initial node features, both are 2-regular and remain indistinguishable to 1-WL. The aim is to build a mathematically grounded foundation for deeper work on expressivity, aggregation functions, and structural information in GNNs.

## Motivation
Graph neural networks learn from relational structure rather than treating observations as independent feature vectors. However, message-passing design imposes representational limits: repeated local aggregation does not necessarily encode all global graph structure. Understanding these limits requires graph-theoretic reasoning and controlled empirical comparisons.

## Questions
- RQ1: How do GCN- and GIN-style aggregation rules differ in structural information retained?
- RQ2: Which non-isomorphic graph pairs remain indistinguishable under 1-WL when initial node features are controlled?
- RQ3: How should theoretical expressivity diagnostics be distinguished from ordinary classification accuracy?

## Method
1. Construct graph families with explicit edge lists and controlled node features.
2. Compute 1-WL colour histograms across refinement rounds.
3. Implement GCN and GIN layers in PyTorch with a shared graph-classification protocol.
4. Repeat training under fixed seeds and report descriptive mean and dispersion.
5. Interpret empirical results separately from the theoretical WL diagnostic.

## Relevance to the theory-and-efficiency direction
The current version is strongest on the expressivity/theory side: it makes the 1-WL comparison explicit and provides controlled GCN/GIN implementations. It does not yet establish computational efficiency. The reference implementation uses dense adjacency matrices for clarity, which incurs quadratic adjacency storage; a meaningful efficiency phase should add sparse message passing (for example, PyTorch Geometric), compare parameter-matched models, and report runtime and memory under a documented hardware/software setup. This limitation should be stated openly in any supervisor discussion.

## Scope and boundaries
This is an educational/research portfolio prototype. It does not claim a new theorem, novel architecture, state-of-the-art performance, or validation on standard real-world graph benchmarks. The synthetic experiment is intended to verify the pipeline and expose modelling choices. A next phase would add recognised datasets, stronger baselines, systematic ablations, confidence intervals, and a comprehensive graph-pair evaluation suite.

## Discussion points
- Which expressivity questions are most relevant to the group's current research agenda?
- Would higher-order WL/GNNs, positional/structural encodings, or aggregation analysis be the strongest next direction?
- Which benchmark suite and evaluation protocol should be prioritised?
- Which theoretical prerequisites should be strengthened: graph isomorphism, combinatorics, algebra, approximation, or spectral graph theory?

The work connects graph theory and machine-learning foundations to practical neural-network implementation. It is appropriate as a discussion artifact for a prospective doctoral researcher developing GNN specialisation, not as evidence of years of dedicated GNN research.
