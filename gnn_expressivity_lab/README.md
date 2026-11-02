# Beyond Neighbourhood Aggregation: A Reproducible GNN Expressivity Lab

**Research portfolio project | PyTorch | Graph theory × machine learning**

> What structural information can message-passing graph neural networks preserve, and where do their expressive limits arise?

This project implements Graph Convolutional Networks (GCNs), Graph Isomorphism Networks (GINs), and a 1-dimensional Weisfeiler–Leman (1-WL) colour-refinement diagnostic. It is a transparent research-learning artifact connecting graph-theoretic reasoning to neural implementation.

## Mathematical motivation

Message-passing GNNs update node representations by aggregating information from neighbouring nodes. This creates a close connection to the 1-WL graph-isomorphism heuristic. Xu et al. (2019) show that suitably injective aggregation can match 1-WL discriminative power, motivating GIN; common GCN aggregation is generally less expressive.

A controlled example compares the 10-cycle, C10, with the disjoint union of two 5-cycles, C5 union C5. These graphs are non-isomorphic: one is connected and the other has two components. Yet every vertex has degree two and all initial node features are identical. 1-WL therefore keeps every vertex in the same colour class at every refinement round. More message-passing layers do not resolve this collision under the standard 1-WL-bounded message-passing framework.

## Implemented

- Explicit undirected graph construction and graph-family generators.
- Constant node features to isolate structural information.
- Deterministic 1-WL colour refinement and colour histograms.
- Two-layer symmetrically normalised GCN and learnable-epsilon GIN.
- Graph-level sum/mean pooling and classification head.
- Stratified train/test split and repeated seeded experiments.
- JSON results with mean accuracy and across-seed population standard deviation.
- Unit tests for adjacency, WL collision, and model forward passes.

## Layout

- src/gnn_expressivity_lab/graphs.py — graph data structure and generators
- src/gnn_expressivity_lab/wl.py — 1-WL refinement and comparison
- src/gnn_expressivity_lab/models.py — GCN, GIN, graph-level classifier
- src/gnn_expressivity_lab/experiment.py — seeded training and evaluation
- tests/test_core.py — core correctness checks
- SUPERVISOR_BRIEF.md — concise research overview
- PRESENTATION.md — 10-slide supervisor discussion narrative

## Run locally

Python 3.10+ is recommended.

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[dev]"
    pytest -q
    python -m gnn_expressivity_lab.experiment --epochs 120 --seeds 11 23 37

The experiment writes results/summary.json. It is intentionally small enough for a CPU.

## Experimental protocol

The synthetic classification task contains five graph families (paths, stars, wheels, complete graphs, complete bipartite graphs), six sizes per family, constant scalar node features, and stratified 70/30 splits. GCN and GIN use the same training protocol and three default random seeds.

The C10 versus C5 union C5 result is a separate theoretical diagnostic. Do not treat classification accuracy as proof of expressivity. The WL collision is the mathematical observation; empirical accuracy is only a pipeline sanity check.

## Research questions

1. How do aggregation and readout choices affect structural information retained by message-passing GNNs?
2. Under what graph and feature conditions do GCN/GIN-style models collapse non-isomorphic structures to indistinguishable representations?
3. How should theoretical expressivity bounds be distinguished from empirical graph-classification performance?

## Limitations

- Synthetic, small dataset; not a recognised benchmark.
- No new theorem or architecture is claimed.
- 1-WL is a diagnostic, not a complete graph-isomorphism solver.
- Training results depend on seeds and optimisation choices.
- No hardware speed, energy, or deployment-efficiency claims.
- This is a foundation for further work, not a completed PhD study.

## Next research-grade extensions

1. Add recognised graph datasets and documented splits (e.g. MUTAG, PROTEINS, IMDB-BINARY).
2. Compare GCN, GIN, GraphSAGE, and GAT under parameter-matched settings.
3. Build systematic 1-WL-indistinguishable graph-pair suites and vary initial features.
4. Add confidence intervals, ablations, parameter counts, runtime, and memory measurements.
5. Separate theoretical claims, implementation checks, and empirical observations in all reports.

## References

- Xu et al. (2019), How Powerful are Graph Neural Networks?, ICLR. https://arxiv.org/abs/1810.00826
- Morris et al. (2019), Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks, AAAI. https://arxiv.org/abs/1810.02244
- Kipf & Welling (2017), Semi-Supervised Classification with Graph Convolutional Networks. https://arxiv.org/abs/1609.02907
- Maron et al. (2019), Provably Powerful Graph Networks, NeurIPS. https://arxiv.org/abs/1905.11136

## Academic integrity

This repository is an independently documented portfolio prototype. Describe it as an implementation and exploratory study, not as a peer-reviewed result or proof of prior specialist GNN research. Run the tests and experiment, inspect the output, and be prepared to explain the mathematics and code before using it in an application or supervisor meeting.
