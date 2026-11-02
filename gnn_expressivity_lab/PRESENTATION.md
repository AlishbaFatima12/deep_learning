# Supervisor Presentation: GNN Expressivity and the Limits of Message Passing

*A visual research discussion. Pair this narrative with the PowerPoint deck and diagrams in VISUAL_GUIDE.md.*

## 1 — Research question
**What structural information can message-passing GNNs distinguish, and where are their limits?**
- Graph theory + PyTorch implementation.
- GCN, GIN, GraphSAGE, and single-head GAT.
- 1-WL colour-refinement diagnostic.

Opening: “I wanted to understand not only how graph neural networks learn from relationships, but also what their aggregation mechanisms may fail to distinguish.”

## 2 — Why graph learning?
Use a small network diagram.
- Nodes represent entities.
- Edges represent relationships.
- Node features describe entities.
- The neighbourhood is part of the input, not an optional column.

## 3 — End-to-end architecture
Use the pipeline diagram in VISUAL_GUIDE.md.
1. Construct graph examples.
2. Set controlled node features.
3. Run 1-WL diagnostic.
4. Train four message-passing architectures.
5. Evaluate on held-out graphs.
6. Report metrics and limitations.

## 4 — Message passing
Show the generic update:
\[
h_v^{(k)} = \mathrm{UPDATE}\left(h_v^{(k-1)}, \mathrm{AGGREGATE}\{h_u^{(k-1)}:u\in N(v)\}\right)
\]
Explain one-hop neighbourhoods, stacked layers, node embeddings, and graph readout.

## 5 — Four architectures
| Model | Aggregation | Intuition |
|---|---|---|
| GCN | Symmetrically degree-normalised messages | Smooth neighbourhood mixing |
| GIN | Sum + MLP | Preserve multiset distinctions |
| GraphSAGE | Mean + self transformation | Inductive neighbourhood representation |
| GAT | Learned attention weights | Learn which neighbours to emphasise |

Important: the code is a transparent reference implementation. GAT is single-head; parameter counts differ; no universal superiority claim is made.

## 6 — 1-WL colour refinement
Explain the steps:
1. Initialise colours from node features.
2. Build a signature from each node's colour and its neighbours' colour multiset.
3. Recolour.
4. Compare colour histograms over rounds.

## 7 — The C10 versus C5 ⊔ C5 example
Draw both graphs.
- Both have 10 vertices and degree 2 at every vertex.
- All initial node features are identical.
- One graph is connected; the other has two components.
- 1-WL retains matching colour histograms.

Interpretation: this is a specific non-isomorphic pair that 1-WL cannot distinguish under the given feature setup. Standard 1-WL-bounded message-passing GNNs inherit this limitation.

## 8 — What was implemented?
- graphs.py: graph structures and synthetic graph families.
- wl.py: deterministic 1-WL refinement.
- models.py: dense reference GCN/GIN.
- sparse_models.py: edge-index GCN/GIN/GraphSAGE/GAT.
- sparse_experiment.py: batched synthetic training and metrics.
- pyg_benchmark.py: optional TU dataset loading.
- cross_validation.py: repeated stratified k-fold evaluation and bootstrap intervals.

## 9 — Synthetic experimental protocol
- Five synthetic graph families × six sizes = 30 graphs.
- Constant scalar node features.
- Stratified 70/30 holdout per seed.
- Four architectures; shared hidden width, optimiser and pooling.
- Seeds 11, 23, 37 by default.
- Report train/test accuracy, dispersion, parameter count and training time.

Do not present this small synthetic task as a recognised benchmark.

## 10 — Real-world dataset protocol
Optional PyTorch Geometric TU datasets, for example MUTAG and PROTEINS.
- Dataset download is explicit and occurs when the command is run.
- Node features and class balance are checked.
- Labels are remapped to contiguous indices.
- Repeated stratified k-fold evaluation is available.
- Report mean fold accuracy, sample SD, bootstrap 95% interval, parameter count and time.

Explain that this is a research-oriented baseline, not automatically a benchmark-standard result. Model selection should be separated from final evaluation.

## 11 — Understanding the output
Each JSON report records protocol settings and per-run metrics.
- Accuracy: performance on the specified split only.
- Standard deviation: variability across observed runs/folds.
- Bootstrap interval: descriptive uncertainty summary.
- Parameter count: model size, not memory usage.
- Training seconds: local wall-clock measurement, hardware dependent.
- 1-WL result: separate theoretical diagnostic, not model accuracy.

Only show numeric scores from an actually executed and inspected output file.

## 12 — Efficiency and scalability
Sparse edge-index layers avoid constructing a dense N × N adjacency matrix. This reduces the adjacency representation from quadratic storage to edge-based storage, but does not by itself prove faster execution. Runtime and memory must be measured on documented hardware, with matched settings and larger graphs.

## 13 — Learning and current limitations
**Learning:** graph representation, message passing, aggregation, GCN/GIN/SAGE/GAT, 1-WL, reproducible training, stratified evaluation, and careful interpretation of metrics.

**Limitations:** small synthetic dataset; single-head reference GAT; TU runner needs optional PyG installation; repeated-fold protocol still needs benchmark-specific model selection; no novel theorem or architecture; no hardware memory/energy study.

## 14 — Next research questions
- Which graph structures or feature conditions expose meaningful gaps between aggregation rules?
- Would structural encodings or higher-order WL/GNNs address the selected limitation?
- Which benchmark protocol is appropriate for the group's current research direction?
- How should expressivity be studied jointly with computational cost?

Closing: “This is a reproducible starting point, not a claimed research breakthrough. I would value guidance on narrowing it into a rigorous question that fits the group's research agenda.”
