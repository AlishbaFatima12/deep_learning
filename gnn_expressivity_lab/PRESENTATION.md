# Supervisor Presentation: GNN Expressivity and the Limits of Message Passing

*A visual 12-slide discussion plan. Use the diagrams in VISUAL_GUIDE.md; keep theoretical claims separate from empirical results.*

## Slide 1 — The research question
**What can message-passing GNNs distinguish, and what structural information can they lose?**
- PyTorch implementation: GCN and GIN
- Graph-theoretic diagnostic: 1-WL colour refinement
- Controlled synthetic graph experiments

Speaker: “I wanted to understand both how these models learn from graph structure and where their mathematical limits come from.”

## Slide 2 — Why graphs?
Visual: draw a small social/citation network beside a table.
- Nodes represent entities.
- Edges represent relationships.
- Node features describe entities.
- The neighbourhood is part of the input, not just an extra column.

## Slide 3 — The complete pipeline
Visual: use the end-to-end Mermaid flowchart in VISUAL_GUIDE.md.
1. Construct graphs and features.
2. Run the 1-WL diagnostic.
3. Train GCN and GIN.
4. Pool node embeddings into graph embeddings.
5. Evaluate over fixed seeds.
6. Save JSON and interpret results cautiously.

## Slide 4 — How message passing works
Visual: use the node-feature → neighbour messages → aggregation → update → readout diagram.
- One layer exchanges information across one-hop neighbourhoods.
- Stacked layers expand the receptive field.
- Readout converts node representations into one graph representation.

## Slide 5 — GCN versus GIN
Show both update equations:
- GCN: (H' = \hat D^{-1/2}\hat A\hat D^{-1/2}HW)
- GIN: (h_v' = MLP((1+\epsilon)h_v + \sum_{u\in N(v)}h_u))

Explain: GCN uses symmetric degree normalisation; GIN uses sum aggregation followed by an MLP. GIN can match 1-WL under suitable assumptions; this is not a claim that every trained implementation achieves the bound.

## Slide 6 — The 1-WL diagnostic
Visual: colour refinement rounds.
- Start from node features.
- Each node receives a signature from its current colour and multiset of neighbour colours.
- Recolour and compare histograms.
- Matching histograms mean the 1-WL procedure did not distinguish the pair.

## Slide 7 — The key counterexample
Visual: (C_{10}) beside (C_5 \sqcup C_5).
- Both have 10 vertices.
- Every vertex has degree 2.
- All initial features are identical.
- One graph is connected; the other has two components.
- 1-WL keeps a single colour class in both graphs.

Key message: non-isomorphic graphs can remain indistinguishable to 1-WL. Standard 1-WL-bounded message-passing GNNs inherit this limitation under the same feature setup.

## Slide 8 — What was built?
Use a four-block architecture diagram:
- graphs.py: graph constructors and features
- wl.py: colour refinement
- models.py: GCN, GIN, pooling and classifier
- experiment.py: splits, training, metrics and JSON

Mention unit tests separately from model-performance results.

## Slide 9 — Experimental method
- Five synthetic graph families, six graph sizes per family.
- Constant scalar node features.
- Stratified 70/30 train/test split.
- Two-layer GCN and GIN.
- Adam optimiser; cross-entropy objective.
- Default seeds: 11, 23, 37.
- Report train/test accuracy, mean test accuracy, and population standard deviation.

## Slide 10 — Understanding the output
Show an example JSON schema, not fabricated scores:
```json
{
  "gcn": {
    "train_accuracy_mean": "...",
    "test_accuracy_mean": "...",
    "test_accuracy_population_sd": "...",
    "per_seed": []
  },
  "gin": {},
  "expressivity_diagnostic": {},
  "protocol": {}
}
```
Explain: accuracy checks the synthetic classification pipeline; the WL diagnostic addresses a theoretical limitation. They answer different questions. Only show actual numeric values after running and checking the experiment.

## Slide 11 — What I learned and what remains
**Learning:** graph representations, message passing, normalisation versus sum aggregation, 1-WL, reproducible training, and the distinction between empirical and theoretical evidence.

**Current limits:** synthetic data, dense adjacency matrices, no PyG, no GraphSAGE/GAT, no hardware measurements, no novel theorem or architecture.

## Slide 12 — Next steps and discussion
1. Verify tests and reproduce the current experiment.
2. Move to sparse operations / PyTorch Geometric.
3. Add recognised graph benchmarks.
4. Add GraphSAGE and GAT baselines.
5. Run controlled ablations and measure runtime, memory, and parameter counts.
6. Refine one focused theoretical question with supervisor guidance.

Questions:
- Which expressivity gap is most relevant to the group?
- Should the next step prioritise higher-order WL methods, structural encodings, or aggregation analysis?
- Which benchmark and statistical protocol should be adopted?

Closing: “This is a transparent starting point. I am using it to build a stronger foundation in graph learning and to identify a precise question that can be studied rigorously.”
