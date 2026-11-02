# Supervisor Presentation: GNN Expressivity and the Limits of Message Passing

*A 10-slide research conversation deck. Keep claims conservative and invite technical discussion.*

---

## 1 — Beyond Neighbourhood Aggregation
**A reproducible study of GNN expressivity and 1-WL limitations**  
PyTorch implementation · Graph theory · Controlled experiments

Opening: “I wanted to understand not only how graph neural networks learn, but also what structural information their message-passing mechanism can fundamentally fail to distinguish.”

## 2 — Why graphs need their own learning models
- Tabular learning often treats rows as independent examples.
- In graphs, entities are connected; neighbourhoods are part of the signal.
- GNNs learn node representations by repeatedly aggregating information from neighbours.

Visual suggestion: node features → neighbourhood aggregation → node embeddings → graph readout.

## 3 — The mathematical question
**When can two different graphs look identical to a message-passing GNN?**
- Local aggregation compresses a multiset of neighbour representations.
- More rounds expand the receptive field but do not guarantee complete structural identification.
- The 1-dimensional Weisfeiler–Leman (1-WL) test is a central theoretical comparison.

## 4 — GCN versus GIN
**GCN:** degree-normalised neighbourhood aggregation.  
**GIN:** sum aggregation followed by an MLP, designed to match 1-WL discriminative power under suitable assumptions.

| Model | Aggregation intuition | Main point |
|---|---|---|
| GCN | Normalised neighbourhood aggregation | Normalisation can lose multiset distinctions |
| GIN | Sum + MLP | Stronger multiset discrimination, still bounded by 1-WL |

## 5 — A concrete counterexample
Compare C10 with C5 union C5.
- Both have ten nodes; every node has degree two.
- All nodes start with identical features.
- One graph is connected; the other has two components.
- 1-WL keeps all nodes in one colour class at every round.

Key point: additional message-passing layers do not resolve this collision under the standard 1-WL-bounded message-passing framework.

## 6 — What I implemented
- Graph constructors with explicit undirected edge lists.
- Canonical 1-WL colour refinement.
- Two-layer GCN and GIN implementations in PyTorch.
- Graph-level sum/mean pooling and classification head.
- Stratified train/test split, fixed seeds, JSON result reporting.
- Unit tests for adjacency, WL collision, and model forward passes.

## 7 — Experimental protocol
- Five synthetic graph families; six sizes per family.
- Constant node features to isolate structure.
- Same split seeds and training budget for GCN and GIN.
- Three default seeds; accuracy and across-seed dispersion reported.

Caution: small pipeline sanity check, not a recognised benchmark or claim of superior performance.

## 8 — What results can and cannot say
**Can say:** whether the implementation runs, whether the synthetic classes are learnable under this protocol, and how variable results are across seeds.  
**Cannot say:** that one architecture is universally more expressive, that it solves graph isomorphism, or that it establishes a new theoretical result.

## 9 — Next research steps
1. Add recognised graph datasets and documented splits.
2. Add GraphSAGE and GAT baselines.
3. Build a systematic suite of 1-WL-indistinguishable graph pairs.
4. Study feature initialisation, pooling, and structural encodings.
5. Add confidence intervals, ablations, complexity analysis, and reproducibility metadata.

## 10 — Questions for the research group
- Which expressivity gap is most interesting for the group's current work?
- Would higher-order WL/GNNs or structural encodings be the more promising direction?
- Which theoretical foundations and benchmark protocols should I prioritise?

Closing: “My goal with this prototype was to create a transparent starting point and use it to ask a sharper mathematical question. I would value guidance on turning that question into a focused doctoral research problem.”
