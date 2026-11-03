# Expressivity Under Finite Precision: a Pilot

## Question

Expressivity results for message-passing GNNs, including the 1-WL upper bound and the
injectivity of sum aggregation in GIN, assume exact real arithmetic. Deployed GNNs run at
limited precision. **When a GIN encoder is quantized or pruned, how many graph pairs that
1-WL distinguishes does it still tell apart, and which mechanism removes the rest?**

## Protocol

- **Model:** two-layer GIN, hidden width 32, sum aggregation, sum readout, constant node
  features (the pure-structure 1-WL setting). Randomly initialised, not trained: this probes
  what the architecture can still represent at a given precision, not what training finds.
- **Pairs (198):** every pair is verified to be 1-WL-distinguishable within two rounds,
  matching the model's two layers, so failures cannot be blamed on depth.
  - 60 stars, k leaves vs k + 1 (degree grows: tests out-of-range magnitudes)
  - 38 paths, n vs n + 1 nodes (difference is one interior node: tests small differences)
  - 100 random graph pairs with equal node and edge counts (only local structure differs)
- **Quantization:** symmetric uniform fake quantization at 8, 6, 4, 3 and 2 bits. Weights are
  quantized per tensor at their own range. Activations are quantized at seven sites (each
  layer's aggregate, MLP hidden, layer output, and the readout), with clip ranges calibrated as
  max |activation| on the lab's benchmark graphs (7 to 12 nodes, so maximum degree 11).
- **Mechanism ablation:** `full` (clip and round), `saturation_only` (clip, no rounding),
  `rounding_only` (round to the same grid, no clipping).
- **Pruning arm:** per-matrix magnitude pruning of float32 weights at 50, 80, 90 and 95%.
- **Control:** C10 vs C5 + C5 (1-WL-equivalent). It must never be separated, and it was not
  separated in any configuration (0 of 80 runs).
- **Seeds:** 11, 23, 37, 41, 53. Python 3.13.7, PyTorch 2.9.1 (CPU), Windows. The whole run
  takes about 5 seconds.

## Results (mean separation rate over five seeds)

**Clip and round (`full`)**

| precision | stars | paths | random same-size | stars, degree <= 11 | stars, degree > 11 |
|---|---|---|---|---|---|
| float32 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 8-bit | 99.7% | 100.0% | 97.6% | 98.0% | 100.0% |
| 6-bit | 85.0% | 50.5% | 94.4% | 88.0% | 84.4% |
| 4-bit | 24.0% | 4.7% | 71.0% | 14.0% | 26.0% |
| 3-bit | 0.0% | 0.0% | 17.0% | 0.0% | 0.0% |
| 2-bit | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

**Round only (no clipping)**

| precision | stars | paths | random same-size | stars, degree <= 11 | stars, degree > 11 |
|---|---|---|---|---|---|
| 8-bit | 99.7% | 100.0% | 97.6% | 98.0% | 100.0% |
| 6-bit | 97.3% | 50.5% | 94.4% | 88.0% | 99.2% |
| 4-bit | 59.0% | 4.7% | 71.0% | 14.0% | 68.0% |
| 3-bit | 26.7% | 0.0% | 17.0% | 0.0% | 32.0% |
| 2-bit | 3.0% | 0.0% | 0.0% | 0.0% | 3.6% |

**Clip only (no rounding):** 100.0% of pairs separated in every family at every bit width.

**Magnitude pruning (float32):** 100.0% at 50, 80 and 90%; at 95%, 100% of stars and paths
and 98.4% of random same-size pairs.

## What the pilot shows

1. **Separation degrades sharply between 8 and 4 bits.** At 8 bits almost every
   WL-distinguishable pair survives; at 4 bits only 24% of star pairs and 4.7% of path pairs
   do; at 3 bits and below essentially nothing structural survives except some random pairs.
2. **Rounding, not saturation, is the primary failure.** Clipping alone never merged a pair.
   Rounding alone reproduces most of the loss, and it hits small differences first: the path
   pairs (one extra interior node) collapse to 50.5% already at 6 bits, and small stars
   (14.0% at 4 bits) fail far more than large ones (68.0%).
3. **Saturation adds damage only outside the calibrated range, and only together with
   rounding.** For stars with degree at most 11 (inside calibration), clip-and-round and
   round-only give the same 14.0% at 4 bits. For stars with degree above 11, adding clipping
   drops separation from 68.0% to 26.0%. A plausible reading, consistent with these numbers
   but not yet tested channel by channel: clipping removes the large-magnitude channels that
   rounding spares, rounding removes the small-magnitude channels that clipping spares, and
   only together do they remove both.
4. **Magnitude pruning is far gentler than quantization** on this probe: separation survives
   up to 90% sparsity, which is consistent with random high-dimensional features staying
   injective on small graph sets.
5. **The theoretical bound and the deployed model can diverge.** Exact-arithmetic GIN attains
   1-WL power on every pair here; the same architecture at 4 bits separates fewer than one star
   pair in four. Precision, not only architecture, decides practical expressivity.

## Limitations

- Untrained encoders with random weights. Trained models learn activation ranges and may
  behave differently; quantization-aware training is not studied.
- One quantizer (symmetric, uniform, per-tensor, max calibration). Per-channel scales or
  percentile calibration would likely change the saturation results.
- Small synthetic pair families and one architecture (GIN, 2 layers, width 32).
- "Separated" means embeddings differ by more than float32 noise; a downstream classifier
  could still fail to use a tiny difference.
- Fake quantization in float32 simulates the arithmetic; it is not an integer kernel.
- Seed-to-seed spread is large near the threshold: at 4 bits (clip and round), star
  separation ranged from 2% to 38% across the five seeds and path separation from 0% to 16%.
  The means above are indicative; per-seed rates are in the JSON. More seeds are needed
  before quoting any single 4-bit number.

## Next steps

1. Channel-level test of finding 3: which embedding dimensions carry the separating
   difference in float32, and which of them each mechanism removes.
2. Per-channel quantization and percentile calibration as alternative quantizers.
3. Train GIN on MUTAG and PROTEINS, apply post-training and quantization-aware training, and
   relate accuracy loss to the separation rate measured here.
4. Relate the failing pairs to WL colour-refinement rounds: do pairs separated only in the
   second round fail at higher bit widths than pairs separated in the first?
5. A quantized lottery-ticket probe: whether pruned-then-quantized subnetworks retain 1-WL
   separation better than quantized dense networks.

## Reproduce

From the gnn_expressivity_lab directory:

    gnn-finite-precision --output results/finite_precision.json

The JSON includes per-seed rates, the control count and the environment.
