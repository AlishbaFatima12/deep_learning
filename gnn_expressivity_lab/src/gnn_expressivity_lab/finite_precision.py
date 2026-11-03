"""Expressivity under finite precision: does a quantized or pruned GIN still
separate graphs that 1-WL separates?

Expressivity results for message-passing GNNs (the 1-WL bound) assume exact
real arithmetic. Deployed models run at limited precision. This module
measures, on graph pairs that 1-WL provably distinguishes, how often a GIN
encoder still maps the two graphs to different embeddings when its weights
and activations are fake-quantized to b bits, and it separates the two ways
precision can destroy information:

* saturation: activations are clipped to a calibrated range, so sums that
  grow with node degree stop growing once they reach the clip value;
* rounding: activations are snapped to a uniform grid, so embeddings closer
  than one grid step coincide.

A magnitude-pruning arm applies the same measurement to sparsified weights.

Scope: randomly initialised (untrained) GIN encoders with sum aggregation and
sum readout, constant node features, post-training uniform symmetric
quantization calibrated on small graphs. This is a controlled probe of a
mechanism, not a statement about trained models or any hardware format.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .graphs import (Graph, benchmark_graphs, cycle_graph, disjoint_cycles,
                     make_graph, path_graph, star_graph)
from .sparse_models import SparseGraphClassifier, batch_graphs
from .wl import wl_equivalent

MODES = ("full", "saturation_only", "rounding_only")


def quantize(t: torch.Tensor, bits: Optional[int], clip: float, mode: str = "full") -> torch.Tensor:
    """Symmetric uniform fake quantization of an activation tensor.

    ``bits=None`` returns the tensor unchanged (float32 reference).
    ``full`` clips to [-clip, clip] then rounds to the b-bit grid;
    ``saturation_only`` clips without rounding; ``rounding_only`` rounds to
    the same grid without clipping.
    """
    if bits is None:
        return t
    if bits < 2:
        raise ValueError("bits must be at least 2")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    clip = max(float(clip), 1e-12)
    step = clip / (2 ** (bits - 1) - 1)
    if mode in ("full", "saturation_only"):
        t = t.clamp(-clip, clip)
    if mode in ("full", "rounding_only"):
        t = torch.round(t / step) * step
    return t


def quantize_weight(w: torch.Tensor, bits: Optional[int], mode: str) -> torch.Tensor:
    """Per-tensor symmetric weight quantization at the tensor's own range
    (weights never saturate). Saturation-only leaves weights exact."""
    if bits is None or mode == "saturation_only":
        return w
    return quantize(w, bits, float(w.abs().max()), "rounding_only")


def build_encoder(seed: int, hidden_dim: int = 32) -> SparseGraphClassifier:
    """A two-layer GIN with sum readout, deterministically initialised."""
    torch.manual_seed(seed)
    random.seed(seed)
    model = SparseGraphClassifier(1, hidden_dim, 2, architecture="gin", pooling="sum", dropout=0.0)
    with torch.no_grad():
        # A random non-zero epsilon keeps (1 + eps) * x away from the
        # degenerate initial value of exactly one.
        for layer in model.layers:
            layer.eps.uniform_(-0.5, 0.5)
    return model.eval()


def prune_(model: SparseGraphClassifier, fraction: float) -> None:
    """In-place magnitude pruning of every GIN weight matrix."""
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must be in [0, 1)")
    if fraction == 0.0:
        return
    with torch.no_grad():
        for layer in model.layers:
            for module in layer.mlp:
                if isinstance(module, torch.nn.Linear):
                    w = module.weight
                    k = int(fraction * w.numel())
                    if k:
                        threshold = w.abs().flatten().kthvalue(k).values
                        w[w.abs() <= threshold] = 0.0


def _linear(x, module, bits, mode):
    return F.linear(x, quantize_weight(module.weight, bits, mode), module.bias)


@torch.no_grad()
def encode(model: SparseGraphClassifier, graphs: Sequence[Graph], bits: Optional[int] = None,
           clips: Optional[Dict[str, float]] = None, mode: str = "full",
           record: Optional[Dict[str, float]] = None) -> torch.Tensor:
    """Graph embeddings with every activation site quantized.

    Sites: each layer's aggregate (1 + eps) * x + sum of neighbours (where
    degree enters), the MLP hidden activation, the layer output, and the sum
    readout. ``record`` collects max |activation| per site for calibration.
    """
    batch = batch_graphs(list(graphs))
    clips = clips or {}

    def site(name, t):
        if record is not None:
            record[name] = max(record.get(name, 0.0), float(t.abs().max()))
        return quantize(t, bits, clips.get(name, 1.0), mode) if bits is not None else t

    h = batch.x
    src, dst = batch.edge_index
    for i, layer in enumerate(model.layers):
        neighbours = h.new_zeros(h.shape)
        if src.numel():
            neighbours.index_add_(0, dst, h[src])
        agg = site(f"l{i}_aggregate", (1.0 + layer.eps) * h + neighbours)
        hidden = site(f"l{i}_mlp_hidden", F.relu(_linear(agg, layer.mlp[0], bits, mode)))
        h = site(f"l{i}_output", F.relu(_linear(hidden, layer.mlp[2], bits, mode)))
    pooled = h.new_zeros((batch.num_graphs, h.shape[-1]))
    pooled.index_add_(0, batch.batch, h)
    return site("readout", pooled)


def calibrate(model: SparseGraphClassifier, calibration: Sequence[Graph]) -> Dict[str, float]:
    """Per-site clip range = max |activation| on the calibration graphs."""
    record: Dict[str, float] = {}
    encode(model, calibration, record=record)
    return record


def separated(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Two embeddings differ beyond float32 noise."""
    scale = max(1.0, float(a.abs().max()), float(b.abs().max()))
    return float((a - b).abs().max()) > 1e-5 * scale


def _random_pair(rng: random.Random, n: int, m: int) -> Tuple[Graph, Graph]:
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    a = make_graph(n, rng.sample(all_edges, m), name=f"random-{n}-{m}-a")
    b = make_graph(n, rng.sample(all_edges, m), name=f"random-{n}-{m}-b")
    return a, b


def pair_families(max_star_leaves: int = 60, max_path: int = 40, random_pairs: int = 100,
                  seed: int = 7) -> Dict[str, List[Tuple[Graph, Graph]]]:
    """Graph pairs that 1-WL distinguishes, grouped by what separates them."""
    families = {
        # degree grows with k: the saturation stress test
        "stars_k_vs_k+1": [(star_graph(k + 1), star_graph(k + 2)) for k in range(1, max_star_leaves + 1)],
        # size grows, degree stays <= 2: separated only through the readout sum
        "paths_n_vs_n+1": [(path_graph(n), path_graph(n + 1)) for n in range(3, max_path + 1)],
    }
    rng = random.Random(seed)
    same_size: List[Tuple[Graph, Graph]] = []
    tries = 0
    while len(same_size) < random_pairs and tries < 50 * random_pairs:
        tries += 1
        n = rng.randint(8, 14)
        m = rng.randint(n, 2 * n)
        a, b = _random_pair(rng, n, m)
        # Two GIN layers can match at most two 1-WL rounds, so pairs must be
        # separated within two rounds or the depth, not precision, is tested.
        if not wl_equivalent(a, b, rounds=2):
            same_size.append((a, b))
    # same node and edge count: separated only by local structure (rounding stress test)
    families["random_same_size"] = same_size
    for name, pairs in families.items():
        for a, b in pairs:
            if wl_equivalent(a, b, rounds=2):
                raise AssertionError(f"{name}: {a.name} and {b.name} are 1-WL equivalent")
    return families


def separation_rate(model, families, bits, clips, mode) -> Dict[str, dict]:
    out = {}
    for name, pairs in families.items():
        left = encode(model, [a for a, _ in pairs], bits, clips, mode)
        right = encode(model, [b for _, b in pairs], bits, clips, mode)
        flags = [separated(left[i], right[i]) for i in range(len(pairs))]
        out[name] = {"separated": sum(flags), "pairs": len(flags), "rate": sum(flags) / len(flags), "flags": flags}
    return out


# The calibration graphs have at most 12 nodes, so the largest degree the clip
# ranges ever saw is 11. Stars with more leaves test out-of-range degrees.
CALIBRATION_MAX_DEGREE = 11


def _star_bands(flags: List[bool]) -> Tuple[float, float]:
    """Separation rate for stars inside vs beyond the calibrated degree range
    (flags[i] is the pair with k = i + 1 leaves vs k + 1)."""
    inside = flags[:CALIBRATION_MAX_DEGREE - 1]
    beyond = flags[CALIBRATION_MAX_DEGREE - 1:]
    return sum(inside) / len(inside), (sum(beyond) / len(beyond) if beyond else float("nan"))


def run(seeds: Sequence[int], bit_widths: Sequence[Optional[int]], prune_fractions: Sequence[float],
        hidden_dim: int = 32, max_star_leaves: int = 60, max_path: int = 40, random_pairs: int = 100) -> dict:
    families = pair_families(max_star_leaves, max_path, random_pairs)
    calibration = benchmark_graphs("constant")  # graphs of 7 to 12 nodes: max degree 11
    control = (cycle_graph(10), disjoint_cycles([5, 5]))
    started = time.perf_counter()
    quant: Dict[str, Dict[str, Dict[str, list]]] = {}
    control_hits = 0
    for mode in MODES:
        for bits in bit_widths:
            key = "float32" if bits is None else f"{bits}-bit"
            if bits is None and mode != "full":
                continue
            for seed in seeds:
                model = build_encoder(seed, hidden_dim)
                clips = calibrate(model, calibration)
                res = separation_rate(model, families, bits, clips, mode)
                ctrl = encode(model, list(control), bits, clips, mode)
                control_hits += int(separated(ctrl[0], ctrl[1]))
                cell = quant.setdefault(mode, {}).setdefault(key, {})
                for fam, r in res.items():
                    cell.setdefault(fam, []).append(r["rate"])
                inside, beyond = _star_bands(res["stars_k_vs_k+1"]["flags"])
                cell.setdefault("stars_inside_calibrated_degree", []).append(inside)
                cell.setdefault("stars_beyond_calibrated_degree", []).append(beyond)
    pruning: Dict[str, Dict[str, list]] = {}
    for fraction in prune_fractions:
        for seed in seeds:
            model = build_encoder(seed, hidden_dim)
            prune_(model, fraction)
            res = separation_rate(model, families, None, None, "full")
            cell = pruning.setdefault(f"{int(round(fraction * 100))}%", {})
            for fam, r in res.items():
                cell.setdefault(fam, []).append(r["rate"])
    return {
        "question": "Do quantized or pruned GIN encoders still separate 1-WL-distinguishable graphs?",
        "setup": {
            "model": f"2-layer GIN, hidden {hidden_dim}, sum aggregation, sum readout, random init",
            "features": "constant node features (pure structure, the 1-WL setting)",
            "quantization": "symmetric uniform fake quantization; weights per-tensor at their own range; "
                            "activations at 7 sites clipped at max |activation| on calibration graphs",
            "calibration_graphs": "benchmark_graphs(): paths, stars, wheels, complete, complete bipartite, 7-12 nodes",
            "seeds": list(seeds),
            "pair_counts": {k: len(v) for k, v in families.items()},
        },
        "control_C10_vs_C5C5_separated_count": control_hits,
        "quantization": quant,
        "pruning": pruning,
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "platform": sys.platform, "seconds": round(time.perf_counter() - started, 2)},
    }


def _mean(xs):
    return sum(xs) / len(xs)


def markdown_tables(result: dict) -> str:
    fams = list(result["setup"]["pair_counts"])
    lines = []
    for mode, by_bits in result["quantization"].items():
        lines.append(f"\n**{mode}** (mean separation rate over seeds)\n")
        lines.append("| precision | " + " | ".join(fams) + " | stars, degree <= 11 | stars, degree > 11 |")
        lines.append("|---" * (len(fams) + 3) + "|")
        for key, cell in by_bits.items():
            row = [f"{_mean(cell[f]) * 100:.1f}%" for f in fams]
            row += [f"{_mean(cell[f]) * 100:.1f}%" for f in
                    ("stars_inside_calibrated_degree", "stars_beyond_calibrated_degree")]
            lines.append(f"| {key} | " + " | ".join(row) + " |")
    lines.append("\n**pruning** (float32 weights, magnitude-pruned)\n")
    lines.append("| pruned | " + " | ".join(fams) + " |")
    lines.append("|---" * (len(fams) + 1) + "|")
    for key, cell in result["pruning"].items():
        lines.append(f"| {key} | " + " | ".join(f"{_mean(cell[f]) * 100:.1f}%" for f in fams) + " |")
    return "\n".join(lines)


def main():
    """Run the finite-precision expressivity probe and write its JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 41, 53])
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 6, 4, 3, 2])
    parser.add_argument("--prune", type=float, nargs="+", default=[0.0, 0.5, 0.8, 0.9, 0.95])
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--max-star-leaves", type=int, default=60)
    parser.add_argument("--random-pairs", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results/finite_precision.json"))
    args = parser.parse_args()
    if any(b < 2 for b in args.bits):
        parser.error("bit widths must be at least 2")
    result = run(args.seeds, [None] + list(args.bits), args.prune, args.hidden_dim,
                 args.max_star_leaves, random_pairs=args.random_pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(markdown_tables(result))


if __name__ == "__main__":
    main()
