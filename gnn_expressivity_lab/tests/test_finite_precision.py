import json
import subprocess
import sys

import pytest
import torch

from gnn_expressivity_lab.finite_precision import (
    build_encoder, calibrate, encode, pair_families, prune_, quantize, separated)
from gnn_expressivity_lab.graphs import benchmark_graphs, cycle_graph, disjoint_cycles, star_graph
from gnn_expressivity_lab.wl import wl_equivalent


def test_quantize_float_reference_is_identity():
    t = torch.tensor([-3.0, 0.1234, 7.0])
    assert torch.equal(quantize(t, None, 1.0), t)


def test_quantize_modes_separate_clipping_from_rounding():
    t = torch.tensor([-5.0, 0.26, 5.0])
    full = quantize(t, 3, 1.0, "full")             # grid step 1/3, range [-1, 1]
    assert torch.allclose(full, torch.tensor([-1.0, 1 / 3, 1.0]))
    sat = quantize(t, 3, 1.0, "saturation_only")
    assert torch.allclose(sat, torch.tensor([-1.0, 0.26, 1.0]))
    rnd = quantize(t, 3, 1.0, "rounding_only")
    assert torch.allclose(rnd, torch.tensor([-5.0, 1 / 3, 5.0]))


def test_quantize_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        quantize(torch.ones(2), 1, 1.0)
    with pytest.raises(ValueError):
        quantize(torch.ones(2), 4, 1.0, "nope")


def test_every_pair_is_one_wl_distinguishable_within_two_rounds():
    families = pair_families(max_star_leaves=8, max_path=8, random_pairs=10)
    for pairs in families.values():
        assert pairs
        for a, b in pairs:
            assert not wl_equivalent(a, b, rounds=2)


def test_wl_equivalent_control_is_never_separated():
    model = build_encoder(11, 16)
    clips = calibrate(model, benchmark_graphs("constant"))
    pair = [cycle_graph(10), disjoint_cycles([5, 5])]
    for bits in (None, 8, 4, 2):
        emb = encode(model, pair, bits, clips, "full")
        assert not separated(emb[0], emb[1])


def test_float_gin_separates_small_stars():
    model = build_encoder(11, 16)
    emb = encode(model, [star_graph(5), star_graph(6)])
    assert separated(emb[0], emb[1])


def test_saturation_collapses_stars_beyond_the_clip_range():
    model = build_encoder(11, 16)
    clips = {k: 1e-3 for k in calibrate(model, benchmark_graphs("constant"))}
    emb = encode(model, [star_graph(40), star_graph(41)], 8, clips, "saturation_only")
    assert not separated(emb[0], emb[1])


def test_zero_pruning_changes_nothing_and_pruning_zeroes_weights():
    a, b = build_encoder(11, 16), build_encoder(11, 16)
    prune_(b, 0.0)
    g = [star_graph(7)]
    assert torch.equal(encode(a, g), encode(b, g))
    prune_(b, 0.9)
    w = b.layers[0].mlp[2].weight
    assert (w == 0).float().mean() >= 0.85


def test_cli_writes_json(tmp_path):
    out = tmp_path / "fp.json"
    subprocess.run([sys.executable, "-m", "gnn_expressivity_lab.finite_precision", "--seeds", "11",
                    "--bits", "4", "--prune", "0.5", "--hidden-dim", "8", "--max-star-leaves", "6",
                    "--random-pairs", "5", "--output", str(out)], check=True, capture_output=True)
    data = json.loads(out.read_text())
    assert data["control_C10_vs_C5C5_separated_count"] == 0
    assert "4-bit" in data["quantization"]["full"]
