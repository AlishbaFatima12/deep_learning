"""Tests for stratified folds and uncertainty summaries."""
from collections import Counter
import pytest
from gnn_expressivity_lab.cross_validation import (
    stratified_kfold_indices, bootstrap_confidence_interval
)

def test_stratified_folds_cover_each_index_once():
    labels = [0] * 10 + [1] * 15
    splits = stratified_kfold_indices(labels, folds=5, seed=17)
    tests = [index for _, test in splits for index in test]
    assert sorted(tests) == list(range(len(labels)))
    for _, test in splits:
        counts = Counter(labels[index] for index in test)
        assert counts[0] == 2
        assert counts[1] == 3

def test_fold_train_and_test_are_disjoint():
    labels = [0] * 8 + [1] * 8
    for train, test in stratified_kfold_indices(labels, 4, 3):
        assert set(train).isdisjoint(test)
        assert set(train) | set(test) == set(range(len(labels)))

def test_kfold_rejects_rare_classes():
    with pytest.raises(ValueError):
        stratified_kfold_indices([0, 0, 0, 1, 1], folds=3, seed=1)

def test_bootstrap_interval_contains_constant_score():
    assert bootstrap_confidence_interval([0.75] * 8) == [0.75, 0.75]
