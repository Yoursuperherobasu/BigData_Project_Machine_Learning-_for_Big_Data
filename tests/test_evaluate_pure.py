"""Pure-Python tests (no PySpark)."""

from wiki_near_dup.jaccard_utils import (
    brute_force_similar_pairs_sets,
    jaccard_distance_indices,
    jaccard_similarity_indices,
    precision_recall_f1,
)


def test_jaccard_identical():
    s = {1, 2, 3}
    assert jaccard_similarity_indices(s, s) == 1.0
    assert jaccard_distance_indices(s, s) == 0.0


def test_jaccard_disjoint():
    assert jaccard_similarity_indices({1}, {2}) == 0.0
    assert jaccard_distance_indices({1}, {2}) == 1.0


def test_brute_force_finds_close_pair():
    sets = {1: {0, 2}, 2: {0, 2}}
    pairs = brute_force_similar_pairs_sets(sets, max_distance=0.2)
    assert (1, 2) in pairs


def test_precision_recall():
    gt = {(1, 2), (3, 4)}
    lsh = {(1, 2), (5, 6)}
    m = precision_recall_f1(lsh, gt)
    assert m["tp"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert abs(m["precision"] - 0.5) < 1e-6
    assert abs(m["recall"] - 0.5) < 1e-6
