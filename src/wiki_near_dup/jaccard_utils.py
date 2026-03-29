"""Set-based Jaccard helpers (no PySpark ML import — safe for Python 3.12+ unit tests)."""

from __future__ import annotations


def jaccard_similarity_indices(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def jaccard_distance_indices(a: set[int], b: set[int]) -> float:
    return 1.0 - jaccard_similarity_indices(a, b)


def brute_force_similar_pairs_sets(
    id_to_indices: dict[int, set[int]],
    max_distance: float,
) -> set[tuple[int, int]]:
    """All unordered pairs with Jaccard distance <= max_distance."""
    ids = sorted(id_to_indices.keys())
    out: set[tuple[int, int]] = set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ia, ib = ids[i], ids[j]
            if jaccard_distance_indices(id_to_indices[ia], id_to_indices[ib]) <= max_distance:
                out.add((ia, ib))
    return out


def precision_recall_f1(lsh_pairs: set[tuple[int, int]], ground_truth: set[tuple[int, int]]):
    if not lsh_pairs:
        return {"precision": None, "recall": None, "f1": None, "tp": 0, "fp": 0, "fn": len(ground_truth)}
    tp = len(lsh_pairs & ground_truth)
    fp = len(lsh_pairs - ground_truth)
    fn = len(ground_truth - lsh_pairs)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
