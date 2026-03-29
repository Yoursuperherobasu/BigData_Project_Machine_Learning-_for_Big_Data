"""Scalability timing helpers and sampled accuracy vs brute-force Jaccard (hashed features)."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from typing import Any

from pyspark.ml.linalg import DenseVector, SparseVector, Vector
from pyspark.sql import DataFrame, SparkSession

from wiki_near_dup.features import featurize
from wiki_near_dup.ingest import pages_to_articles, read_wikipedia_xml, sample_pages
from wiki_near_dup.jaccard_utils import brute_force_similar_pairs_sets, precision_recall_f1
from wiki_near_dup.lsh_pipeline import candidate_pairs, fit_lsh


def vector_to_indices(v: Vector) -> set[int]:
    if isinstance(v, SparseVector):
        return set(int(i) for i in v.indices)
    if isinstance(v, DenseVector):
        arr = v.toArray()
        return {int(i) for i, x in enumerate(arr) if x != 0.0}
    raise TypeError(f"Unsupported vector type: {type(v)}")


def brute_force_similar_pairs(
    rows: list[tuple[int, Vector]],
    max_distance: float,
) -> set[tuple[int, int]]:
    """All unordered pairs with Jaccard distance <= max_distance (on hashed binary features)."""
    id_to_indices = {pid: vector_to_indices(v) for pid, v in rows}
    return brute_force_similar_pairs_sets(id_to_indices, max_distance)


def pairs_from_lsh_dataframe(pairs_df: DataFrame) -> set[tuple[int, int]]:
    rows = pairs_df.select("page_id_a", "page_id_b").collect()
    return {(int(r.page_id_a), int(r.page_id_b)) for r in rows}


def run_lsh_subset_timed(
    spark: SparkSession,
    xml_path: str,
    *,
    sample_fraction: float,
    num_partitions: int | None,
    num_features: int,
    num_hash_tables: int,
    jaccard_distance_threshold: float,
    seed: int,
    collect_pairs: bool = False,
) -> tuple[int, int, dict[str, float], set[tuple[int, int]] | None]:
    """
    End-to-end on a sampled corpus.

    Returns ``(n_docs, n_candidate_pairs, timings_sec, lsh_pairs_or_none)``.
    Avoid ``collect_pairs=True`` on large outputs (driver OOM risk).
    """
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    raw = read_wikipedia_xml(spark, xml_path)
    articles = pages_to_articles(raw)
    articles = sample_pages(articles, sample_fraction, seed=seed)
    if num_partitions:
        articles = articles.repartition(num_partitions)
    articles = articles.cache()
    n = articles.count()
    timings["ingest_sample_sec"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    _, feat = featurize(articles, num_features=num_features)
    feat = feat.select("page_id", "title", "features").cache()
    feat.count()
    timings["featurize_sec"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    model = fit_lsh(feat, num_hash_tables=num_hash_tables)
    pairs = candidate_pairs(model, feat, jaccard_distance_threshold)
    pairs = pairs.cache()
    npairs = pairs.count()
    timings["lsh_join_sec"] = time.perf_counter() - t2
    timings["total_sec"] = time.perf_counter() - t0

    collected: set[tuple[int, int]] | None = None
    if collect_pairs:
        collected = pairs_from_lsh_dataframe(pairs)

    articles.unpersist()
    feat.unpersist()
    pairs.unpersist()
    return n, npairs, timings, collected


def evaluate_accuracy_sample(
    spark: SparkSession,
    xml_path: str,
    *,
    sample_fraction: float,
    max_docs_for_bruteforce: int,
    num_features: int,
    num_hash_tables: int,
    jaccard_distance_threshold: float,
    seed: int,
) -> dict[str, Any]:
    """
    If sampled doc count <= max_docs_for_bruteforce, compute brute-force ground truth
    on hashed binary features and compare to LSH candidates.
    """
    raw = read_wikipedia_xml(spark, xml_path)
    articles = pages_to_articles(raw)
    articles = sample_pages(articles, sample_fraction, seed=seed)
    articles = articles.cache()
    n = articles.count()
    if n > max_docs_for_bruteforce:
        articles.unpersist()
        return {
            "skipped": True,
            "reason": f"sample size {n} > max_docs_for_bruteforce {max_docs_for_bruteforce}",
            "n_docs": n,
        }

    _, feat = featurize(articles, num_features=num_features)
    feat = feat.select("page_id", "title", "features").cache()
    rows = [(int(r.page_id), r.features) for r in feat.collect()]
    gt = brute_force_similar_pairs(rows, jaccard_distance_threshold)

    model = fit_lsh(feat, num_hash_tables=num_hash_tables)
    pairs_df = candidate_pairs(
        model,
        feat.select("page_id", "title", "features"),
        jaccard_distance_threshold,
    )
    lsh = pairs_from_lsh_dataframe(pairs_df)

    metrics = precision_recall_f1(lsh, gt)
    articles.unpersist()
    feat.unpersist()
    return {
        "skipped": False,
        "n_docs": n,
        "n_ground_truth_pairs": len(gt),
        "n_lsh_pairs": len(lsh),
        **metrics,
    }


def scaling_study_json(
    spark: SparkSession,
    xml_path: str,
    fractions: Iterable[float],
    *,
    num_partitions: int | None,
    num_features: int,
    num_hash_tables: int,
    jaccard_distance_threshold: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Run several sample fractions and return one record per run (for plots / report)."""
    records: list[dict[str, Any]] = []
    for frac in fractions:
        n, npairs, timings, _ = run_lsh_subset_timed(
            spark,
            xml_path,
            sample_fraction=frac,
            num_partitions=num_partitions,
            num_features=num_features,
            num_hash_tables=num_hash_tables,
            jaccard_distance_threshold=jaccard_distance_threshold,
            seed=seed,
            collect_pairs=False,
        )
        rec = {
            "sample_fraction": frac,
            "n_docs": n,
            "n_candidate_pairs": npairs,
            **{f"time_{k}": v for k, v in timings.items()},
        }
        records.append(rec)
    return records


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
