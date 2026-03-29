"""MinHash LSH: fit model, self-join for candidate near-duplicate pairs."""

from __future__ import annotations

from pyspark.ml.feature import MinHashLSH, MinHashLSHModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from wiki_near_dup.config import DEFAULT_NUM_HASH_TABLES


def fit_lsh(
    featurized: DataFrame,
    *,
    num_hash_tables: int = DEFAULT_NUM_HASH_TABLES,
) -> MinHashLSHModel:
    mh = MinHashLSH(
        inputCol="features",
        outputCol="hashes",
        numHashTables=num_hash_tables,
    )
    return mh.fit(featurized)


def candidate_pairs(
    model: MinHashLSHModel,
    featurized: DataFrame,
    jaccard_distance_threshold: float,
    *,
    id_col: str = "page_id",
) -> DataFrame:
    """
    Self-join on Jaccard distance (Spark: distance = 1 - Jaccard similarity).

    Returns columns including ``page_id_a``, ``title_a``, ``page_id_b``, ``title_b``, ``jaccardDist``.
    """
    cols = [id_col, "title", "features"]
    missing = [c for c in cols if c not in featurized.columns]
    if missing:
        raise ValueError(f"featurized DataFrame missing columns: {missing}")

    left = featurized.select(
        F.col(id_col).alias("page_id_a"),
        F.col("title").alias("title_a"),
        F.col("features"),
    )
    right = featurized.select(
        F.col(id_col).alias("page_id_b"),
        F.col("title").alias("title_b"),
        F.col("features").alias("features_b"),
    ).withColumnRenamed("features_b", "features")

    joined = model.approxSimilarityJoin(
        left,
        right,
        jaccard_distance_threshold,
        distCol="jaccardDist",
    )

    joined = joined.select(
        F.col("datasetA.page_id_a").alias("page_id_a"),
        F.col("datasetA.title_a").alias("title_a"),
        F.col("datasetB.page_id_b").alias("page_id_b"),
        F.col("datasetB.title_b").alias("title_b"),
        F.col("jaccardDist"),
    )
    joined = joined.filter(F.col("page_id_a") < F.col("page_id_b"))
    return joined
