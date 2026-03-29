#!/usr/bin/env python3
"""
Near-duplicate detection: Wikipedia XML -> MinHash LSH candidate pairs.

Prefer running via scripts/spark_submit.sh so spark-xml is on the classpath.
You can also set PYSPARK_SUBMIT_ARGS or spark.jars.packages (see --spark-xml-package).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Python 3.12+: stdlib distutils removed; PySpark 3.5 still does `from distutils.version import ...`.
import setuptools  # noqa: F401

# Repo root: .../scripts/run_pipeline.py -> parent is scripts, parent.parent is root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from pyspark.sql import SparkSession

from wiki_near_dup.config import (
    DEFAULT_HASHING_NUM_FEATURES,
    DEFAULT_JACCARD_DISTANCE_THRESHOLD,
    DEFAULT_NUM_HASH_TABLES,
    SPARK_XML_PACKAGE_DEFAULT,
)
from wiki_near_dup.evaluate import (
    evaluate_accuracy_sample,
    scaling_study_json,
    write_json,
)
from wiki_near_dup.features import featurize
from wiki_near_dup.ingest import pages_to_articles, read_wikipedia_xml, sample_pages
from wiki_near_dup.lsh_pipeline import candidate_pairs, fit_lsh


def build_spark(app_name: str, spark_xml_package: str | None) -> SparkSession:
    # Spark 4 PySpark may start Spark Connect if these are set; this pipeline uses classic JVM SQL.
    for _k in ("SPARK_CONNECT_MODE_ENABLED", "SPARK_LOCAL_REMOTE", "SPARK_REMOTE"):
        os.environ.pop(_k, None)
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.api.mode", "classic")
        .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "200"))
    )
    pkg = spark_xml_package or os.environ.get("SPARK_XML_PACKAGE")
    if pkg:
        builder = builder.config("spark.jars.packages", pkg)
    return builder.getOrCreate()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wikipedia near-duplicates with MinHash LSH")
    p.add_argument("--input", required=True, help="Path to pages-articles .xml or .xml.bz2 (Spark-readable URI)")
    p.add_argument(
        "--output",
        default="",
        help="Directory to write candidate pairs as Parquet (required unless --eval-scaling or --eval-accuracy only)",
    )
    p.add_argument("--sample-fraction", type=float, default=1.0, help="Fraction of articles to use (0,1]")
    p.add_argument("--num-partitions", type=int, default=0, help="repartition articles (0 = no extra repartition)")
    p.add_argument("--num-features", type=int, default=DEFAULT_HASHING_NUM_FEATURES)
    p.add_argument("--num-hash-tables", type=int, default=DEFAULT_NUM_HASH_TABLES)
    p.add_argument(
        "--jaccard-distance-threshold",
        type=float,
        default=DEFAULT_JACCARD_DISTANCE_THRESHOLD,
        help="Max Jaccard distance (1 - similarity) for candidates",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--spark-xml-package",
        default=SPARK_XML_PACKAGE_DEFAULT,
        help="Maven coordinate for databricks spark-xml (Scala version must match Spark)",
    )
    p.add_argument(
        "--eval-scaling",
        action="store_true",
        help="Run scaling study over --scaling-fractions; writes --metrics-json or stdout",
    )
    p.add_argument(
        "--scaling-fractions",
        default="0.01,0.02,0.05,0.1",
        help="Comma-separated sample fractions for scaling study",
    )
    p.add_argument(
        "--eval-accuracy",
        action="store_true",
        help="Brute-force vs LSH on small sample (needs small --sample-fraction)",
    )
    p.add_argument(
        "--max-docs-bruteforce",
        type=int,
        default=400,
        help="Skip accuracy eval if sampled doc count exceeds this (all-pairs cost)",
    )
    p.add_argument(
        "--metrics-json",
        default="",
        help="Write evaluation/scaling metrics JSON to this path",
    )
    return p.parse_args()


def run_main_pipeline(spark: SparkSession, args: argparse.Namespace) -> None:
    if not args.output:
        raise SystemExit("--output is required for the main pipeline (or use --eval-scaling / --eval-accuracy only)")
    raw = read_wikipedia_xml(spark, args.input)
    articles = pages_to_articles(raw)
    articles = sample_pages(articles, args.sample_fraction, seed=args.seed)
    if args.num_partitions and args.num_partitions > 0:
        articles = articles.repartition(args.num_partitions)

    _, feat = featurize(articles, num_features=args.num_features)
    feat = feat.select("page_id", "title", "features").cache()
    feat.count()

    model = fit_lsh(feat, num_hash_tables=args.num_hash_tables)
    pairs = candidate_pairs(model, feat, args.jaccard_distance_threshold)
    pairs.write.mode("overwrite").parquet(args.output)
    feat.unpersist()


def main() -> None:
    args = parse_args()
    spark = build_spark("WikiNearDupLSH", args.spark_xml_package)

    try:
        if args.eval_scaling:
            fracs = [float(x.strip()) for x in args.scaling_fractions.split(",") if x.strip()]
            num_part = args.num_partitions if args.num_partitions > 0 else None
            records = scaling_study_json(
                spark,
                args.input,
                fracs,
                num_partitions=num_part,
                num_features=args.num_features,
                num_hash_tables=args.num_hash_tables,
                jaccard_distance_threshold=args.jaccard_distance_threshold,
                seed=args.seed,
            )
            out_obj = {"scaling_runs": records, "input": args.input}
            if args.metrics_json:
                write_json(args.metrics_json, out_obj)
            else:
                print(json.dumps(out_obj, indent=2))
            return

        if args.eval_accuracy:
            acc = evaluate_accuracy_sample(
                spark,
                args.input,
                sample_fraction=args.sample_fraction,
                max_docs_for_bruteforce=args.max_docs_bruteforce,
                num_features=args.num_features,
                num_hash_tables=args.num_hash_tables,
                jaccard_distance_threshold=args.jaccard_distance_threshold,
                seed=args.seed,
            )
            if args.metrics_json:
                write_json(args.metrics_json, acc)
            else:
                print(json.dumps(acc, indent=2))
            return

        run_main_pipeline(spark, args)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
