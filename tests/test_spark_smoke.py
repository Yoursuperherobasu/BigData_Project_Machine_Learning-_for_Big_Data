"""
Spark + spark-xml integration smoke test (skipped unless RUN_SPARK_INTEGRATION=1).

Downloads Maven artifacts on first run; needs network once.
"""

import os
from pathlib import Path

import setuptools  # noqa: F401 — PySpark 3.5 needs distutils shim on Python 3.12+
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SPARK_INTEGRATION") != "1",
    reason="Set RUN_SPARK_INTEGRATION=1 to run Spark tests (downloads spark-xml).",
)


@pytest.fixture(scope="module")
def spark_session():
    from pyspark.sql import SparkSession

    from wiki_near_dup.config import SPARK_XML_PACKAGE_DEFAULT

    spark = (
        SparkSession.builder.appName("pytest_wiki_lsh")
        .master("local[2]")
        .config("spark.jars.packages", SPARK_XML_PACKAGE_DEFAULT)
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield spark
    spark.stop()


def test_read_tiny_wiki_and_lsh(spark_session):
    from wiki_near_dup.features import featurize
    from wiki_near_dup.ingest import pages_to_articles, read_wikipedia_xml
    from wiki_near_dup.lsh_pipeline import candidate_pairs, fit_lsh

    fixture = Path(__file__).resolve().parent / "fixtures" / "tiny_wiki.xml"
    path = fixture.as_uri()
    raw = read_wikipedia_xml(spark_session, path)
    articles = pages_to_articles(raw)
    n = articles.count()
    assert n == 2  # redirect dropped

    _, feat = featurize(articles, num_features=1 << 12)
    feat = feat.select("page_id", "title", "features").cache()
    model = fit_lsh(feat, num_hash_tables=3)
    pairs = candidate_pairs(model, feat, jaccard_distance_threshold=0.5)
    c = pairs.count()
    assert c >= 1
    feat.unpersist()
