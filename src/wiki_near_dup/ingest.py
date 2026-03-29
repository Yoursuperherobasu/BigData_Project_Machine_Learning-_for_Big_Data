"""Load Wikipedia pages-articles XML via spark-xml."""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StructType


def read_wikipedia_xml(spark: SparkSession, path: str) -> DataFrame:
    """
    Read MediaWiki export XML (row tag ``page``).

    ``path`` may be ``file://``, ``hdfs://``, or a local path resolvable by Spark.
    Requires the Maven package ``com.databricks:spark-xml`` on the Spark classpath.
    """
    return (
        spark.read.format("com.databricks.spark.xml")
        .option("rowTag", "page")
        .option("mode", "PERMISSIVE")
        .option("charset", "UTF-8")
        .load(path)
    )


def _revision_text_column(df: DataFrame):
    if "revision" not in df.columns:
        raise ValueError(
            "Wikipedia XML dataframe missing 'revision'. Columns: " + ",".join(df.columns)
        )
    r = df.schema["revision"].dataType
    if isinstance(r, StructType):
        return F.col("revision.text").cast("string")
    if isinstance(r, ArrayType):
        return F.element_at(F.col("revision"), 1).getField("text").cast("string")
    raise ValueError(f"Unexpected Spark type for revision: {r}")


def pages_to_articles(
    raw: DataFrame,
    *,
    article_namespace: int = 0,
    drop_redirects: bool = True,
    min_text_len: int = 50,
) -> DataFrame:
    """
    Normalize to ``page_id``, ``title``, ``text`` for main-namespace articles.
    """
    text_col = _revision_text_column(raw)
    out = raw.select(
        F.col("id").cast("long").alias("page_id"),
        F.col("title").cast("string").alias("title"),
        F.col("ns").cast("int").alias("ns"),
        text_col.alias("text"),
    )
    out = out.filter(F.col("ns") == F.lit(article_namespace))
    out = out.filter(F.col("text").isNotNull())
    out = out.filter(F.length(F.trim(F.col("text"))) >= F.lit(min_text_len))
    if drop_redirects:
        t = F.upper(F.trim(F.col("text")))
        out = out.filter(~t.startswith("#REDIRECT"))
    return out.select("page_id", "title", "text")


def sample_pages(df: DataFrame, fraction: float, seed: int = 42) -> DataFrame:
    """Random sample without replacement (approximate when fraction < 1)."""
    if fraction >= 1.0:
        return df
    return df.sample(False, fraction, seed=seed)
