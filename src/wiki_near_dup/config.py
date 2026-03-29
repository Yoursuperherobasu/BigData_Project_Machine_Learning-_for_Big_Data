"""Default hyperparameters and Spark package coordinates."""

# Match Scala binary version to your Spark install (see spark-submit --version).
SPARK_XML_PACKAGE_DEFAULT = "com.databricks:spark-xml_2.12:0.18.0"

DEFAULT_HASHING_NUM_FEATURES = 1 << 18
DEFAULT_MIN_TOKEN_LENGTH = 2
DEFAULT_NUM_HASH_TABLES = 5
# PySpark 3.5 MinHashLSH exposes only numHashTables (hash function count is JVM-internal).
DEFAULT_JACCARD_DISTANCE_THRESHOLD = 0.3  # similarity >= 0.7
