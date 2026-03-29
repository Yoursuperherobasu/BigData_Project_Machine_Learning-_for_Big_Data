#!/usr/bin/env bash
# Example: submit the pipeline with spark-xml on the driver classpath.
# Adjust Scala tag (_2.12 vs _2.13) to match your Spark distribution.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

# Spark 3.4/3.5 pre-built often uses Scala 2.12; some builds use 2.13.
XML_PKG="${SPARK_XML_COORD:-com.databricks:spark-xml_2.12:0.18.0}"

spark-submit \
  --packages "${XML_PKG}" \
  --master "${SPARK_MASTER:-local[*]}" \
  "${ROOT}/scripts/run_pipeline.py" \
  "$@"
