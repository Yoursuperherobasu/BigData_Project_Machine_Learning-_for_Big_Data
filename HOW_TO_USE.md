# How to use this codebase (after cloning from GitHub)

This guide explains how to set up the environment, place Wikipedia data, and run the near-duplicate detection pipeline. The same content is in [HOW_TO_USE.txt](HOW_TO_USE.txt) for plain-text readers. See also [README.md](README.md) and [Spark_LSH_Wikipedia_plan.md](Spark_LSH_Wikipedia_plan.md).

---

## 1. Clone the repository

```bash
git clone <your-repo-url>
cd BigData_Project_Machine_Learning-_for_Big_Data
```

Use the default branch your team agrees on (for example: `main`).

---

## 2. Prerequisites

- **Java** 11 or 17 (match your Spark distribution).
- **Apache Spark** on the cluster, or local Spark that **matches** the `pyspark` version installed by `pip` (mismatched `SPARK_HOME` vs PySpark often causes JVM errors).
- **Python 3.10–3.12** recommended (PySpark + Python 3.13 can be problematic).
- **Hadoop HDFS** client tools if you copy data with `hdfs dfs`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .             # optional: editable install of wiki_near_dup
```

First run downloads the **spark-xml** JAR via Maven (needs network once).

---

## 3. Obtain the Wikipedia data (not in Git)

- Download from [https://dumps.wikimedia.org](https://dumps.wikimedia.org) — `enwiki-*-pages-articles*.xml.bz2` or a **multistream** shard of the same schema.
- Keep `.bz2` files out of Git (see `.gitignore`).
- HDFS upload example:

```bash
hdfs dfs -mkdir -p /user/<you>/wikipedia
hdfs dfs -put /local/path/to/your-file.xml.bz2 /user/<you>/wikipedia/
```

---

## 4. Run the main pipeline (Parquet output)

Prefer **`scripts/spark_submit.sh`** so `com.databricks:spark-xml` is on the classpath.

**Local file** (use an absolute `file://` path):

```bash
chmod +x scripts/spark_submit.sh
export SPARK_MASTER=local[*]

./scripts/spark_submit.sh \
  --input file:///absolute/path/to/enwiki-....xml.bz2 \
  --output file:///absolute/path/to/output_pairs \
  --sample-fraction 0.05 \
  --num-partitions 64
```

**HDFS:**

```bash
./scripts/spark_submit.sh \
  --input hdfs://namenode:8020/user/you/wikipedia/enwiki-....xml.bz2 \
  --output hdfs://namenode:8020/user/you/out/near_dup_pairs
```

Output: Parquet with `page_id_a`, `title_a`, `page_id_b`, `title_b`, `jaccardDist`.

### Useful flags

- `--sample-fraction` — random fraction of articles (after XML parse).
- `--num-hash-tables` — LSH parameter (accuracy vs speed; PySpark 3.5 `MinHashLSH` has no separate hash-function count in Python).
- `--jaccard-distance-threshold` — max Jaccard distance (1 − similarity) for pairs.
- `--spark-xml-package` — override Maven coordinate if your Spark uses Scala 2.13.

---

## 5. Scalability experiment (JSON metrics)

```bash
./scripts/spark_submit.sh \
  --input file:///absolute/path/to/dump.xml.bz2 \
  --eval-scaling \
  --scaling-fractions 0.01,0.02,0.05,0.1 \
  --metrics-json ./metrics_scaling.json
```

Use the JSON fields `n_docs`, `n_candidate_pairs`, and `time_*` for report plots.

---

## 6. Accuracy check (small sample only)

Brute-force **all pairs** on the sample (hashed binary features) vs LSH candidates. Keep `--sample-fraction` tiny so document count stays below `--max-docs-bruteforce` (default 400).

```bash
./scripts/spark_submit.sh \
  --input file:///absolute/path/to/dump.xml.bz2 \
  --eval-accuracy \
  --sample-fraction 0.0001 \
  --max-docs-bruteforce 400 \
  --metrics-json ./metrics_accuracy.json
```

---

## 7. Tests

```bash
pytest tests/test_evaluate_pure.py -q
```

Optional Spark integration (downloads JARs; requires working Spark):

```bash
export RUN_SPARK_INTEGRATION=1
pytest tests/test_spark_smoke.py -v
```

---

## 8. Reproducibility for the report

Record: Spark and Python versions, `SPARK_MASTER`, input URI, `--sample-fraction`, LSH parameters, wall-clock and Spark UI shuffle metrics, and candidate pair counts. See [Spark_LSH_Wikipedia_plan.md](Spark_LSH_Wikipedia_plan.md).

---

## 9. Troubleshooting

- **Py4J / SharedState / classpath errors:** Align `pyspark` pip version with your Spark installation, or run only via cluster `spark-submit` without a conflicting `SPARK_HOME`.
- **Out of memory:** Lower `--sample-fraction`, raise executor memory, increase `--num-partitions`, avoid `collect()` on large results.
- **Wrong Scala / spark-xml:** Set `SPARK_XML_COORD` for `spark_submit.sh` or `--spark-xml-package` (e.g. `_2.13` instead of `_2.12`).
- **HDFS permission denied:** Check `hdfs dfs -ls` and your `/user/<name>` home.

---

## 10. Contact

For course-related questions, use your team’s agreed channel (email, LMS, or GitHub Issues if enabled).
