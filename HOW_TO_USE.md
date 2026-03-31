# How to use this codebase (after cloning from GitHub)

This guide explains how to set up the environment, place Wikipedia data, and run the near-duplicate detection pipeline. The same content is in [HOW_TO_USE.txt](HOW_TO_USE.txt) for plain-text readers. See also [README.md](README.md) and [Spark_LSH_Wikipedia_plan.md](Spark_LSH_Wikipedia_plan.md).

---

## 1. Clone the repository

```bash
git clone https://github.com/Yoursuperherobasu/BigData_Project_Machine_Learning-_for_Big_Data.git
cd BigData_Project_Machine_Learning-_for_Big_Data
```

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

**Important:** If your `SPARK_HOME` points to a different Spark version (e.g. Spark 4.x) than pip pyspark (3.5.x), unset it before running anything:

```bash
env -u SPARK_HOME streamlit run app.py
# or
env -u SPARK_HOME ./scripts/spark_submit.sh ...
```

---

## 3. Obtain the Wikipedia data (not in Git)

- Download from [https://dumps.wikimedia.org](https://dumps.wikimedia.org) — `enwiki-*-pages-articles*.xml.bz2` or a **multistream** shard of the same schema.
- Keep `.bz2` files out of Git (see `.gitignore`).
- You can also **upload datasets directly through the Streamlit UI** (see section 4).
- HDFS upload example:

```bash
hdfs dfs -mkdir -p /user/<you>/wikipedia
hdfs dfs -put /local/path/to/your-file.xml.bz2 /user/<you>/wikipedia/
```

---

## 4. Run using the Web UI (recommended)

The easiest way to use the project:

```bash
source .venv/bin/activate
streamlit run app.py
```

Opens at `http://localhost:8501`. The UI has five modes:

### Run Pipeline
- **Upload** a Wikipedia XML or .bz2 file, **or** select an existing file from the project folder, **or** enter a path manually.
- Adjust parameters: sample % (how much of the data to use), hash tables, hash features, distance threshold.
- Choose what to run: Main pipeline (find duplicates), Accuracy evaluation, or Scaling study.
- Click **Run** and wait. Results appear as an interactive table showing near-duplicate pairs with Jaccard distances. You can download results as CSV.

### Jaccard Calculator
- Enter two sets of words and see their Jaccard similarity and distance instantly.

### Text Comparison
- Enter multiple short documents and find near-duplicate pairs using brute-force Jaccard comparison.

### View Results
- Load metrics JSON files from previous pipeline runs to view accuracy (precision, recall, F1) or scaling study results.

### About
- Project info, default parameters, and tech stack.

### Approximate runtimes

| Sample % | Articles | Estimated Time |
|----------|----------|---------------|
| 1%       | ~1,000   | 1-2 min       |
| 5%       | ~5,000   | 3-5 min       |
| 10%      | ~10,000  | 5-8 min       |
| 50%      | ~50,000  | 15-25 min     |
| 100%     | ~100,000 | 30-50 min     |

---

## 5. Run the main pipeline (CLI)

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

- `--sample-fraction` — fraction of articles to use (e.g. 0.05 = 5%). Smaller = faster.
- `--num-hash-tables` — LSH parameter (more tables = better accuracy but slower).
- `--jaccard-distance-threshold` — max Jaccard distance (1 - similarity) for pairs.
- `--spark-xml-package` — override Maven coordinate if your Spark uses Scala 2.13.

---

## 6. Scalability experiment (JSON metrics)

```bash
./scripts/spark_submit.sh \
  --input file:///absolute/path/to/dump.xml.bz2 \
  --eval-scaling \
  --scaling-fractions 0.01,0.02,0.05,0.1 \
  --metrics-json ./metrics_scaling.json
```

Use the JSON fields `n_docs`, `n_candidate_pairs`, and `time_*` for report plots. You can also run this from the Streamlit UI by selecting "Scaling study".

---

## 7. Accuracy check (small sample only)

Brute-force **all pairs** on the sample (hashed binary features) vs LSH candidates. Keep `--sample-fraction` tiny so document count stays below `--max-docs-bruteforce` (default 400).

```bash
./scripts/spark_submit.sh \
  --input file:///absolute/path/to/dump.xml.bz2 \
  --eval-accuracy \
  --sample-fraction 0.0001 \
  --max-docs-bruteforce 400 \
  --metrics-json ./metrics_accuracy.json
```

Or use "Accuracy evaluation" in the Streamlit UI.

---

## 8. Tests

```bash
pytest tests/test_evaluate_pure.py -q
```

Optional Spark integration (downloads JARs; requires working Spark):

```bash
env -u SPARK_HOME RUN_SPARK_INTEGRATION=1 pytest tests/test_spark_smoke.py -v
```

---

## 9. Reproducibility for the report

Record: Spark and Python versions, `SPARK_MASTER`, input URI, sample fraction, LSH parameters, wall-clock and Spark UI shuffle metrics, and candidate pair counts. See [Spark_LSH_Wikipedia_plan.md](Spark_LSH_Wikipedia_plan.md).

---

## 10. Troubleshooting

- **Py4J / SharedState / classpath errors:** Unset `SPARK_HOME` if it points to a different Spark version than pip pyspark. Run with `env -u SPARK_HOME ...`.
- **Out of memory:** Lower sample %, raise executor memory, increase `--num-partitions`, avoid `collect()` on large results.
- **Wrong Scala / spark-xml:** Set `SPARK_XML_COORD` for `spark_submit.sh` or `--spark-xml-package` (e.g. `_2.13` instead of `_2.12`).
- **HDFS permission denied:** Check `hdfs dfs -ls` and your `/user/<name>` home.
- **Streamlit not starting:** Make sure `streamlit` is installed (`pip install -r requirements.txt`).

---

## 11. Contact

For course-related questions, use your team's agreed channel (email, LMS, or GitHub Issues if enabled).
