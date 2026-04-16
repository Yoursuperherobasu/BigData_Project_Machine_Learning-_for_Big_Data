# Scalable Near-Duplicate Detection on Wikipedia (Spark + LSH)

Distributed near-duplicate document detection on large-scale Wikipedia text using **Apache Spark (PySpark)**, **MinHash**, and **Locality Sensitive Hashing (LSH)** on **Hadoop HDFS** (or local `file://` paths). Includes a **Streamlit web UI** for interactive exploration and pipeline execution.

> **Dataset notice:** The Wikipedia XML dump is **not committed to this repository** — the files are too large for GitHub (hundreds of MB to several GB). Download it yourself from [dumps.wikimedia.org](https://dumps.wikimedia.org) or upload via the Streamlit UI. See **Step 5** below.

## Team

| Name          | Student ID  | Role (proposal) |
|---------------|-------------|------------------|
| Basu Singh    | M24AID008   | Coding, QA review, documentation, GitHub repository |
| Nitin Jain    | M24AID022   | Project report, design and architecture, literature review |
| Shashi Saurav | M24AID048   | Design, coding, unit testing, version management |

## Repository layout

| Path | Purpose |
|------|---------|
| [app.py](app.py) | Streamlit web UI (run pipeline, Jaccard calculator, view results) |
| [src/wiki_near_dup/](src/wiki_near_dup/) | Ingest (spark-xml), `HashingTF` features, `MinHashLSH`, evaluation helpers |
| [scripts/run_pipeline.py](scripts/run_pipeline.py) | CLI: main job, `--eval-scaling`, `--eval-accuracy` |
| [scripts/spark_submit.sh](scripts/spark_submit.sh) | Example `spark-submit` with `spark-xml` Maven coordinate |
| [Project_Report.docx](Project_Report.docx) | Project report |

---

# Step-by-step setup and run guide

Follow these steps in order. Steps 1–4 are one-time setup; steps 5–7 are how you run the project day-to-day.

## Step 1 — Install prerequisites

Before cloning, make sure the following are installed on your machine:

- **Python 3.10, 3.11, or 3.12** (do **not** use 3.13 — PySpark fails because `distutils` was removed).
  - Check: `python3.12 --version`
- **Java 11 or 17** (required by Spark).
  - Check: `java -version`
- **Git** (to clone the repo).
  - Check: `git --version`
- **Internet access** on first run (Spark downloads the `spark-xml` JAR from Maven Central).

Optional (only if you plan to run on a Hadoop cluster):
- **Apache Spark** whose version matches the `pyspark` pinned in `requirements.txt` (3.4–3.5.x).
- **Hadoop HDFS** client tools (`hdfs dfs`).

## Step 2 — Clone the repository

```bash
git clone https://github.com/Yoursuperherobasu/BigData_Project_Machine_Learning-_for_Big_Data.git
cd BigData_Project_Machine_Learning-_for_Big_Data
```

## Step 3 — Create and activate a Python virtual environment

Use Python 3.12 (or 3.10/3.11). **Do not use Python 3.13** — PySpark will fail to import.

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate           # macOS / Linux
# Windows PowerShell:  .venv312\Scripts\Activate.ps1
```

You should now see `(.venv312)` at the start of your shell prompt. Confirm:

```bash
python --version      # should print Python 3.12.x
which python          # should point inside .venv312
```

## Step 4 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                       # optional: editable install of wiki_near_dup
```

This installs `pyspark`, `numpy`, `streamlit`, `pytest`, and `setuptools` (the last is required on Python 3.12+ to supply `distutils` for PySpark).

## Step 5 — Get the Wikipedia dataset

> **Note:** The Wikipedia dataset is **not included in this GitHub repository** because the file size is far too large to host on GitHub (the sample shard alone is ~380 MB, and full dumps are many GB). Wikipedia `.bz2` files are explicitly excluded via `.gitignore`. You must obtain the data yourself using one of the options below.

You have three options:

**Option A — Download from Wikimedia:**
Visit [https://dumps.wikimedia.org](https://dumps.wikimedia.org) and grab any `enwiki-*-pages-articles*.xml.bz2` file or a multistream shard. Save it anywhere on your machine (or in the project folder — `*.bz2` is already `.gitignore`d).

**Option B — Upload through the Streamlit UI:**
Skip this step; you can upload an XML or `.xml.bz2` file directly in the UI (step 6). Upload limit is 1 GB.

**Option C — Use HDFS (cluster deployments):**
```bash
hdfs dfs -mkdir -p /user/<you>/wikipedia
hdfs dfs -put /local/path/to/your-file.xml.bz2 /user/<you>/wikipedia/
```

## Step 6 — Run the Streamlit Web UI (recommended)

**Important:** If your system has a `SPARK_HOME` environment variable pointing to a different Spark version (e.g. Spark 4.x) than the pip-installed `pyspark` (3.5.x), **unset it first** — otherwise you will get JVM / classpath errors.

```bash
source .venv312/bin/activate           # if not already active
env -u SPARK_HOME streamlit run app.py
```

The UI opens at **http://localhost:8501**. You have five modes:

1. **Run Pipeline** — Select an existing file, upload a new `.xml` / `.xml.bz2`, or enter a path. Tune sample %, hash tables, hash features, and distance threshold. Choose **Main pipeline** (find duplicates), **Accuracy evaluation**, or **Scaling study**. Click **Run**. Results appear as an interactive table; download as CSV.
2. **Jaccard Calculator** — Enter two sets of words; see Jaccard similarity and distance instantly.
3. **Text Comparison** — Paste 2–10 short documents; get near-duplicate pairs via brute-force Jaccard.
4. **View Results** — Load a metrics JSON from a previous run to see precision/recall/F1 or scaling-study charts.
5. **About** — Project info, default parameters, and tech stack.

**Approximate runtimes** (single machine, ~100K articles after filtering):

| Sample % | Articles | Estimated Time |
|----------|----------|----------------|
| 1%       | ~1,000   | 1–2 min        |
| 5%       | ~5,000   | 3–5 min        |
| 10%      | ~10,000  | 5–8 min        |
| 50%      | ~50,000  | 15–25 min      |
| 100%     | ~100,000 | 30–50 min      |

## Step 7 — Run the pipeline from the CLI (alternative)

If you prefer a terminal workflow instead of the UI:

```bash
chmod +x scripts/spark_submit.sh
export SPARK_MASTER=local[*]           # or yarn-client, spark://..., etc.

env -u SPARK_HOME ./scripts/spark_submit.sh \
  --input file:///absolute/path/to/enwiki-....xml.bz2 \
  --output file:///absolute/path/to/out_pairs \
  --sample-fraction 0.05 \
  --num-partitions 64
```

Output is Parquet with columns: `page_id_a`, `title_a`, `page_id_b`, `title_b`, `jaccardDist`.

**HDFS input/output:**
```bash
./scripts/spark_submit.sh \
  --input hdfs://namenode:8020/user/you/wikipedia/enwiki-....xml.bz2 \
  --output hdfs://namenode:8020/user/you/out/near_dup_pairs
```

### Useful CLI flags

- `--sample-fraction` — fraction of articles after XML parse (e.g. `0.05` = 5%). Smaller = faster.
- `--num-features` — `HashingTF` space (default `2^18` = 262,144).
- `--num-hash-tables` — LSH band count (default 5; more = better recall, slower).
- `--jaccard-distance-threshold` — max Jaccard **distance** (1 − similarity) for candidates (default 0.3).
- `--spark-xml-package` — override Maven coordinate if your Spark uses Scala 2.13.

### Scalability study (JSON metrics for plots)

```bash
env -u SPARK_HOME ./scripts/spark_submit.sh \
  --input file:///path/to/dump.xml.bz2 \
  --eval-scaling \
  --scaling-fractions 0.01,0.02,0.05,0.1 \
  --metrics-json ./metrics_scaling.json
```

### Accuracy vs brute force (small sample only)

Uses all-pairs exact Jaccard on **hashed binary features** for docs in the sample and compares to LSH candidates. Keep the sample small so the doc count stays below `--max-docs-bruteforce`.

```bash
env -u SPARK_HOME ./scripts/spark_submit.sh \
  --input file:///path/to/dump.xml.bz2 \
  --eval-accuracy \
  --sample-fraction 0.0001 \
  --max-docs-bruteforce 5000 \
  --metrics-json ./metrics_accuracy.json
```

---

## Troubleshooting

- **Py4J / SharedState / classpath errors, or "Java gateway process exited":** Your `SPARK_HOME` points to a different Spark version than pip `pyspark`. Always prefix commands with `env -u SPARK_HOME ...`.
- **`ModuleNotFoundError: No module named 'distutils'`:** You are on Python 3.13. Recreate the venv with Python 3.10/3.11/3.12.
- **Streamlit will not start:** Reinstall deps with `pip install -r requirements.txt` inside the active venv.
- **Wrong Scala / spark-xml version:** Set `SPARK_XML_COORD='com.databricks:spark-xml_2.13:0.18.0'` (or pass `--spark-xml-package`) if your Spark is built with Scala 2.13.
- **Out of memory:** Lower `--sample-fraction`, raise executor memory, increase `--num-partitions`, and avoid `collect()` on large results.
- **HDFS permission denied:** Check `hdfs dfs -ls /user/<you>` and your home directory permissions.

## Matching Spark to `pyspark` from pip

If you installed Spark via `pip install pyspark`, use the **`spark-submit` inside that package** and unset any global `SPARK_HOME`:

```bash
SUBMIT="$(python -c 'import pyspark, os; print(os.path.join(os.path.dirname(pyspark.__file__), "bin", "spark-submit"))')"
env -u SPARK_HOME PYSPARK_PYTHON="$(which python)" "$SUBMIT" ...
```

## License / course use

Academic project for **Machine Learning for Big Data**, IIT Jodhpur. M.Tech. in Data and Computational Sciences.
