# Scalable Near-Duplicate Detection on Wikipedia (Spark + LSH)

Distributed near-duplicate document detection on large-scale Wikipedia text using **Apache Spark (PySpark)**, **MinHash**, and **Locality Sensitive Hashing (LSH)** on **Hadoop HDFS** (or local `file://` paths).

## Team

| Name          | Student ID  | Role (proposal) |
|---------------|-------------|------------------|
| Basu Singh    | M24AID0008  | Coding, QA review, documentation, GitHub repository |
| Nitin Jain    | M24AID022   | Project report, design and architecture, literature review |
| Shashi Saurav | M24AID0048  | Design, coding, unit testing, version management |

## Repository layout

| Path | Purpose |
|------|---------|
| [src/wiki_near_dup/](src/wiki_near_dup/) | Ingest (spark-xml), `HashingTF` features, `MinHashLSH`, evaluation helpers |
| [scripts/run_pipeline.py](scripts/run_pipeline.py) | CLI: main job, `--eval-scaling`, `--eval-accuracy` |
| [scripts/spark_submit.sh](scripts/spark_submit.sh) | Example `spark-submit` with `spark-xml` Maven coordinate |
| [tests/](tests/) | Pytest (pure Jaccard utils + optional Spark smoke test) |
| [Spark_LSH_Wikipedia_plan.md](Spark_LSH_Wikipedia_plan.md) | Technical plan and evaluation notes |
| [HOW_TO_USE.md](HOW_TO_USE.md) / [HOW_TO_USE.txt](HOW_TO_USE.txt) | Setup and run guide |

## Requirements

- **Python 3.10–3.12** (recommended). PySpark 3.5 + **Python 3.13** may fail (`distutils` / JVM classpath issues depending on your install).
- **Java 11 or 17** and a **Spark** distribution whose version **matches** the `pyspark` version from `pip` (or run only on a cluster where versions are aligned).
- **Maven access** on first run (Spark downloads `com.databricks:spark-xml`).

Install:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# optional editable install
pip install -e .
```

## Run the pipeline

Use the wrapper script so **spark-xml** is on the classpath:

```bash
chmod +x scripts/spark_submit.sh
export SPARK_MASTER=local[*]   # or yarn-client, spark://..., etc.

./scripts/spark_submit.sh \
  --input file:///absolute/path/to/enwiki-...xml.bz2 \
  --output file:///absolute/path/to/out_pairs \
  --sample-fraction 0.05 \
  --num-partitions 64
```

HDFS example:

```bash
./scripts/spark_submit.sh \
  --input hdfs://namenode:8020/user/you/wikipedia/enwiki-...xml.bz2 \
  --output hdfs://namenode:8020/user/you/out/near_dup_pairs
```

### CLI options (high level)

- `--sample-fraction` — fraction of articles after XML parse (for experiments).
- `--num-features` — `HashingTF` space (default `2^18`).
- `--num-hash-tables` — LSH band count (PySpark 3.5 exposes only this for `MinHashLSH`).
- `--jaccard-distance-threshold` — max Jaccard **distance** (1 − similarity) for candidates.

### Scalability study (JSON for report plots)

```bash
./scripts/spark_submit.sh \
  --input file:///path/to/dump.xml.bz2 \
  --eval-scaling \
  --scaling-fractions 0.01,0.02,0.05,0.1 \
  --metrics-json ./metrics_scaling.json
```

### Accuracy vs brute force (small sample only)

Uses all-pairs exact Jaccard on **hashed binary features** for docs in the sample; compare to LSH candidates.

```bash
./scripts/spark_submit.sh \
  --input file:///path/to/dump.xml.bz2 \
  --eval-accuracy \
  --sample-fraction 0.0001 \
  --max-docs-bruteforce 400 \
  --metrics-json ./metrics_accuracy.json
```

## Tests

```bash
pytest tests/test_evaluate_pure.py -q
```

Optional Spark + spark-xml integration (downloads JARs; needs working local Spark):

```bash
export RUN_SPARK_INTEGRATION=1
pytest tests/test_spark_smoke.py -v
```

## Dataset

Wikipedia **pages-articles** XML (often `.xml.bz2`). Example multistream shard: `enwiki-20260301-pages-articles-multistream2.xml-p41243p151573.bz2`.

Do **not** commit large `.bz2` files; use HDFS or local paths. See [.gitignore](.gitignore).

## Matching Spark to `pyspark` from pip

If you install Spark with `pip install pyspark`, use the **`spark-submit` inside that package** and **unset** a global `SPARK_HOME` that points to a different Spark version (otherwise you may see Scala/JVM errors or wrong Spark 4.x behavior):

```bash
SUBMIT="$(python -c "import pyspark, os; print(os.path.join(os.path.dirname(pyspark.__file__), 'bin', 'spark-submit'))")"
env -u SPARK_HOME PYSPARK_PYTHON="$(which python)" "$SUBMIT" ...
```

Python **3.12** needs `setuptools` installed; `run_pipeline.py` imports it before PySpark so `distutils` is available.

## spark-xml Scala version

Default Maven coordinate: `com.databricks:spark-xml_2.12:0.18.0`. If your Spark is built with Scala **2.13**, set:

```bash
export SPARK_XML_COORD='com.databricks:spark-xml_2.13:0.18.0'
./scripts/spark_submit.sh ...
```

## License / course use

Academic project for **Machine Learning for Big Data**. Adapt usage to your institution’s policies.
