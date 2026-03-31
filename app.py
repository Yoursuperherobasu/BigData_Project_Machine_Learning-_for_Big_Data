"""
Simple Streamlit UI for Wikipedia Near-Duplicate Detection.
Run: streamlit run app.py
"""

import sys
import os
from pathlib import Path

# add src to path so we can import our modules
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
import time
import json

from wiki_near_dup.jaccard_utils import (
    jaccard_similarity_indices,
    jaccard_distance_indices,
    brute_force_similar_pairs_sets,
    precision_recall_f1,
)

st.set_page_config(page_title="Wiki Near-Dup Detector", layout="wide")

st.title("Wikipedia Near-Duplicate Detection")
st.write("Using MinHash + Locality-Sensitive Hashing on Apache Spark")
st.write("---")

# sidebar
st.sidebar.header("Settings")
mode = st.sidebar.radio("Choose mode:", ["Jaccard Calculator", "Text Comparison", "View Results", "About"])


# ---- Jaccard Calculator ----
if mode == "Jaccard Calculator":
    st.header("Jaccard Similarity Calculator")
    st.write("Enter two sets of words to compute their Jaccard similarity and distance.")

    col1, col2 = st.columns(2)
    with col1:
        text_a = st.text_area("Document A (words separated by spaces)", value="apple banana cherry date elderberry")
    with col2:
        text_b = st.text_area("Document B (words separated by spaces)", value="banana cherry fig grape apple")

    if st.button("Compute"):
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        if not words_a and not words_b:
            st.warning("Please enter some words in both fields.")
        else:
            # convert words to integer indices for our function
            all_words = sorted(words_a | words_b)
            word_to_idx = {w: i for i, w in enumerate(all_words)}
            set_a = {word_to_idx[w] for w in words_a}
            set_b = {word_to_idx[w] for w in words_b}

            sim = jaccard_similarity_indices(set_a, set_b)
            dist = jaccard_distance_indices(set_a, set_b)

            st.write("### Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Jaccard Similarity", f"{sim:.4f}")
            c2.metric("Jaccard Distance", f"{dist:.4f}")
            c3.metric("Common Words", str(len(words_a & words_b)))

            st.write("**Set A:**", words_a)
            st.write("**Set B:**", words_b)
            st.write("**Intersection:**", words_a & words_b)
            st.write("**Union:**", words_a | words_b)


# ---- Text Comparison ----
elif mode == "Text Comparison":
    st.header("Compare Multiple Documents")
    st.write("Enter multiple short documents and find near-duplicate pairs using brute-force Jaccard.")

    num_docs = st.slider("Number of documents", 2, 10, 3)

    docs = {}
    for i in range(num_docs):
        docs[i] = st.text_input(f"Document {i+1}", value="" if i > 0 else "the cat sat on the mat", key=f"doc_{i}")

    threshold = st.slider("Max Jaccard Distance (threshold)", 0.0, 1.0, 0.5, 0.05)

    if st.button("Find Near-Duplicates"):
        # tokenize each doc
        id_to_tokens = {}
        all_tokens = set()
        for i, text in docs.items():
            tokens = set(text.lower().split())
            id_to_tokens[i] = tokens
            all_tokens |= tokens

        if not all_tokens:
            st.warning("Please enter text in at least one document.")
        else:
            token_to_idx = {t: idx for idx, t in enumerate(sorted(all_tokens))}
            id_to_indices = {}
            for i, tokens in id_to_tokens.items():
                id_to_indices[i] = {token_to_idx[t] for t in tokens}

            pairs = brute_force_similar_pairs_sets(id_to_indices, threshold)

            if pairs:
                st.success(f"Found {len(pairs)} near-duplicate pair(s)!")
                for a, b in sorted(pairs):
                    sim = jaccard_similarity_indices(id_to_indices[a], id_to_indices[b])
                    dist = jaccard_distance_indices(id_to_indices[a], id_to_indices[b])
                    st.write(f"- **Doc {a+1}** & **Doc {b+1}**: similarity = {sim:.3f}, distance = {dist:.3f}")
            else:
                st.info("No near-duplicate pairs found at this threshold.")


# ---- View Results ----
elif mode == "View Results":
    st.header("View Pipeline Results")
    st.write("Load and view output from a previous Spark pipeline run.")

    results_path = st.text_input("Path to metrics JSON file", value="metrics_accuracy.json")

    if st.button("Load"):
        if os.path.exists(results_path):
            with open(results_path) as f:
                data = json.load(f)
            st.json(data)

            # if it has accuracy metrics show them nicely
            if "precision" in data and data["precision"] is not None:
                st.write("### Accuracy Metrics")
                c1, c2, c3 = st.columns(3)
                c1.metric("Precision", f"{data['precision']:.4f}")
                c2.metric("Recall", f"{data['recall']:.4f}")
                c3.metric("F1 Score", f"{data['f1']:.4f}")

            # if it has scaling runs show them
            if "scaling_runs" in data:
                st.write("### Scaling Study")
                for run in data["scaling_runs"]:
                    st.write(f"- **{run.get('sample_fraction', '?')}** fraction: "
                             f"{run.get('n_docs', '?')} docs, "
                             f"{run.get('n_candidate_pairs', '?')} pairs, "
                             f"{run.get('time_total_sec', '?'):.1f}s total")
        else:
            st.error(f"File not found: {results_path}")

    st.write("---")
    st.write("**Note:** Run the Spark pipeline first to generate metrics files.")
    st.code(
        "./scripts/spark_submit.sh \\\n"
        "  --input file:///path/to/dump.xml.bz2 \\\n"
        "  --eval-accuracy \\\n"
        "  --sample-fraction 0.001 \\\n"
        "  --metrics-json ./metrics_accuracy.json",
        language="bash"
    )


# ---- About ----
elif mode == "About":
    st.header("About This Project")

    st.write("""
    This project implements a scalable near-duplicate detection pipeline for Wikipedia articles
    using Apache Spark (PySpark) and MinHash Locality-Sensitive Hashing (LSH).

    **Pipeline stages:**
    1. Ingest Wikipedia XML dump using spark-xml
    2. Filter articles (remove redirects, stubs, non-article pages)
    3. Tokenize and build binary feature vectors (HashingTF)
    4. Fit MinHash LSH model
    5. Self-join to find candidate near-duplicate pairs
    6. Output pairs with Jaccard distance to Parquet

    **Tech stack:**
    - Python 3.12, PySpark 3.4+
    - Spark MLlib (MinHashLSH)
    - spark-xml for XML parsing
    - Hadoop HDFS (optional, for cluster deployment)
    """)

    st.write("---")

    st.write("### Default Parameters")
    st.table({
        "Parameter": ["Hash features", "Hash tables", "Distance threshold", "Min token length"],
        "Value": ["262,144 (2^18)", "5", "0.3", "2"],
    })

    st.write("---")
    st.write("**Course:** Machine Learning for Big Data")
    st.write("**Institute:** IIT Jodhpur")
    st.write("**Program:** M.Tech, Data and Computational Sciences")
