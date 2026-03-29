"""Tokenization and binary sparse features for MinHash / Jaccard."""

from __future__ import annotations

from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, RegexTokenizer
from pyspark.sql import DataFrame

from wiki_near_dup.config import DEFAULT_HASHING_NUM_FEATURES, DEFAULT_MIN_TOKEN_LENGTH


def build_featurizer(
    *,
    num_features: int = DEFAULT_HASHING_NUM_FEATURES,
    min_token_length: int = DEFAULT_MIN_TOKEN_LENGTH,
) -> Pipeline:
    tokenizer = RegexTokenizer(
        inputCol="text",
        outputCol="tokens",
        pattern=r"\W+",
        minTokenLength=min_token_length,
        gaps=True,
        toLowercase=True,
    )
    hashing_tf = HashingTF(
        inputCol="tokens",
        outputCol="features",
        numFeatures=num_features,
        binary=True,
    )
    return Pipeline(stages=[tokenizer, hashing_tf])


def featurize(df: DataFrame, num_features: int = DEFAULT_HASHING_NUM_FEATURES) -> tuple[Pipeline, DataFrame]:
    """Fit tokenizer + HashingTF and return (fitted pipeline, dataframe with ``features``)."""
    pipe = build_featurizer(num_features=num_features)
    model = pipe.fit(df)
    return model, model.transform(df)
