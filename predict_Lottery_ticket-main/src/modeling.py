# -*- coding: utf-8 -*-
"""
Model building module。

Provides TensorFlow 2.15.1 multi-layer LSTM sequence model, targeting Red/Blue outputs
position-wise category probabilities。
"""

from __future__ import annotations

from typing import Dict

import importlib
import warnings

try:
    import tensorflow as tf
    TF_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - runtime environment dependent
    tf = None  # type: ignore
    TF_IMPORT_ERROR = exc

# Use the project's logger shim (may be backed by loguru or stdlib logging)
from .common import logger


# Ensure tf.keras is available; prefer bundled tf.keras over standalone `keras` package.
if tf is None or not hasattr(tf, "keras"):
    # Defer error reporting to `_ensure_tf_available()` which provides actionable
    # installation instructions instead of attempting to import `keras` here.
    pass

from src.config import LotteryModelConfig, SequenceModelSpec


def _time_distributed_lstm(
    inputs: tf.Tensor,
    units: int,
    name: str,
) -> tf.Tensor:
    """Apply LSTM independently to each ball position, extract window dimension features。"""

    layer = tf.keras.layers.TimeDistributed(
        tf.keras.layers.LSTM(units, return_sequences=False, name=f"{name}_inner"),
        name=name,
    )
    return layer(inputs)


def _ensure_tf_available() -> None:
    """Raise a clear SystemExit with installation instructions if TensorFlow isn't available.

    This avoids import-time crashes and gives the user actionable commands to fix the
    environment. The message covers Intel macOS (tensorflow-intel) and Apple Silicon
    (tensorflow-macos + tensorflow-metal) installation hints.
    """
    if TF_IMPORT_ERROR is None:
        return
    # Detect platform guidance
    import platform

    arch = platform.machine()
    instructions = []
    instructions.append("Create and activate the project's recommended conda environment (environment.yml)：")
    instructions.append("  conda env create -f environment.yml && conda activate predict_lottery")
    if arch == "x86_64":
        instructions.append("Install TensorFlow (Intel Mac / x86_64):")
        instructions.append("  python -m pip install tensorflow-intel==2.15.1")
    else:
        instructions.append("Apple Silicon (arm64) users can install：")
        instructions.append("  python -m pip install tensorflow-macos==2.15.1 tensorflow-metal")
    instructions.append("or run after activating environment: python -m pip install -r requirements.lock.txt")

    msg = (
        "TensorFlow import failed：{0!s}\nPlease set up the environment following these steps and retry：\n\n{1}".format(
            TF_IMPORT_ERROR, "\n".join(instructions)
        )
    )
    raise SystemExit(msg)


def build_sequence_model(
    spec: SequenceModelSpec,
    window_size: int,
    learning_rate: float,
    name: str,
) -> tf.keras.Model:
    """Build sequence model according to given specifications。"""

    _ensure_tf_available()

    inputs = tf.keras.layers.Input(
        shape=(window_size, spec.sequence_len),
        dtype=tf.int32,
        name=f"{name}_input",
    )
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=spec.num_classes,
        output_dim=spec.embedding_dim,
        embeddings_initializer="he_normal",
        name=f"{name}_embedding",
    )
    embedded = embedding_layer(inputs)  # (batch, window, seq_len, embed_dim)
    # Swap ball position and time dimensions to facilitate LSTM for each ball
    per_ball_sequence = tf.transpose(embedded, perm=(0, 2, 1, 3), name=f"{name}_permute")
    per_ball_encoded = _time_distributed_lstm(
        per_ball_sequence,
        units=int(spec.hidden_units[0]),
        name=f"{name}_per_ball_lstm",
    )

    x = per_ball_encoded
    for layer_idx, units in enumerate(spec.hidden_units[1:], start=1):
        x = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            dropout=spec.dropout,
            recurrent_dropout=0.0,
            name=f"{name}_global_lstm_{layer_idx}",
        )(x)

    if len(spec.hidden_units) == 1:
        x = tf.keras.layers.LSTM(
            spec.hidden_units[0],
            return_sequences=True,
            dropout=spec.dropout,
            name=f"{name}_global_lstm",
        )(x)

    x = tf.keras.layers.Dropout(spec.dropout, name=f"{name}_dropout")(x)
    logits = tf.keras.layers.Dense(
        spec.num_classes,
        name=f"{name}_logits",
    )(x)
    output = tf.keras.layers.Activation("softmax", name=f"{name}_softmax")(logits)

    model = tf.keras.Model(inputs=inputs, outputs=output, name=f"{name}_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    logger.debug("Building model {}：window={}，sequence length={}，number of classes={}", name, window_size, spec.sequence_len, spec.num_classes)
    return model


def build_models_for_lottery(
    config: LotteryModelConfig,
    window_size: int,
) -> Dict[str, tf.keras.Model]:
    """Build red/blue models for specified lottery。"""
    _ensure_tf_available()

    models: Dict[str, tf.keras.Model] = {
        "red": build_sequence_model(config.red, window_size, config.learning_rate, f"{config.code}_red"),
    }
    if config.blue:
        models["blue"] = build_sequence_model(
            config.blue,
            window_size,
            config.learning_rate,
            f"{config.code}_blue",
        )
    return models


__all__ = ["build_models_for_lottery", "build_sequence_model"]
