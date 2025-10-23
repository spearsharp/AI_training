# -*- coding: utf-8 -*-
"""
Training and prediction pipeline wrapper。

Exposed core functions：
- train_lottery_models：Train model based on historical data and save locally；
- load_trained_models：Load trained model from disk；
- predict_next_draw：Generate prediction result using latest window data。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Compatibility shim: older h5py / TensorFlow internals expect numpy.typeDict which was
# removed/renamed in newer numpy versions. Provide a safe alias to avoid import-time
# AttributeError when loading TensorFlow/h5py in environments with newer numpy.
if not hasattr(np, "typeDict"):
    try:
        np.typeDict = np.sctypeDict  # type: ignore[attr-defined]
    except Exception:
        np.typeDict = {}

# Defer importing heavy ML frameworks (tensorflow/h5py) until actually needed. This
# avoids import-time crashes in environments where binary extensions are incompatible.

from .common import logger

from .config import (
    DATA_FILE_NAME,
    MODEL_METADATA_FILE,
    PATHS,
    LotteryModelConfig,
    ensure_runtime_directories,
    get_lottery_config,
)
from .data_fetcher import load_history
from .modeling import build_models_for_lottery
from .preprocessing import ComponentDataset, prepare_training_arrays, train_validation_split


@dataclass
class ComponentTrainingSummary:
    train_samples: int
    val_samples: int
    best_val_loss: Optional[float]
    best_val_metric: Optional[float]
    epochs_trained: int


@dataclass
class TrainingSummary:
    code: str
    name: str
    window_size: int
    trained_on_issues: Tuple[str, str]
    components: Dict[str, ComponentTrainingSummary]
    timestamp: str


def _ensure_enough_samples(dataset: ComponentDataset, window_size: int, name: str) -> None:
    if dataset.features.shape[0] == 0:
        raise ValueError(
            f"{name} Insufficient data available，window size {window_size} generated samples count is 0，please increase historical issues or decrease window size。"
        )


def _build_tf_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    try:
        import tensorflow as tf  # imported locally to avoid global import-time errors
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Unable to import TensorFlow：{}。Please use the project's recommended virtual environment or conda environment and install requirements.txt".format(exc)
        )

    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        buffer = min(len(features), max(batch_size * 4, 256))
        ds = ds.shuffle(buffer)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _denormalize(pred: np.ndarray, spec_classes: int) -> np.ndarray:
    if pred.min() < 0:
        raise ValueError("Prediction result contains negative numbers, possibly model output anomaly")
    return pred + 1


def _get_latest_window(arr: np.ndarray, window_size: int) -> np.ndarray:
    if arr.shape[0] < window_size:
        raise ValueError(f"Insufficient historical data，unable to get {window_size} window sequences from original array")
    return arr[-window_size:]


def train_lottery_models(
    code: str,
    window_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    red_epochs: Optional[int] = None,
    blue_epochs: Optional[int] = None,
    validation_ratio: float = 0.15,
) -> TrainingSummary:
    """Train model for specified lottery and return training summary."""

    ensure_runtime_directories()
    cfg: LotteryModelConfig = get_lottery_config(code)
    df = load_history(cfg.code)
    window = window_size or cfg.default_window
    arrays = prepare_training_arrays(df, cfg, window)
    summary_components: Dict[str, ComponentTrainingSummary] = {}

    try:
        import tensorflow as tf  # local import
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Unable to import TensorFlow：{}。Please use the project's recommended virtual environment or conda environment and install requirements.txt".format(exc)
        )

    models = build_models_for_lottery(cfg, window)
    save_dir = PATHS["model"] / cfg.code / f"window_{window}"
    save_dir.mkdir(parents=True, exist_ok=True)

    first_issue = str(df["Issue"].min())
    last_issue = str(df["Issue"].max())

    for component, model in models.items():
        dataset = arrays[component]
        _ensure_enough_samples(dataset, window, f"{cfg.name}-{component}")
        (x_train, y_train), (x_val, y_val) = train_validation_split(
            dataset.features, dataset.labels, validation_ratio=validation_ratio
        )
        effective_batch = max(1, min(batch_size or cfg.default_batch_size, x_train.shape[0]))
        train_ds = _build_tf_dataset(x_train, y_train, effective_batch, shuffle=True)
        val_ds = None
        if x_val.shape[0] > 0:
            val_ds = _build_tf_dataset(x_val, y_val, effective_batch, shuffle=False)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=1,
            ),
        ]
        if val_ds is None:
            callbacks = []

        epochs = red_epochs if component == "red" else blue_epochs
        if epochs is None:
            epochs = cfg.default_red_epochs if component == "red" else cfg.default_blue_epochs
        epochs = max(1, epochs)

        logger.info(
            "Train model {}-{}: samples={}，validation set={}，window={}，batch size={}，epochs={}",
            cfg.code,
            component,
            dataset.features.shape[0],
            x_val.shape[0],
            window,
            effective_batch,
            epochs,
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks,
        )

        model_path = save_dir / f"{component}.keras"
        model.save(model_path, overwrite=True)
        logger.success("Model saved to {}", model_path)

        best_loss = min(history.history.get("val_loss", history.history.get("loss", [None])))
        metric_key = None
        for candidate in ("val_accuracy", "val_sparse_categorical_accuracy", "accuracy"):
            if candidate in history.history:
                metric_key = candidate
                break
        best_metric = None
        if metric_key is not None:
            best_metric = max(history.history[metric_key])
        summary_components[component] = ComponentTrainingSummary(
            train_samples=int(x_train.shape[0]),
            val_samples=int(x_val.shape[0]),
            best_val_loss=float(best_loss) if best_loss is not None else None,
            best_val_metric=float(best_metric) if best_metric is not None else None,
            epochs_trained=len(history.history.get("loss", [])),
        )

    metadata = TrainingSummary(
        code=cfg.code,
        name=cfg.name,
        window_size=window,
        trained_on_issues=(first_issue, last_issue),
        components=summary_components,
        timestamp=datetime.utcnow().isoformat(),
    )
    metadata_path = save_dir / MODEL_METADATA_FILE
    metadata_path.write_text(
        json.dumps(
            {
                **asdict(metadata),
                "components": {key: asdict(value) for key, value in summary_components.items()},
                "data_file": str(PATHS["data"] / cfg.code / DATA_FILE_NAME),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.success("Training summary written to {}", metadata_path)
    return metadata


def load_trained_models(code: str, window_size: Optional[int] = None) -> Dict[str, tf.keras.Model]:
    """Load trained model from disk."""

    cfg = get_lottery_config(code)
    window = window_size or cfg.default_window
    directory = PATHS["model"] / cfg.code / f"window_{window}"
    if not directory.exists():
        raise FileNotFoundError(f"Trained model directory not found: {directory}")

    try:
        import tensorflow as tf  # local import
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Unable to import TensorFlow：{}。Please use the project's recommended virtual environment or conda environment and install requirements.txt".format(exc)
        )

    models: Dict[str, tf.keras.Model] = {}

    for component in ("red", "blue"):
        model_path = directory / f"{component}.keras"
        if model_path.exists():
            models[component] = tf.keras.models.load_model(
                model_path,
                compile=True,
                safe_mode=False,
            )
            logger.info("Loading model {}", model_path)
    if not models:
        raise FileNotFoundError(f"{directory} , red/blue model files not found")
    return models


def predict_next_draw(
    code: str,
    window_size: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Use latest model to predict next draw numbers。"""

    cfg = get_lottery_config(code)
    window = window_size or cfg.default_window
    df = load_history(cfg.code)
    arrays = prepare_training_arrays(df, cfg, window)
    models = load_trained_models(cfg.code, window)

    predictions: Dict[str, np.ndarray] = {}

    # Latest window features taken from last window entries in prepare_training_arrays
    red_dataset = arrays["red"]
    latest_features = _get_latest_window(red_dataset.features, 1).reshape(1, window, cfg.red.sequence_len)

    red_model = models["red"]
    red_pred = red_model.predict(latest_features, verbose=0)  # shape: (1, ball positions, number of classes)
    red_pred = red_pred.squeeze(axis=0)  # shape: (ball positions, number of classes)
    num_balls = cfg.red.sequence_len
    num_classes = cfg.red.num_classes
    # Greedy deduplication sampling: select unselected number with highest probability each time
    chosen = set()
    result = []
    for i in range(num_balls):
        probs = red_pred[i].copy()
        # Set probability of already selected numbers to-1，to avoid duplication
        for idx in chosen:
            if idx < len(probs):
                probs[idx] = -1
        pick = int(np.argmax(probs))
        chosen.add(pick)
        result.append(pick)
    predictions["red"] = np.array(result, dtype=int)

    if cfg.blue and "blue" in models:
        blue_dataset = arrays["blue"]
        latest_blue = _get_latest_window(blue_dataset.features, 1).reshape(1, window, cfg.blue.sequence_len)
        blue_pred_raw = models["blue"].predict(latest_blue, verbose=0)
        predictions["blue"] = np.argmax(blue_pred_raw, axis=-1).squeeze(axis=0).astype(int)

    # Convert prediction result back to original numbering（0-based -> 1-based）
    if red_dataset.needs_offset:
        predictions["red"] = _denormalize(predictions["red"], cfg.red.num_classes)
    if "blue" in predictions:
        blue_dataset = arrays["blue"]
        if blue_dataset.needs_offset:
            predictions["blue"] = _denormalize(predictions["blue"], cfg.blue.num_classes)
    return predictions


__all__ = ["train_lottery_models", "load_trained_models", "predict_next_draw", "TrainingSummary"]
