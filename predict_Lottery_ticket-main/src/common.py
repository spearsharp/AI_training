# -*- coding: utf-8 -*-
"""
Common interface wrapper。

Provides the following capabilities for script layer：
1. Downloadinghistorical data：`get_data_run`
2. Query latest issue number：`get_current_number`
3. Train model：`train_pipeline`
4. Predict next draw：`predict_latest`
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

try:  # pragma: no cover - optional dependency
    from loguru import logger
except Exception:  # pragma: no cover - provide fallback when loguru is not installed
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _std_logger = logging.getLogger("common")

    class _SimpleLogger:
        def _fmt(self, msg: str, *args):
            try:
                return msg.format(*args) if args else msg
            except Exception:
                return msg

        def info(self, msg: str, *args, **kwargs):
            _std_logger.info(self._fmt(msg, *args))

        def success(self, msg: str, *args, **kwargs):
            _std_logger.info(self._fmt(msg, *args))

        def warning(self, msg: str, *args, **kwargs):
            _std_logger.warning(self._fmt(msg, *args))

        def error(self, msg: str, *args, **kwargs):
            _std_logger.error(self._fmt(msg, *args))

    logger = _SimpleLogger()

from .config import LOTTERY_CONFIGS, ensure_runtime_directories, get_lottery_config
from .data_fetcher import download_history, get_current_issue, load_history

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import TrainingSummary


_PIPELINE_MODULE = None


def _load_pipeline():
    global _PIPELINE_MODULE
    if _PIPELINE_MODULE is None:
        from . import pipeline as _pipeline  # noqa: WPS433

        _PIPELINE_MODULE = _pipeline
    return _PIPELINE_MODULE


def get_data_run(
    name: str,
    cq: int = 0,
    start_issue: Optional[int] = None,
    end_issue: Optional[int] = None,
) -> None:
    """Download historical data for specified lottery。"""

    ensure_runtime_directories()
    code = name.lower().strip()
    if code not in LOTTERY_CONFIGS:
        raise ValueError(f"Unsupported lottery type: {name}")
    use_sequence = bool(cq) and code == "kl8"
    download_history(code, start=start_issue, end=end_issue, use_sequence_order=use_sequence)


def get_current_number(name: str) -> str:
    """Return current issue for specified lottery。"""

    code = name.lower().strip()
    if code not in LOTTERY_CONFIGS:
        raise ValueError(f"Unsupported lottery type: {name}")
    return get_current_issue(code)


def train_pipeline(
    name: str,
    window_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    red_epochs: Optional[int] = None,
    blue_epochs: Optional[int] = None,
) -> "TrainingSummary":
    """High-level training interface, wraps pipeline.train_lottery_models。"""

    code = name.lower().strip()
    logger.info("Starting training【{}】model...", LOTTERY_CONFIGS[code].name)
    pipeline_module = _load_pipeline()
    summary = pipeline_module.train_lottery_models(
        code=code,
        window_size=window_size,
        batch_size=batch_size,
        red_epochs=red_epochs,
        blue_epochs=blue_epochs,
    )
    logger.success("Training completed: {}", summary)
    return summary


def predict_latest(name: str, window_size: Optional[int] = None) -> Dict[str, list]:
    """Use latest model to predict next issue numbers。"""

    code = name.lower().strip()
    cfg = get_lottery_config(code)
    pipeline_module = _load_pipeline()
    predictions = pipeline_module.predict_next_draw(code=code, window_size=window_size)
    readable = {key: list(map(int, value.tolist())) for key, value in predictions.items()}
    logger.info("【{}】Prediction result: {}", cfg.name, readable)
    return readable


__all__ = [
    "get_data_run",
    "get_current_number",
    "train_pipeline",
    "predict_latest",
    "download_history",
    "load_history",
]
