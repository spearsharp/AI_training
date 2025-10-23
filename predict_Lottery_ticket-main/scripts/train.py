# -*- coding: utf-8 -*-
"""
Model training script(TensorFlow 2.15+ version)。

Example：
    python scripts/train.py --name ssq --window-size 5 --red-epochs 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:  # pragma: no cover - optional dependency
    from loguru import logger
except Exception:  # pragma: no cover - provide fallback when loguru is not installed
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _std_logger = logging.getLogger("train")

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

# Ensure the project root is on sys.path so `src` imports work and the
# bootstrap shim can be imported early.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# import as early as possible; src.bootstrap is best-effort
try:
    import src.bootstrap  # noqa: F401
except Exception:
    # If bootstrap fails, continue; the bootstrap shim is non-critical
    pass

from src.common import get_data_run, train_pipeline  # noqa: E402
from src.config import LOTTERY_CONFIGS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM model for specified lottery")
    parser.add_argument(
        "--name",
        default="ssq",
        help="Lottery type code, such as ssq / dlt / kl8, default is ssq",
    )
    parser.add_argument("--window-size", type=int, default=None, help="Time window size, defaults to game configuration")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size, defaults to game configuration")
    parser.add_argument("--red-epochs", type=int, default=None, help="Red model training epochs")
    parser.add_argument("--blue-epochs", type=int, default=None, help="Blue model training epochs")
    parser.add_argument("--download-data", action="store_true", help="Automatically download latest data before training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = args.name.lower().strip()
    if code not in LOTTERY_CONFIGS:
        raise SystemExit(f"Unsupported lottery type：{args.name}，valid options：{', '.join(LOTTERY_CONFIGS.keys())}")

    if args.download_data:
        logger.info("Starting to download data...")
        get_data_run(code)

    summary = train_pipeline(
        name=code,
        window_size=args.window_size,
        batch_size=args.batch_size,
        red_epochs=args.red_epochs,
        blue_epochs=args.blue_epochs,
    )
    logger.success("Training completed, see details in model/{}/window_{}/{}", summary.code, summary.window_size, "metadata.json")


if __name__ == "__main__":
    main()

