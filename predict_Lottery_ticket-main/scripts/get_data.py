# -*- coding:utf-8 -*-
"""
Historical lottery data download script.

Example:
    python scripts/get_data.py --name ssq --start 25092 --end 25121
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
    _std_logger = logging.getLogger("get_data")

    class _SimpleLogger:
        """A tiny logger shim that exposes a subset of loguru's API used in this project.

        It supports .info, .success, .warning and .error with basic str.format-style
        interpolation so scripts can run without the external `loguru` dependency.
        """

        def _fmt(self, msg: str, *args):
            try:
                return msg.format(*args) if args else msg
            except Exception:
                # If formatting fails, fall back to original message
                return msg

        def info(self, msg: str, *args, **kwargs):
            _std_logger.info(self._fmt(msg, *args))

        def success(self, msg: str, *args, **kwargs):
            # map success to info level
            _std_logger.info(self._fmt(msg, *args))

        def warning(self, msg: str, *args, **kwargs):
            _std_logger.warning(self._fmt(msg, *args))

        def error(self, msg: str, *args, **kwargs):
            _std_logger.error(self._fmt(msg, *args))

    logger = _SimpleLogger()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import get_data_run  # noqa: E402
from src.config import LOTTERY_CONFIGS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download historical lottery draw data from 500.com")
    parser.add_argument(
        "--name",
        default="ssq",
        help="Lottery type code, such as ssq / dlt / kl8, default is ssq",
    )
    parser.add_argument("--start", type=int, default=None, help="Starting issue number (inclusive), defaults to earliest available")
    parser.add_argument("--end", type=int, default=None, help="Ending issue number (inclusive), defaults to latest")
    parser.add_argument(
        "--sequence",
        action="store_true",
        help="Whether to use draw sequence data for KL8 (only valid for kl8)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = args.name.lower().strip()
    if code not in LOTTERY_CONFIGS:
        raise SystemExit(f"Unsupported lottery type: {args.name}, valid options: {', '.join(LOTTERY_CONFIGS.keys())}")
    get_data_run(code, cq=int(args.sequence), start_issue=args.start, end_issue=args.end)
    logger.success("Data download completed: {}", code)


if __name__ == "__main__":
    main()

