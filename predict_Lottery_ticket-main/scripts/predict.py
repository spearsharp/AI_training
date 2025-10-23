# -*- coding: utf-8 -*-
"""
预测脚本，基于最新训练好的model输出下一Issue码。

Example：
    python scripts/predict.py --name ssq --window-size 5 --save
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import predict_latest  # noqa: E402
from src.config import LOTTERY_CONFIGS, PATHS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用最新model预测彩票开奖号码")
    parser.add_argument("--name", default=None, help="彩票类型代码，如 ssq / dlt / kl8 （必需）")
    parser.add_argument("--list-models", action="store_true", help="列出已训练的modelwindow并Exit")
    parser.add_argument("--window-size", type=int, default=None, help="使用指定window size的model进行预测")
    parser.add_argument("--save", action="store_true", help="是否将Prediction result保存到 predict/<code>/ 目录")
    return parser.parse_args()


def save_prediction(code: str, data: dict) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = PATHS["predict"] / code
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"prediction_{timestamp}.json"
    path.write_text(
        json.dumps(
            {"code": code, "timestamp": timestamp, "prediction": data},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    args = parse_args()
    if args.list_models:
        # List available model windows for each lottery type or for provided name
        target = args.name.lower().strip() if args.name else None
        for code, cfg in LOTTERY_CONFIGS.items():
            if target and code != target:
                continue
            model_dir = PATHS["model"] / code
            if not model_dir.exists():
                print(f"{code}: (no models)")
                continue
            windows = sorted([p.name for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("window_")])
            print(f"{code}: {', '.join(windows) if windows else '(no models)'}")
        return

    if not args.name:
        parser = argparse.ArgumentParser()
        parser.print_help()
        raise SystemExit("参数 --name 必需。Example：python scripts/predict.py --name ssq")

    code = args.name.lower().strip()
    if code not in LOTTERY_CONFIGS:
        raise SystemExit(f"Unsupported lottery type：{args.name}，valid options：{', '.join(LOTTERY_CONFIGS.keys())}")

    # If window_size not provided, pick the latest window_* folder under model/<code>/
    window_size = args.window_size
    if window_size is None:
        model_dir = PATHS["model"] / code
        if model_dir.exists():
            windows = [p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("window_")]
            if windows:
                latest = sorted(windows, key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1)[-1]
                try:
                    window_size = int(latest.name.split("_")[-1])
                except Exception:
                    window_size = None
    predictions = predict_latest(code, window_size=args.window_size)
    logger.info("Prediction result: {}", predictions)
    if args.save:
        file_path = save_prediction(code, predictions)
        logger.success("Prediction result已保存到 {}", file_path)


if __name__ == "__main__":
    main()

