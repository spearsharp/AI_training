# -*- coding: utf-8 -*-
"""
Project global configuration module。

This module is responsible for：
1. Reading `config/config.yaml` to get runtime configuration；
2. Define model hyperparameters and default training settings for lottery games；
3. Provide path constants and utility functions for reuse by data, models, and scripts.

Author: Codex Upgrade
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_FILE = BASE_DIR / "config" / "config.yaml"


@dataclass(frozen=True)
class SequenceModelSpec:
    """Describes structural parameters of a single sequence model。"""

    sequence_len: int
    num_classes: int
    embedding_dim: int
    hidden_units: Iterable[int]
    dropout: float = 0.2


@dataclass(frozen=True)
class LotteryModelConfig:
    """Describes required configuration for training a single lottery game。"""

    code: str
    name: str
    red: SequenceModelSpec
    blue: Optional[SequenceModelSpec] = None
    default_window: int = 3
    default_batch_size: int = 32
    default_red_epochs: int = 40
    default_blue_epochs: int = 30
    learning_rate: float = 5e-4
    allow_sequence_order: bool = False


def _load_yaml_config() -> Dict[str, object]:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"System configuration file not found: {CONFIG_FILE}")
    with CONFIG_FILE.open(encoding="utf-8") as fp:
        return yaml.safe_load(fp)


YAML_CONFIG: Dict[str, object] = _load_yaml_config()

PATHS = {
    "data": Path(YAML_CONFIG.get("paths", {}).get("data", BASE_DIR / "data")).resolve(),
    "model": Path(YAML_CONFIG.get("paths", {}).get("model", BASE_DIR / "model")).resolve(),
    "predict": Path(YAML_CONFIG.get("paths", {}).get("predict", BASE_DIR / "predict")).resolve(),
    "logs": Path(YAML_CONFIG.get("paths", {}).get("logs", BASE_DIR / "logs")).resolve(),
}

NETWORK_CONFIG = {
    "timeout": YAML_CONFIG.get("network", {}).get("timeout", 20),
    "retry_count": YAML_CONFIG.get("network", {}).get("retry_count", 3),
    "backoff_factor": YAML_CONFIG.get("network", {}).get("backoff_factor", 0.6),
    "user_agent": YAML_CONFIG.get("network", {}).get(
        "user_agent",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36",
    ),
}


ALLOWED_DOMAINS = {"datachart.500.com", "data.917500.cn", "datachart.500.com"}    # lottery data sources

DATA_FILE_NAME = "data.csv"
MODEL_METADATA_FILE = "metadata.json"


LOTTERY_CONFIGS: Dict[str, LotteryModelConfig] = {
    "ssq": LotteryModelConfig(
        code="ssq",
        name="Double Color Ball",
        red=SequenceModelSpec(
            sequence_len=6,
            num_classes=33,
            embedding_dim=64,
            hidden_units=(128, 64),
            dropout=0.3,
        ),
        blue=SequenceModelSpec(
            sequence_len=1,
            num_classes=16,
            embedding_dim=32,
            hidden_units=(64,),
            dropout=0.2,
        ),
        default_window=5,
        default_batch_size=32,
        default_red_epochs=60,
        default_blue_epochs=30,
        learning_rate=8e-4,
    ),
    "dlt": LotteryModelConfig(
        code="dlt",
        name="Super Lotto",
        red=SequenceModelSpec(
            sequence_len=5,
            num_classes=35,
            embedding_dim=64,
            hidden_units=(128, 64),
            dropout=0.3,
        ),
        blue=SequenceModelSpec(
            sequence_len=2,
            num_classes=12,
            embedding_dim=32,
            hidden_units=(64,),
            dropout=0.2,
        ),
        default_window=5,
        default_batch_size=32,
        default_red_epochs=60,
        default_blue_epochs=30,
        learning_rate=8e-4,
    ),
    "pls": LotteryModelConfig(
        code="pls",
        name="P3",
        red=SequenceModelSpec(
            sequence_len=3,
            num_classes=10,
            embedding_dim=32,
            hidden_units=(64, 32),
            dropout=0.2,
        ),
        blue=None,
        default_window=5,
        default_batch_size=32,
        default_red_epochs=50,
        default_blue_epochs=0,
        learning_rate=5e-4,
    ),
    "qxc": LotteryModelConfig(
        code="qxc",
        name="7-Star Lottery",
        red=SequenceModelSpec(
            sequence_len=7,
            num_classes=10,
            embedding_dim=48,
            hidden_units=(96, 48),
            dropout=0.25,
        ),
        blue=None,
        default_window=6,
        default_batch_size=32,
        default_red_epochs=60,
        default_blue_epochs=0,
        learning_rate=6e-4,
    ),
    "kl8": LotteryModelConfig(
        code="kl8",
        name="Happy 8",
        red=SequenceModelSpec(
            sequence_len=20,
            num_classes=80,
            embedding_dim=48,
            hidden_units=(128, 128, 64),
            dropout=0.35,
        ),
        blue=None,
        default_window=6,
        default_batch_size=48,
        default_red_epochs=40,
        default_blue_epochs=0,
        learning_rate=5e-4,
        allow_sequence_order=True,
    ),
    "sd": LotteryModelConfig(
        code="sd",
        name="3D Lottery",
        red=SequenceModelSpec(
            sequence_len=3,
            num_classes=10,
            embedding_dim=32,
            hidden_units=(64, 32),
            dropout=0.2,
        ),
        blue=None,
        default_window=5,
        default_batch_size=32,
        default_red_epochs=50,
        default_blue_epochs=0,
        learning_rate=4e-4,
    ),
}


def ensure_runtime_directories() -> None:
    """Ensure required directories for project execution exist。"""

    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)


def get_lottery_config(code: str) -> LotteryModelConfig:
    """Get configuration by game code。"""

    normalized = code.lower().strip()
    if normalized not in LOTTERY_CONFIGS:
        raise ValueError(f"Unknown lottery type: {code}")
    return LOTTERY_CONFIGS[normalized]


name_path = {
    code: {
        "name": cfg.name,
        "path": f"{(PATHS['data'] / code).as_posix()}/",
    }
    for code, cfg in LOTTERY_CONFIGS.items()
}
predict_path = f"{PATHS['predict'].as_posix()}/"
data_file_name = DATA_FILE_NAME


__all__ = [
    "BASE_DIR",
    "CONFIG_FILE",
    "DATA_FILE_NAME",
    "data_file_name",
    "MODEL_METADATA_FILE",
    "PATHS",
    "LOTTERY_CONFIGS",
    "LotteryModelConfig",
    "SequenceModelSpec",
    "NETWORK_CONFIG",
    "ALLOWED_DOMAINS",
    "ensure_runtime_directories",
    "get_lottery_config",
    "name_path",
    "predict_path",
]
