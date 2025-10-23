# -*- coding: utf-8 -*-
"""
Data fetching module for retrieving lottery historical data from 500.com and saving locally.

Features:
1. Uses requests.Session with retry logic to meet network security requirements;
2. Outputs Pandas DataFrame for preprocessing and training;
3. Provides both sequence and regular download modes for KL8 lottery.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
try:  # pragma: no cover - optional dependency
    from loguru import logger
except Exception:  # pragma: no cover - provide fallback when loguru is not installed
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _std_logger = logging.getLogger("data_fetcher")

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
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from .config import (
    ALLOWED_DOMAINS,
    DATA_FILE_NAME,
    LOTTERY_CONFIGS,
    NETWORK_CONFIG,
    PATHS,
    LotteryModelConfig,
    ensure_runtime_directories,
)


@dataclass
class DownloadResult:
    """Metadata for a download operation."""

    code: str
    total_issues: int
    saved_path: str
    timestamp: str


class LotteryHttpClient:
    """Encapsulates network access logic, providing GET method with retry and domain validation."""

    def __init__(
        self,
        timeout: float,
        retries: int,
        backoff_factor: float,
        user_agent: str,
    ) -> None:
        self._timeout = timeout
        self._session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }

    def get_text(self, url: str) -> str:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if all(allowed not in domain for allowed in ALLOWED_DOMAINS):
            raise ValueError(f"Access to domain not allowed: {domain}")
        response = self._session.get(url, headers=self._headers, timeout=self._timeout)
        response.raise_for_status()
        response.encoding = "utf-8"
        return response.text


def _build_history_url(config: LotteryModelConfig, start: Optional[int], end: Optional[int]) -> str:
    base = f"https://datachart.500.com/{config.code}/history/"
    if config.code in {"qxc", "pls", "sd"}:
        path = "inc/history.php"
    elif config.code == "kl8":
        path = "newinc/jbzs_redblue.php"
    else:
        path = "history.shtml"

    if path.endswith(".shtml"):
        return f"{base}{path}"

    start_issue = start or 1
    end_issue = end or 999999
    limit = end_issue - start_issue + 1
    query = f"{path}?start={start_issue}&end={end_issue}&limit={limit}"
    return f"{base}{query}"


def _parse_issue_list(config: LotteryModelConfig, html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    rows = []
    if config.code in {"ssq", "dlt", "kl8"}:
        tbody = soup.find("tbody", attrs={"id": "tdata"})
        if not tbody:
            raise ValueError("Draw number data table not found (id=tdata)")
        trs = tbody.find_all("tr")
    else:
        table = soup.find("table", id="tablelist")
        if not table:
            raise ValueError("Draw number data table not found (id=tablelist)")
        trs = table.find_all("tr")

    for tr in trs:
        tds = tr.find_all("td")
        if not tds:
            continue
        issue = tds[0].get_text(strip=True)
        if not issue or issue == "Issue":
            continue
        record = {"Issue": issue}
        if config.code == "ssq":
            for idx in range(config.red.sequence_len):
                record[f"Red_{idx + 1}"] = tds[idx + 1].get_text(strip=True)
            record["Blue_1"] = tds[7].get_text(strip=True)
        elif config.code == "dlt":
            for idx in range(config.red.sequence_len):
                record[f"Red_{idx + 1}"] = tds[idx + 1].get_text(strip=True)
            for idx in range(config.blue.sequence_len):
                record[f"Blue_{idx + 1}"] = tds[6 + idx].get_text(strip=True)
        elif config.code in {"pls", "sd", "qxc"}:
            digits = tds[1].get_text(strip=True).split(" ")
            for idx, value in enumerate(digits):
                record[f"Red_{idx + 1}"] = value
        elif config.code == "kl8":
            numbers = [td.get_text(strip=True) for td in tds if td.get_text(strip=True).isdigit()]
            for idx, value in enumerate(numbers):
                record[f"Red_{idx + 1}"] = value
        rows.append(record)

    if not rows:
        raise ValueError("Failed to parse draw numbers, no data retrieved")
    df = pd.DataFrame(rows)
    df.sort_values("Issue", ascending=False, inplace=True)
    return df.reset_index(drop=True)




def get_current_issue(code: str, client: Optional[LotteryHttpClient] = None) -> str:
    """Get the latest issue number for a specified lottery。"""

    cfg = LOTTERY_CONFIGS[code]
    client = client or LotteryHttpClient(
        timeout=NETWORK_CONFIG["timeout"],
        retries=NETWORK_CONFIG["retry_count"],
        backoff_factor=NETWORK_CONFIG.get("backoff_factor", 0.6),
        user_agent=NETWORK_CONFIG["user_agent"],
    )

    if cfg.code in {"qxc", "pls", "sd"}:
        url = f"https://datachart.500.com/{cfg.code}/history/inc/history.php"
    elif cfg.code == "kl8":
        url = f"https://datachart.500.com/{cfg.code}/history/newinc/jbzs_redblue.php"
    else:
        url = f"https://datachart.500.com/{cfg.code}/history/history.shtml"

    html = client.get_text(url)
    soup = BeautifulSoup(html, "lxml")
    if cfg.code == "kl8":
        value = soup.find("div", class_="wrap_datachart").find("input", {"id": "to"})["value"]
    else:
        value = soup.find("div", class_="wrap_datachart").find("input", {"id": "end"})["value"]
    logger.info("【{}】Latest issue number: {}", cfg.name, value)
    return value


def download_history(
    code: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    use_sequence_order: bool = False,
    client: Optional[LotteryHttpClient] = None,
) -> DownloadResult:
    """Download historical data and save to data/<code>/data.csv。"""

    ensure_runtime_directories()
    cfg = LOTTERY_CONFIGS[code]
    client = client or LotteryHttpClient(
        timeout=NETWORK_CONFIG["timeout"],
        retries=NETWORK_CONFIG["retry_count"],
        backoff_factor=NETWORK_CONFIG.get("backoff_factor", 0.6),
        user_agent=NETWORK_CONFIG["user_agent"],
    )

    if cfg.code == "kl8" and use_sequence_order:
        raise NotImplementedError("KL8-related features have been migrated to independent project：https://github.com/KittenCN/kl8-lottery-analyzer")
    else:
        url = _build_history_url(cfg, start, end)
        logger.info("Downloading【{}】historical data: {}", cfg.name, url)
        html = client.get_text(url)
        df = _parse_issue_list(cfg, html)

    save_dir = PATHS["data"] / cfg.code
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / DATA_FILE_NAME
    df.to_csv(output_path, index=False, encoding="utf-8")
    meta = DownloadResult(
        code=cfg.code,
        total_issues=len(df),
        saved_path=str(output_path),
        timestamp=datetime.utcnow().isoformat(),
    )
    logger.success("Data download completed，total {} issues, saved to {}", meta.total_issues, output_path)
    (output_path.parent / "download_meta.json").write_text(
        json.dumps(meta.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def load_history(code: str) -> pd.DataFrame:
    """Load locally downloaded historical data。"""

    cfg = LOTTERY_CONFIGS[code]
    path = PATHS["data"] / cfg.code / DATA_FILE_NAME
    if not path.exists():
        raise FileNotFoundError(f"Historical data for {cfg.name} not found, please download first: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    if "Issue" not in df.columns:
        raise ValueError(f"{path} Missing【Issue】field, data is corrupted or format is abnormal")
    return df


__all__ = [
    "DownloadResult",
    "LotteryHttpClient",
    "download_history",
    "get_current_issue",
    "load_history",
]

