from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ASConfig:
    """Avellaneda-Stoikov strategy configuration."""
    gamma: float
    k: float
    volatility_window: int
    horizon_seconds: float
    min_spread: float
    max_spread: float
    tick_size: float
    cancellation_threshold: float


@dataclass
class Config:
    """Backtester configuration."""
    order_book_path: Path
    trades_path: Path
    batch_size: int
    strategy_id: str
    order_volume: int
    as_config: ASConfig | None = None


def load_config(config_path: Path) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    base_dir = config_path.parent.parent

    as_config = None
    if "avellaneda_stoikov" in raw:
        as_raw = raw["avellaneda_stoikov"]
        as_config = ASConfig(
            gamma=as_raw["gamma"],
            k=as_raw["k"],
            volatility_window=as_raw["volatility_window"],
            horizon_seconds=as_raw["horizon_seconds"],
            min_spread=as_raw["min_spread"],
            max_spread=as_raw["max_spread"],
            tick_size=as_raw["tick_size"],
            cancellation_threshold=as_raw.get("cancellation_threshold", 0.0002),
        )

    return Config(
        order_book_path=base_dir / raw["data"]["order_book"],
        trades_path=base_dir / raw["data"]["trades"],
        batch_size=raw["data"]["batch_size"],
        strategy_id=raw["strategy"]["id"],
        order_volume=raw["strategy"]["order_volume"],
        as_config=as_config,
    )
