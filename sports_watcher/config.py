"""Settings and league configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide settings, read from the environment with the ``SW_`` prefix."""

    model_config = SettingsConfigDict(env_prefix="SW_", env_file=".env", extra="ignore")

    device: str = "cpu"
    configs_dir: Path = Path("configs")
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    cache_dir: Path = Path("data/cache")
    log_level: str = "INFO"


class Roi(BaseModel):
    """A rectangle expressed as fractions of the frame width and height."""

    x: float
    y: float
    w: float
    h: float


class FieldRois(BaseModel):
    """One box per scoreboard field, as fractions of the full frame. When set, each box
    is read with recognition only, which is what large single digits need."""

    home: Roi
    away: Roi
    clock: Roi | None = None
    period: Roi | None = None
    shot_clock: Roi | None = None


class BroadcasterConfig(BaseModel):
    scoreboard_roi: Roi
    home_side: Literal["left", "right"] = "left"
    shot_clock_visible: bool = True
    fields: FieldRois | None = None
    # The longest the score on the bug trails a shot's release. Shot, rebound and turnover
    # rules wait this long for the score change before they give up on a made shot.
    score_lag_s: float = 3.0


class LeagueConfig(BaseModel):
    name: str
    court_length_m: float
    court_width_m: float
    three_point_radius_m: float
    three_point_corner_m: float
    basket_from_baseline_m: float
    free_throw_line_from_baseline_m: float
    key_width_m: float
    period_length_s: int
    periods: int
    overtime_length_s: int
    shot_clock_s: int
    asr_languages: list[str]
    broadcasters: dict[str, BroadcasterConfig]


def load_league(name: str, configs_dir: Path | None = None) -> LeagueConfig:
    """Load ``<configs_dir>/leagues/<name>.yaml`` into a validated LeagueConfig."""
    base = configs_dir if configs_dir is not None else Settings().configs_dir
    path = Path(base) / "leagues" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"league config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return LeagueConfig.model_validate(data)
