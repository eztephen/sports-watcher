"""League configs for pipeline tests: detection-path broadcasters, independent of the
production calibration in configs/leagues."""

from __future__ import annotations

from pathlib import Path

from sports_watcher.config import LeagueConfig, Settings, load_league

TEST_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "configs"


def fixture_league(name: str = "pba") -> LeagueConfig:
    return load_league(name, TEST_CONFIGS_DIR)


def fixture_settings() -> Settings:
    """Settings for pipeline tests: the fixture configs, and model and cache directories
    that never exist, so whatever weights sit under models/ cannot leak into the tests."""
    empty = TEST_CONFIGS_DIR.parent / "no_models"
    return Settings(_env_file=None, configs_dir=TEST_CONFIGS_DIR, models_dir=empty, cache_dir=empty / "cache")
