"""Builders for Event objects used across the unit tests."""

from __future__ import annotations

from datetime import UTC, datetime

from sports_watcher.clock import LatencyClass
from sports_watcher.events.model import Event, EventStatus, Evidence, SourceInfo

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
SOURCE = SourceInfo(adapter="file", latency_class=LatencyClass.FILE)


def make_event(**overrides) -> Event:
    """A valid provisional score_change for game g1; override any field."""
    base = {
        "game_id": "g1",
        "seq": 1,
        "type": "score_change",
        "status": EventStatus.provisional,
        "t_pts": 10.0,
        "t_wall": NOW,
        "emitted_wall": NOW,
        "behind_live_s": None,
        "period": 1,
        "game_clock_s": 590.0,
        "shot_clock_s": None,
        "team": "home",
        "player": None,
        "court_xy": None,
        "zone": None,
        "evidence": Evidence(score_before=(0, 0), score_after=(2, 0), points=2),
        "confidence": 0.9,
        "source": SOURCE,
    }
    base.update(overrides)
    return Event(**base)
