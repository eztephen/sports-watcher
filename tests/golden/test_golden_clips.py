"""Scores the watcher against hand-labelled clips in data/golden. Skips when none exist.

Each clip's ``.meta.json`` names the event types to check and their floors under
``checks`` (default: score_change at 0.9). Vision types are checked only when a perceiver
is available: weights in models/ or a perception cache for the clip in data/cache.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from sports_watcher.events.model import EventStatus

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "data" / "golden"
DEFAULT_CHECKS = {"score_change": 0.9}
VISION_TYPES = {"possession_start", "shot_attempt", "shot_made", "shot_missed", "rebound", "turnover"}
MADE_WINDOW_S = 3.0


def _clips() -> list[Path]:
    if not GOLDEN_DIR.is_dir():
        return []
    return sorted(p for p in GOLDEN_DIR.glob("*.mp4") if p.with_suffix(".truth.jsonl").exists())


_CLIPS = _clips()
_PARAMS = [pytest.param(c, id=c.stem) for c in _CLIPS] or [
    pytest.param(None, id="none", marks=pytest.mark.skip(reason="no golden clips in data/golden"))
]


def _meta(clip: Path) -> dict:
    meta_path = clip.with_suffix(".meta.json")
    return json.loads(meta_path.read_text()) if meta_path.exists() else {}


def _run(clip: Path, meta: dict, **kw):
    from sports_watcher.config import Settings, load_league
    from sports_watcher.ingest.base import SourceSpec
    from sports_watcher.perception.perceiver import build_perceiver
    from sports_watcher.watcher import GameWatcher

    league = load_league(meta.get("league", "pba"))
    perceiver = build_perceiver(Settings(), league, clip, meta.get("home_color"), meta.get("away_color"))
    watcher = GameWatcher(
        SourceSpec(kind="file", path=str(clip)),
        league,
        meta.get("broadcaster", "one_sports"),
        game_id=clip.stem,
        home=meta.get("home", "home"),
        away=meta.get("away", "away"),
        ocr_every_n_frames=meta.get("ocr_every_n_frames", 5),
        perceiver=perceiver,
        **kw,
    )
    watcher.run()
    assert watcher.status().state == "ended", watcher.status()
    return watcher


@pytest.mark.parametrize("clip", _PARAMS)
def test_golden_clip_events(clip: Path) -> None:
    pytest.importorskip("rapidocr_onnxruntime")
    from sports_watcher.eval.score import score
    from sports_watcher.replay.truth import load_truth

    meta = _meta(clip)
    checks: dict[str, float] = meta.get("checks", DEFAULT_CHECKS)
    watcher = _run(clip, meta)
    report = score(watcher.events, load_truth(clip.with_suffix(".truth.jsonl")), tolerance_s=1.0)
    skipped = []
    for event_type, floor in checks.items():
        if event_type in VISION_TYPES and not watcher.vision_enabled:
            skipped.append(event_type)
            continue
        s = report.per_type.get(event_type)
        assert s is not None, (event_type, report.model_dump())
        assert s.recall >= floor and s.precision >= floor, (event_type, report.model_dump())
    if watcher.vision_enabled:
        confirmed = [e for e in watcher.events if e.status == EventStatus.confirmed]
        # The score change follows the release by up to the broadcaster's scoreboard lag.
        window = max(MADE_WINDOW_S, watcher.broadcaster_config.score_lag_s)
        for made in (e for e in confirmed if e.type == "shot_made"):
            assert any(
                e.type == "score_change" and e.team == made.team and abs(e.t_pts - made.t_pts) <= window
                for e in confirmed
            ), f"shot_made at {made.t_pts} has no score_change within {window} s"
    if skipped and len(skipped) == len(checks):
        pytest.skip(f"no perception cache or weights: {sorted(skipped)} not checked")
    if skipped:
        warnings.warn(f"{clip.stem}: no perception cache or weights, {sorted(skipped)} not checked", stacklevel=1)
