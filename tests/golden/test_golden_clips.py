"""Scores the watcher against hand-labelled clips in data/golden. Skips when none exist."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "data" / "golden"
MIN_PRECISION = 0.9
MIN_RECALL = 0.9


def _clips() -> list[Path]:
    if not GOLDEN_DIR.is_dir():
        return []
    return sorted(p for p in GOLDEN_DIR.glob("*.mp4") if p.with_suffix(".truth.jsonl").exists())


_CLIPS = _clips()
_PARAMS = [pytest.param(c, id=c.stem) for c in _CLIPS] or [
    pytest.param(None, id="none", marks=pytest.mark.skip(reason="no golden clips in data/golden"))
]


@pytest.mark.parametrize("clip", _PARAMS)
def test_golden_clip_score_changes(clip: Path) -> None:
    pytest.importorskip("rapidocr_onnxruntime")
    from sports_watcher.config import load_league
    from sports_watcher.eval.score import score
    from sports_watcher.ingest.base import SourceSpec
    from sports_watcher.replay.truth import load_truth
    from sports_watcher.watcher import GameWatcher

    meta_path = clip.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    watcher = GameWatcher(
        SourceSpec(kind="file", path=str(clip)),
        load_league(meta.get("league", "pba")),
        meta.get("broadcaster", "one_sports"),
        game_id=clip.stem,
        home=meta.get("home", "home"),
        away=meta.get("away", "away"),
        ocr_every_n_frames=meta.get("ocr_every_n_frames", 5),
    )
    watcher.run()
    assert watcher.status().state == "ended", watcher.status()
    report = score(watcher.events, load_truth(clip.with_suffix(".truth.jsonl")), tolerance_s=1.0)
    s = report.per_type.get("score_change")
    assert s is not None, report.model_dump()
    assert s.recall >= MIN_RECALL and s.precision >= MIN_PRECISION, report.model_dump()
