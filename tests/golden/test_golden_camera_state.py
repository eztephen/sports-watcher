"""Camera-state accuracy against hand-labelled segments, and no vision events off the
main camera. Skips without clips, segments or a perceiver."""

from __future__ import annotations

from pathlib import Path

import pytest

from sports_watcher.events.model import EventStatus
from tests.golden.test_golden_clips import GOLDEN_DIR, VISION_TYPES, _meta, _run

MIN_CAMERA_ACCURACY = 0.9
OFF_MAIN = {"closeup", "replay", "graphics"}


def _clips() -> list[Path]:
    if not GOLDEN_DIR.is_dir():
        return []
    return sorted(p for p in GOLDEN_DIR.glob("*.mp4") if (p.parent / f"{p.stem}.camera.jsonl").exists())


_CLIPS = _clips()
_PARAMS = [pytest.param(c, id=c.stem) for c in _CLIPS] or [
    pytest.param(None, id="none", marks=pytest.mark.skip(reason="no clips with camera segments"))
]


@pytest.mark.parametrize("clip", _PARAMS)
def test_golden_camera_state(clip: Path) -> None:
    pytest.importorskip("rapidocr_onnxruntime")
    from sports_watcher.eval.score import camera_state_accuracy
    from sports_watcher.replay.truth import load_camera_truth

    meta = _meta(clip)
    watcher = _run(clip, meta, log_camera_states=True)
    if not watcher.vision_enabled:
        pytest.skip("no perception cache or weights: camera state not checked")
    segments = load_camera_truth(clip.parent / f"{clip.stem}.camera.jsonl")
    accuracy = camera_state_accuracy(watcher.camera_log, segments)
    assert accuracy is not None and accuracy >= MIN_CAMERA_ACCURACY, accuracy
    off_main = [s for s in segments if s.state in OFF_MAIN]
    for e in watcher.events:
        if e.status != EventStatus.confirmed or e.type not in VISION_TYPES:
            continue
        assert not any(s.start <= e.t_pts < s.end for s in off_main), f"{e.type} at {e.t_pts} inside a {OFF_MAIN} segment"
