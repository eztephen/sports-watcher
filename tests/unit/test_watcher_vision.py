from sports_watcher.ingest.base import SourceSpec
from sports_watcher.ingest.factory import register_source
from sports_watcher.perception.perceiver import ScriptedPerceiver
from sports_watcher.perception.types import VisionFrame
from sports_watcher.watcher import GameWatcher
from tests.helpers.leagues import fixture_league
from tests.helpers.sources import StubSource
from tests.helpers.text_readers import SeqTextReader


def _watcher(perceiver, n=60, fps=60, **kw):
    register_source("stub", lambda spec: StubSource(n=n, fps=fps))
    return GameWatcher(
        SourceSpec(kind="stub"), fixture_league(), "one_sports", "g", "H", "A",
        text_reader=SeqTextReader([[]]), perceiver=perceiver, **kw,
    )


def test_perception_runs_on_every_second_frame_at_sixty_fps():
    p = ScriptedPerceiver([])
    w = _watcher(p)
    w.run()
    assert p.calls == list(range(0, 60, 2))


def test_perception_runs_on_every_frame_at_thirty_fps():
    p = ScriptedPerceiver([])
    w = _watcher(p, n=30, fps=30)
    w.run()
    assert p.calls == list(range(30))


def test_camera_state_in_status_and_log():
    frames = [
        VisionFrame(i, i / 60, "graphics" if i >= 20 else "main", 0.9, [], None, None, None)
        for i in range(0, 60, 2)
    ]
    w = _watcher(ScriptedPerceiver(frames), log_camera_states=True)
    w.run()
    assert w.status().camera_state == "graphics" and w.vision_enabled is True
    assert w.camera_log[0] == (0.0, "main") and w.camera_log[-1][1] == "graphics"
    assert any(s == "main" for t, s in w.camera_log if 20 / 60 <= t < 0.8)  # hysteresis


def test_without_perceiver_behaviour_is_m0():
    w = _watcher(None)
    w.run()
    assert w.status().camera_state == "main" and w.vision_enabled is False and w.camera_log == []
