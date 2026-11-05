from sports_watcher.ingest.base import SourceSpec
from sports_watcher.ingest.factory import register_source
from sports_watcher.perception.cache import CachedPerceiver
from sports_watcher.perception.offline import build_cache
from sports_watcher.perception.perceiver import ScriptedPerceiver
from tests.helpers.sources import StubSource


def test_build_cache_perceives_every_second_frame_at_sixty_fps(tmp_path):
    register_source("stub", lambda spec: StubSource(n=20, fps=60))
    p = ScriptedPerceiver([], version="v9")
    stats = build_cache(SourceSpec(kind="stub"), p, tmp_path / "c.v9.jsonl")
    assert (stats.frames_read, stats.frames_perceived) == (20, 10) and p.calls == list(range(0, 20, 2))
    assert CachedPerceiver(None, tmp_path / "c.v9.jsonl").version == "v9"
