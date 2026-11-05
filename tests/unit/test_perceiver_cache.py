from datetime import UTC, datetime

import numpy as np

from sports_watcher.ingest.base import Frame
from sports_watcher.perception.cache import CachedPerceiver, cache_path, find_cache
from sports_watcher.perception.perceiver import ScriptedPerceiver
from sports_watcher.perception.types import Track, VisionFrame, empty_vision


def _frame(i):
    return Frame(
        index=i, pts_s=i / 30, wall=datetime(2026, 1, 1, tzinfo=UTC), image=np.zeros((4, 4, 3), np.uint8)
    )


def _vf(i):
    track = Track(1, (0.0, 0.0, 10.0, 20.0), "player", "home", None, 0.9)
    return VisionFrame(i, i / 30, "main", 0.9, [track], None, None, None)


def test_scripted_perceiver_returns_frame_by_index_or_empty():
    p = ScriptedPerceiver([_vf(0), _vf(2)])
    assert p.perceive(_frame(2)).tracks[0].track_id == 1
    assert p.perceive(_frame(1)) == empty_vision(1, 1 / 30)
    assert p.calls == [2, 1]


def test_cache_path_and_find(tmp_path):
    clip = tmp_path / "clip.mp4"
    p = cache_path(tmp_path, clip, "abc123abc123")
    assert p.name == "clip.abc123abc123.jsonl"
    assert find_cache(tmp_path, clip) is None
    p.write_text('{"version": "abc123abc123", "clip": "clip"}\n')
    assert find_cache(tmp_path, clip) == p


def test_cached_perceiver_writes_through_then_hits(tmp_path):
    inner = ScriptedPerceiver([_vf(0), _vf(1)], version="v1")
    path = cache_path(tmp_path, tmp_path / "c.mp4", inner.version)
    c = CachedPerceiver(inner, path)
    assert c.perceive(_frame(0)).tracks[0].team == "home" and c.misses == 1
    assert c.perceive(_frame(0)).tracks[0].team == "home" and c.hits == 1 and inner.calls == [0]
    reopened = CachedPerceiver(None, path)
    assert reopened.version == "v1" and reopened.perceive(_frame(0)) == _vf(0)


def test_build_perceiver_without_models_or_cache_is_none(tmp_path):
    from sports_watcher.config import Settings
    from sports_watcher.perception.perceiver import build_perceiver
    from tests.helpers.leagues import fixture_league

    s = Settings(_env_file=None, models_dir=tmp_path / "m", cache_dir=tmp_path / "c")
    assert build_perceiver(s, fixture_league(), tmp_path / "clip.mp4", None, None) is None


def test_build_perceiver_uses_cache_without_models(tmp_path):
    from sports_watcher.config import Settings
    from sports_watcher.perception.perceiver import build_perceiver
    from tests.helpers.leagues import fixture_league

    s = Settings(_env_file=None, models_dir=tmp_path / "m", cache_dir=tmp_path / "c")
    clip = tmp_path / "clip.mp4"
    version = "0123456789ab"
    CachedPerceiver(ScriptedPerceiver([_vf(0)], version), cache_path(s.cache_dir, clip, version)).perceive(_frame(0))
    p = build_perceiver(s, fixture_league(), clip, None, None)
    assert isinstance(p, CachedPerceiver) and p.version == version


def test_find_cache_matches_the_stem_and_header_exactly(tmp_path):
    clip = tmp_path / "game.mp4"
    (tmp_path / "game.v2.abcdefabcdef.jsonl").write_text('{"version": "abcdefabcdef", "clip": "game.v2"}\n')
    (tmp_path / "game.abcdefabcdef.jsonl").write_text('{"version": "abcdefabcdef", "clip": "other"}\n')
    assert find_cache(tmp_path, clip) is None
    good = tmp_path / "game.0123456789ab.jsonl"
    good.write_text('{"version": "0123456789ab", "clip": "game"}\n')
    assert find_cache(tmp_path, clip) == good


def test_partial_cache_with_an_inner_perceiver_is_rebuilt(tmp_path):
    path = tmp_path / "c.v1.jsonl"
    CachedPerceiver(ScriptedPerceiver([_vf(0), _vf(2)], "v1"), path).perceive(_frame(0))  # no completion marker
    inner = ScriptedPerceiver([_vf(0), _vf(2)], "v1")
    rebuilt = CachedPerceiver(inner, path)
    rebuilt.perceive(_frame(0))
    assert inner.calls == [0] and rebuilt.misses == 1


def test_complete_cache_is_reused_by_an_inner_perceiver(tmp_path):
    from sports_watcher.ingest.base import SourceSpec
    from sports_watcher.ingest.factory import register_source
    from sports_watcher.perception.offline import build_cache
    from tests.helpers.sources import StubSource

    register_source("stub", lambda spec: StubSource(n=4, fps=30))
    path = tmp_path / "c.v1.jsonl"
    build_cache(SourceSpec(kind="stub"), ScriptedPerceiver([_vf(i) for i in range(4)], "v1"), path)
    inner = ScriptedPerceiver([], "v1")
    reopened = CachedPerceiver(inner, path)
    assert reopened.perceive(_frame(1)).tracks and inner.calls == [] and reopened.complete


def test_cache_only_miss_degrades_to_empty_vision(tmp_path):
    path = tmp_path / "c.v1.jsonl"
    CachedPerceiver(ScriptedPerceiver([_vf(0)], "v1"), path).perceive(_frame(0))
    assert CachedPerceiver(None, path).perceive(_frame(5)) == empty_vision(5, 5 / 30)
