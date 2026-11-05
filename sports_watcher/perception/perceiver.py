"""The perceiver contract: one frame in, one VisionFrame out.

Three implementations share it: the ONNX models (``onnx_perceiver.py``), the per-clip
cache (``cache.py``) and the scripted one below for tests.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from sports_watcher.config import LeagueConfig, Settings
from sports_watcher.ingest.base import Frame
from sports_watcher.perception.types import VisionFrame, empty_vision

log = logging.getLogger(__name__)


class Perceiver(Protocol):
    version: str  # hashes weights and settings; keys the cache

    def perceive(self, frame: Frame) -> VisionFrame: ...

    def reset(self) -> None: ...  # after a reconnect or seek


class ScriptedPerceiver:
    """Returns the scripted frame whose ``frame_index`` matches, else an empty one."""

    def __init__(self, frames: Sequence[VisionFrame], version: str = "scripted") -> None:
        self._frames = {vf.frame_index: vf for vf in frames}
        self.version = version
        self.calls: list[int] = []

    def perceive(self, frame: Frame) -> VisionFrame:
        self.calls.append(frame.index)
        vf = self._frames.get(frame.index)
        return vf if vf is not None else empty_vision(frame.index, frame.pts_s)

    def reset(self) -> None:
        return None


def build_perceiver(
    settings: Settings,
    league: LeagueConfig,
    clip: Path | None,
    home_color: str | None,
    away_color: str | None,
    use_cache: bool = True,
) -> Perceiver | None:
    """The perceiver a watcher should run with, or ``None`` for scoreboard-only mode.

    With weights in ``settings.models_dir`` the ONNX perceiver runs, wrapped in the
    per-clip cache for file sources. Without weights, an existing cache for the clip is
    used on its own. Otherwise there is no vision and the watcher behaves as M0.
    """
    from sports_watcher.perception.cache import CachedPerceiver, cache_path, find_cache
    from sports_watcher.perception.models import ModelSet

    models = ModelSet.load(Path(settings.models_dir), verify=True)
    if models is not None:
        from sports_watcher.perception.onnx_perceiver import OnnxPerceiver

        perceiver: Perceiver = OnnxPerceiver(models, league, home_color, away_color)
        if clip is not None and use_cache:
            return CachedPerceiver(perceiver, cache_path(Path(settings.cache_dir), clip, perceiver.version))
        return perceiver
    if clip is not None and use_cache:
        cached = find_cache(Path(settings.cache_dir), clip)
        if cached is not None:
            log.info("no vision models in %s; using the perception cache %s", settings.models_dir, cached)
            return CachedPerceiver(None, cached)
    log.info("no vision models in %s: scoreboard-only mode", settings.models_dir)
    return None
