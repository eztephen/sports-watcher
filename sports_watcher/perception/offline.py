"""Build the perception cache for a clip once, so later runs iterate in seconds."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sports_watcher.ingest.base import SourceSpec
from sports_watcher.ingest.factory import open_source
from sports_watcher.perception.cache import CachedPerceiver
from sports_watcher.perception.perceiver import Perceiver
from sports_watcher.watcher import PERCEPTION_HZ


@dataclass
class CacheStats:
    frames_read: int
    frames_perceived: int
    seconds: float


def build_cache(
    spec: SourceSpec,
    perceiver: Perceiver,
    path: Path,
    every_n: int | None = None,
    progress: Callable[[int], None] | None = None,
) -> CacheStats:
    """Perceive every ``every_n``-th frame of the source into the cache at ``path``.
    ``every_n`` defaults to the watcher's cadence (about 30 Hz)."""
    cached = CachedPerceiver(perceiver, Path(path))
    source = open_source(spec)
    t0 = time.perf_counter()
    read = perceived = 0
    first_pts: float | None = None
    try:
        for frame in source.frames():
            read += 1
            if every_n is None:
                if first_pts is None:
                    first_pts = frame.pts_s
                    n = 1
                else:
                    dt = frame.pts_s - first_pts
                    every_n = max(1, round((1.0 / dt) / PERCEPTION_HZ)) if dt > 0 else 1
                    n = every_n
            else:
                n = every_n
            if (read - 1) % n == 0:
                cached.perceive(frame)
                perceived += 1
                if progress is not None:
                    progress(perceived)
        cached.mark_complete()
    finally:
        source.close()
    return CacheStats(read, perceived, time.perf_counter() - t0)
