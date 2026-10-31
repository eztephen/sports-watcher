"""Test-only frame sources."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import numpy as np

from sports_watcher.clock import LatencyClass
from sports_watcher.ingest.base import AudioChunk, Frame


class StubSource:
    """Yields ``n`` black frames at 25 fps and no audio."""

    latency_class = LatencyClass.FILE
    name = "stub"

    def __init__(self, n: int = 10, size: tuple[int, int] = (64, 96), fps: int = 25) -> None:
        self.n, self.size, self.fps = n, size, fps
        self.opened = False
        self.closed = False
        self._start = datetime(2026, 1, 1, tzinfo=UTC)

    def open(self) -> None:
        self.opened = True

    def frames(self) -> Iterator[Frame]:
        h, w = self.size
        for i in range(self.n):
            pts = i / self.fps
            yield Frame(
                index=i,
                pts_s=pts,
                wall=self._start + timedelta(seconds=pts),
                image=np.zeros((h, w, 3), np.uint8),
            )

    def audio(self) -> Iterator[AudioChunk]:
        return iter(())

    def close(self) -> None:
        self.closed = True


class FailingStubSource(StubSource):
    """Raises ``OSError`` from ``frames()`` after ``fail_after`` frames."""

    def __init__(self, fail_after: int = 10, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fail_after = fail_after

    def frames(self) -> Iterator[Frame]:
        for i, frame in enumerate(super().frames()):
            if i >= self.fail_after:
                raise OSError("connection reset")
            yield frame
