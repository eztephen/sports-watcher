"""Maps stream presentation timestamps to wall-clock time and estimates feed latency."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum


class LatencyClass(StrEnum):
    """How far behind the venue a feed of this kind usually runs."""

    FILE = "file"
    PUSH = "push"
    PULL = "pull"
    HLS = "hls"
    CAPTURE = "capture"


BEHIND_LIVE_ESTIMATE_S: dict[LatencyClass, float | None] = {
    LatencyClass.FILE: None,
    LatencyClass.PUSH: 0.5,
    LatencyClass.PULL: 1.0,
    LatencyClass.HLS: 15.0,
    LatencyClass.CAPTURE: 0.2,
}

# A backwards jump larger than this is treated as a discontinuity, not jitter.
DISCONTINUITY_S = 1.0


class StreamClock:
    """Anchors stream pts to wall time; re-anchors on pts discontinuities so wall time
    for later frames never runs backwards."""

    def __init__(self, latency_class: LatencyClass) -> None:
        self.latency_class = latency_class
        self._anchor_pts: float | None = None
        self._anchor_wall: datetime | None = None
        self._last_pts: float | None = None

    def observe(self, pts_s: float, wall: datetime) -> None:
        if self._anchor_pts is None or (
            self._last_pts is not None and pts_s < self._last_pts - DISCONTINUITY_S
        ):
            self._anchor_pts, self._anchor_wall = pts_s, wall
        self._last_pts = pts_s

    def wall_for(self, pts_s: float) -> datetime:
        if self._anchor_pts is None or self._anchor_wall is None:
            raise RuntimeError("StreamClock.wall_for called before any observe()")
        return self._anchor_wall + timedelta(seconds=pts_s - self._anchor_pts)

    def behind_live_s(self) -> float | None:
        return BEHIND_LIVE_ESTIMATE_S[self.latency_class]
