"""PyAV decode loops shared by the file and URL sources."""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import av
import numpy as np

from sports_watcher.ingest.base import AudioChunk, Frame

AUDIO_RATE = 16000


def open_container(
    target: str,
    options: dict[str, str] | None = None,
    timeout: tuple[float, float] | None = None,
) -> av.container.InputContainer:
    """``timeout`` is PyAV's ``(open_s, read_s)``; set it for network sources so a stalled
    read returns instead of blocking a thread forever."""
    kwargs: dict = {"mode": "r", "options": options or {}}
    if timeout is not None:
        kwargs["timeout"] = timeout
    return av.open(target, **kwargs)


def decode_frames(
    container: av.container.InputContainer,
    start_wall: datetime,
    realtime: bool = False,
    live: bool = False,
) -> Iterator[Frame]:
    """Yield BGR frames with stream pts.

    ``wall`` is the arrival time for live sources (``live=True``), so it never runs
    backwards when the stream's pts jump; for files it is ``start_wall`` plus elapsed
    stream time, which keeps offline runs deterministic. With ``realtime`` the loop
    sleeps so file frames are released no faster than their pts.
    """
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    first_pts: float | None = None
    t0 = time.monotonic()
    index = 0
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        pts_s = float(frame.pts * stream.time_base)
        if first_pts is None:
            first_pts = pts_s
        elapsed = pts_s - first_pts
        if realtime:
            ahead = elapsed - (time.monotonic() - t0)
            if ahead > 0:
                time.sleep(ahead)
        wall = datetime.now(UTC) if live else start_wall + timedelta(seconds=elapsed)
        yield Frame(index=index, pts_s=pts_s, wall=wall, image=frame.to_ndarray(format="bgr24"))
        index += 1


def decode_audio(container: av.container.InputContainer) -> Iterator[AudioChunk]:
    """Yield 16 kHz mono int16 chunks from the first audio stream, if any."""
    if not container.streams.audio:
        return
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="s16", layout="mono", rate=AUDIO_RATE)

    def emit(frames: list[av.AudioFrame]) -> Iterator[AudioChunk]:
        for rf in frames:
            pcm = rf.to_ndarray().reshape(-1).astype(np.int16, copy=False)
            pts_s = float(rf.pts * rf.time_base) if rf.pts is not None and rf.time_base else 0.0
            yield AudioChunk(pts_s=pts_s, pcm=pcm, sample_rate=AUDIO_RATE)

    for frame in container.decode(stream):
        yield from emit(resampler.resample(frame))
    yield from emit(resampler.resample(None))
