"""Any URL FFmpeg can demux: HLS/DASH manifests, RTSP, RTMP, SRT, UDP MPEG-TS, or a path."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from urllib.parse import urlparse

from sports_watcher.clock import LatencyClass
from sports_watcher.ingest._decode import decode_audio, decode_frames, open_container
from sports_watcher.ingest.base import AudioChunk, Frame

OPEN_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 10.0

BASE_OPTIONS: dict[str, str] = {
    "fflags": "nobuffer",
    "flags": "low_delay",
    "probesize": "500000",
    "analyzeduration": "1000000",
}


def latency_class_for(url: str) -> LatencyClass:
    scheme = urlparse(url).scheme.lower()
    if scheme == "rtsp":
        return LatencyClass.PULL
    if scheme in {"rtmp", "rtmps", "srt", "udp"}:
        return LatencyClass.PUSH
    if scheme in {"http", "https"}:
        return LatencyClass.HLS
    return LatencyClass.FILE


def options_for(url: str) -> dict[str, str]:
    scheme = urlparse(url).scheme.lower()
    opts = dict(BASE_OPTIONS)
    if scheme == "rtsp":
        opts["rtsp_transport"] = "tcp"
    if scheme in {"http", "https"}:
        opts["live_start_index"] = "-1"
    return opts


class UrlSource:
    name = "url"

    def __init__(self, url: str, options: dict[str, str] | None = None) -> None:
        self.url = url
        self.latency_class = latency_class_for(url)
        self._options = options
        self._containers: list = []
        self._start_wall = datetime.now(UTC)

    @property
    def live(self) -> bool:
        return self.latency_class != LatencyClass.FILE

    def _open_options(self) -> dict[str, str]:
        if self._options is not None:
            return self._options
        # Local paths get no live options; libavformat would only warn about them.
        return {} if not self.live else options_for(self.url)

    def open_kwargs(self) -> dict:
        """Keyword arguments for ``open_container``: options, plus read timeouts when live."""
        kwargs: dict = {"options": self._open_options()}
        if self.live:
            kwargs["timeout"] = (OPEN_TIMEOUT_S, READ_TIMEOUT_S)
        return kwargs

    def open(self) -> None:
        self._start_wall = datetime.now(UTC)

    def frames(self) -> Iterator[Frame]:
        container = open_container(self.url, **self.open_kwargs())
        self._containers.append(container)
        try:
            yield from decode_frames(container, self._start_wall, realtime=False, live=self.live)
        finally:
            container.close()
            self._containers.remove(container)

    def audio(self) -> Iterator[AudioChunk]:
        container = open_container(self.url, **self.open_kwargs())
        self._containers.append(container)
        try:
            yield from decode_audio(container)
        finally:
            container.close()
            self._containers.remove(container)

    def close(self) -> None:
        for c in list(self._containers):
            c.close()
        self._containers.clear()
