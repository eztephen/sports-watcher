"""Local file (or directory of files) source."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from sports_watcher.clock import LatencyClass
from sports_watcher.ingest._decode import decode_audio, decode_frames, open_container
from sports_watcher.ingest.base import AudioChunk, Frame

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".ts", ".m4v", ".webm"}


class FileSource:
    latency_class = LatencyClass.FILE
    name = "file"

    def __init__(self, path: str | Path, realtime: bool = False, audio: bool = True) -> None:
        self._paths = _expand(Path(path))
        self._realtime = realtime
        self._audio = audio
        self._containers: list = []
        self._start_wall = datetime.now(UTC)

    def open(self) -> None:
        if not self._paths:
            raise FileNotFoundError("no video files to open")
        self._start_wall = datetime.now(UTC)

    def frames(self) -> Iterator[Frame]:
        offset = 0
        for p in self._paths:
            container = open_container(str(p))
            self._containers.append(container)
            try:
                for frame in decode_frames(container, self._start_wall, self._realtime):
                    frame.index += offset
                    yield frame
                    last = frame.index
                offset = last + 1
            finally:
                container.close()
                self._containers.remove(container)

    def audio(self) -> Iterator[AudioChunk]:
        if not self._audio:
            return
        for p in self._paths:
            container = open_container(str(p))
            try:
                yield from decode_audio(container)
            finally:
                container.close()

    def close(self) -> None:
        for c in list(self._containers):
            c.close()
        self._containers.clear()


def _expand(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"no such video file or directory: {path}")
