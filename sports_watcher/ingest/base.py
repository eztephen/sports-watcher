"""Shared ingest types: Frame, AudioChunk, SourceSpec and the FrameSource protocol."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel

from sports_watcher.clock import LatencyClass


@dataclass(slots=True)
class Frame:
    index: int
    pts_s: float
    wall: datetime
    image: np.ndarray  # BGR uint8, HxWx3


@dataclass(slots=True)
class AudioChunk:
    pts_s: float
    pcm: np.ndarray  # int16 mono
    sample_rate: int = 16000


class SourceSpec(BaseModel):
    kind: Literal["file", "url", "youtube", "stub"]
    path: str | None = None
    url: str | None = None
    realtime: bool = False


class FrameSource(Protocol):
    latency_class: LatencyClass
    name: str

    def open(self) -> None: ...

    def frames(self) -> Iterator[Frame]: ...

    def audio(self) -> Iterator[AudioChunk]: ...

    def close(self) -> None: ...
