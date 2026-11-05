"""The perceiver contract: one frame in, one VisionFrame out.

Three implementations share it: the ONNX models (``onnx_perceiver.py``), the per-clip
cache (``cache.py``) and the scripted one below for tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from sports_watcher.ingest.base import Frame
from sports_watcher.perception.types import VisionFrame, empty_vision


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
