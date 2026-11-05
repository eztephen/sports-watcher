"""Per-clip perception cache: one JSONL per clip and perceiver version.

The first line is a header ``{"version": ..., "clip": ...}``; every other line is a
``VisionFrame`` as written by ``vision_to_dict``. Perception runs once per clip; the rules,
engine, firsts and evaluation then iterate in seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

from sports_watcher.ingest.base import Frame
from sports_watcher.perception.perceiver import Perceiver
from sports_watcher.perception.types import VisionFrame, vision_from_dict, vision_to_dict


class CacheMiss(KeyError):
    """A frame is not in the cache and there is no inner perceiver to compute it."""


def cache_path(cache_dir: Path, clip: Path, version: str) -> Path:
    return Path(cache_dir) / f"{Path(clip).stem}.{version}.jsonl"


def find_cache(cache_dir: Path, clip: Path) -> Path | None:
    """The newest cache for ``clip`` under ``cache_dir``, whatever its version."""
    root = Path(cache_dir)
    if not root.is_dir():
        return None
    matches = sorted(root.glob(f"{Path(clip).stem}.*.jsonl"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


class CachedPerceiver:
    def __init__(self, inner: Perceiver | None, path: Path) -> None:
        self._inner = inner
        self._path = Path(path)
        self._frames: dict[int, VisionFrame] = {}
        self._header_version: str | None = None
        self.hits = 0
        self.misses = 0
        self._load()

    @property
    def version(self) -> str:
        if self._inner is not None:
            return self._inner.version
        return self._header_version or "cache"

    def perceive(self, frame: Frame) -> VisionFrame:
        vf = self._frames.get(frame.index)
        if vf is not None:
            self.hits += 1
            return vf
        if self._inner is None:
            raise CacheMiss(frame.index)
        self.misses += 1
        vf = self._inner.perceive(frame)
        self._append(vf)
        return vf

    def reset(self) -> None:
        if self._inner is not None:
            self._inner.reset()

    # -- file ---------------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.is_file():
            return
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if "frame_index" not in d:
                    self._header_version = d.get("version")
                    continue
                vf = vision_from_dict(d)
                self._frames[vf.frame_index] = vf

    def _append(self, vf: VisionFrame) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self._path.exists() or self._path.stat().st_size == 0
        with self._path.open("a", encoding="utf-8") as fh:
            if new_file:
                fh.write(json.dumps({"version": self.version, "clip": self._path.stem.split(".")[0]}) + "\n")
                self._header_version = self.version
            fh.write(json.dumps(vision_to_dict(vf)) + "\n")
            fh.flush()
        self._frames[vf.frame_index] = vf
