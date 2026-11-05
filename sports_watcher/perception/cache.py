"""Per-clip perception cache: one JSONL per clip and perceiver version.

The first line is a header ``{"version": ..., "clip": ...}``; every other line is a
``VisionFrame`` as written by ``vision_to_dict``. Perception runs once per clip; the rules,
engine, firsts and evaluation then iterate in seconds.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from sports_watcher.ingest.base import Frame
from sports_watcher.perception.perceiver import Perceiver
from sports_watcher.perception.types import (
    VisionFrame,
    empty_vision,
    vision_from_dict,
    vision_to_dict,
)

log = logging.getLogger(__name__)
VERSION_RE = re.compile(r"^[0-9a-f]{12}$")


class CacheMiss(KeyError):
    """A frame is not in the cache and there is no inner perceiver to compute it."""


def cache_path(cache_dir: Path, clip: Path, version: str) -> Path:
    return Path(cache_dir) / f"{Path(clip).stem}.{version}.jsonl"


def _header(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            first = fh.readline().strip()
    except OSError:
        return None
    if not first:
        return None
    try:
        d = json.loads(first)
    except json.JSONDecodeError:
        return None
    return d if "frame_index" not in d else None


def find_cache(cache_dir: Path, clip: Path) -> Path | None:
    """The newest complete-or-partial cache whose name is exactly ``<stem>.<12 hex>.jsonl``
    and whose header names this clip."""
    root = Path(cache_dir)
    stem = Path(clip).stem
    if not root.is_dir():
        return None
    matches = []
    for p in root.glob(f"{stem}.*.jsonl"):
        version = p.name[len(stem) + 1 : -len(".jsonl")]
        if not VERSION_RE.match(version):
            continue
        header = _header(p)
        if header is None or header.get("clip") != stem:
            continue
        matches.append(p)
    matches.sort(key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


class CachedPerceiver:
    """A partial cache (no completion marker) is only trusted on its own: with an inner
    perceiver it is rebuilt from the start, since the inner tracker's ids would not line
    up with the cached frames."""

    def __init__(self, inner: Perceiver | None, path: Path) -> None:
        self._inner = inner
        self._path = Path(path)
        self._frames: dict[int, VisionFrame] = {}
        self._header_version: str | None = None
        self.complete = False
        self.hits = 0
        self.misses = 0
        self._warned_miss = False
        self._load()
        if inner is not None and self._frames and not self.complete:
            log.warning("perception cache %s is incomplete: rebuilding it", self._path)
            self._frames = {}
            self._path.write_text("", encoding="utf-8")

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
            if not self._warned_miss:
                log.warning("perception cache %s lacks frame %d: no vision for such frames", self._path, frame.index)
                self._warned_miss = True
            self.misses += 1
            return empty_vision(frame.index, frame.pts_s)
        self.misses += 1
        vf = self._inner.perceive(frame)
        self._append(vf)
        return vf

    def reset(self) -> None:
        if self._inner is not None:
            self._inner.reset()

    def mark_complete(self) -> None:
        """Append the completion marker: every perception frame of the clip is in."""
        if self.complete:
            return
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"complete": True, "frames": len(self._frames)}) + "\n")
        self.complete = True

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
                    if d.get("complete"):
                        self.complete = True
                    elif "version" in d:
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
