"""Build and open a FrameSource from a SourceSpec. Tests register extra kinds."""

from __future__ import annotations

from collections.abc import Callable

from sports_watcher.ingest.base import FrameSource, SourceSpec
from sports_watcher.ingest.file import FileSource
from sports_watcher.ingest.url import UrlSource
from sports_watcher.ingest.youtube import YouTubeSource

_REGISTRY: dict[str, Callable[[SourceSpec], FrameSource]] = {}


def register_source(kind: str, factory: Callable[[SourceSpec], FrameSource]) -> None:
    _REGISTRY[kind] = factory


def _file(spec: SourceSpec) -> FrameSource:
    if not spec.path:
        raise ValueError("file source needs 'path'")
    return FileSource(spec.path, realtime=spec.realtime)


def _url(spec: SourceSpec) -> FrameSource:
    if not spec.url:
        raise ValueError("url source needs 'url'")
    return UrlSource(spec.url)


def _youtube(spec: SourceSpec) -> FrameSource:
    if not spec.url:
        raise ValueError("youtube source needs 'url'")
    return YouTubeSource(spec.url)


register_source("file", _file)
register_source("url", _url)
register_source("youtube", _youtube)


def open_source(spec: SourceSpec) -> FrameSource:
    try:
        factory = _REGISTRY[spec.kind]
    except KeyError as exc:
        raise ValueError(f"unknown source kind: {spec.kind}") from exc
    src = factory(spec)
    src.open()
    return src
