"""Where events go: memory, a JSONL file, a queue, or every websocket subscriber."""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Protocol

from sports_watcher.events.model import Event


class EventSink(Protocol):
    def emit(self, event: Event) -> None: ...

    def close(self) -> None: ...


class MemorySink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)

    def close(self) -> None:
        return None


class QueueSink:
    def __init__(self, q: queue.Queue) -> None:
        self._q = q

    def emit(self, event: Event) -> None:
        self._q.put(event)

    def close(self) -> None:
        return None


class JsonlSink:
    """Appends one JSON line per event; flushes on every write so a crash loses nothing."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")

    def emit(self, event: Event) -> None:
        self._fh.write(event.model_dump_json() + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class Broadcaster:
    """Fan-out to subscriber queues. A full subscriber drops the event rather than
    blocking the pipeline."""

    def __init__(self, maxsize: int = 1000) -> None:
        self._maxsize = maxsize
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=self._maxsize)
        with self._lock:
            self._subs.append(q)
        return q

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def emit(self, event: Event) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

    def close(self) -> None:
        with self._lock:
            self._subs.clear()
