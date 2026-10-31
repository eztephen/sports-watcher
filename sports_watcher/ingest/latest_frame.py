"""Keep only the newest frame so inference never falls behind a live feed.

Adapted from GodsEye's LatestFrameReader: a daemon thread drains the frame iterator and
retains the most recent frame; ``read`` hands out that frame and reports how many were
skipped since the previous call.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

from sports_watcher.ingest.base import Frame


class LatestFrameReader:
    def __init__(self, frames: Iterator[Frame]) -> None:
        self._frames = frames
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._seq = 0  # frames pulled from the iterator
        self._consumed_seq = 0  # seq of the last frame handed out
        self._stopped = False
        self._exhausted = False
        self.error: BaseException | None = None
        self.dropped_total = 0
        self._thread = threading.Thread(target=self._run, daemon=True, name="latest-frame-reader")

    def start(self) -> LatestFrameReader:
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            for frame in self._frames:
                with self._cond:
                    if self._stopped:
                        return
                    self._frame = frame
                    self._seq += 1
                    self._cond.notify_all()
        except BaseException as exc:  # noqa: BLE001 - surfaced to the consumer via .error
            self.error = exc
        finally:
            with self._cond:
                self._exhausted = True
                self._cond.notify_all()

    def read(self, timeout: float = 5.0) -> tuple[bool, Frame | None, int]:
        """Block until a frame newer than the last one returned exists.

        Returns ``(ok, frame, dropped)``. ``ok`` is False when the reader was stopped, the
        source ended or failed with nothing pending, or no new frame arrived in ``timeout``.
        """
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self._stopped and not self._exhausted and self._seq <= self._consumed_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None, 0
                self._cond.wait(remaining)
            if not self._stopped and self._frame is not None and self._seq > self._consumed_seq:
                dropped = self._seq - self._consumed_seq - 1
                self._consumed_seq = self._seq
                self.dropped_total += dropped
                return True, self._frame, dropped
            return False, None, 0

    @property
    def finished(self) -> bool:
        with self._cond:
            return self._exhausted and self._seq <= self._consumed_seq

    def stop(self, join_timeout: float = 2.0) -> None:
        """Stop handing out frames and wait up to ``join_timeout`` for the producer thread
        to leave the source, so the caller can close the source safely afterwards."""
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=join_timeout)

    @property
    def producer_alive(self) -> bool:
        return self._thread.is_alive()
