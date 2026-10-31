"""Runs one game: source → frames → scoreboard → engine → sinks, with reconnects."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

from sports_watcher.clock import LatencyClass, StreamClock
from sports_watcher.config import BroadcasterConfig, LeagueConfig
from sports_watcher.events.engine import EventEngine
from sports_watcher.events.firsts import Firsts, fold_firsts
from sports_watcher.events.model import Event, Evidence, SourceInfo
from sports_watcher.events.sinks import Broadcaster, EventSink, MemorySink
from sports_watcher.ingest.base import Frame, FrameSource, SourceSpec
from sports_watcher.ingest.factory import open_source
from sports_watcher.ingest.latest_frame import LatestFrameReader
from sports_watcher.perception.scoreboard import ScoreboardReader, TextReader
from sports_watcher.sports.base import Observation
from sports_watcher.sports.basketball.plugin import BasketballPlugin

WatcherState = Literal["starting", "running", "ended", "failed"]
RECONNECTABLE_KINDS = {"url", "youtube"}


class WatcherStatus(BaseModel):
    state: WatcherState
    frames_read: int
    frames_dropped: int
    frames_processed: int
    reconnects: int
    t_pts: float
    period: int | None
    clock_s: float | None
    score: tuple[int, int]
    events_emitted: int
    error: str | None


class GameWatcher:
    def __init__(
        self,
        spec: SourceSpec,
        league: LeagueConfig,
        broadcaster: str,
        game_id: str,
        home: str,
        away: str,
        sinks: Sequence[EventSink] = (),
        text_reader: TextReader | None = None,
        broadcaster_config: BroadcasterConfig | None = None,
        ocr_every_n_frames: int = 5,
        ttl_s: float = 4.0,
        max_reconnects: int = 5,
        vote_window: int = 5,
        reconnect_backoff_s: float = 1.0,
        stall_timeout_s: float = 30.0,
    ) -> None:
        self.spec = spec
        self.league = league
        self.game_id = game_id
        self.home = home
        self.away = away
        self.broadcaster_config = broadcaster_config or league.broadcasters[broadcaster]
        self.ocr_every_n_frames = max(1, ocr_every_n_frames)
        self.ttl_s = ttl_s
        self.max_reconnects = max_reconnects
        self.vote_window = vote_window
        self.reconnect_backoff_s = reconnect_backoff_s
        self.stall_timeout_s = stall_timeout_s

        self._memory = MemorySink()
        self.broadcaster = Broadcaster()
        self._sinks: list[EventSink] = [self._memory, self.broadcaster, *sinks]
        self._text_reader = text_reader
        self._scoreboard: ScoreboardReader | None = None
        self._plugin = BasketballPlugin(league)
        self._engine: EventEngine | None = None

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._state: WatcherState = "starting"
        self._error: str | None = None
        self._frames_processed = 0
        self._frames_dropped = 0
        self._reconnects = 0
        self._last_t_pts = 0.0
        self._last_wall = datetime.now(UTC)

    # -- lifecycle ----------------------------------------------------------------------

    def start_in_thread(self) -> threading.Thread:
        thread = threading.Thread(target=self.run, daemon=True, name=f"watcher-{self.game_id}")
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        backoff = self.reconnect_backoff_s
        try:
            while not self._stop.is_set():
                error = self._run_once()
                if error is None or self._stop.is_set():
                    break
                if not self._can_reconnect() or self._reconnects >= self.max_reconnects:
                    raise error
                self._reconnects += 1
                self._note_gap(error)
                time.sleep(backoff)
                backoff = min(backoff * 2, self.reconnect_backoff_s * 8)
            self._finish("ended")
        except Exception as exc:  # noqa: BLE001 - every failure ends the job with a reason
            self._error = f"{type(exc).__name__}: {exc}"
            self._finish("failed")

    def _run_once(self) -> BaseException | None:
        """Open the source and consume it. Returns the error that ended it, if any."""
        try:
            source = open_source(self.spec)
        except Exception as exc:  # noqa: BLE001
            return exc
        try:
            self._ensure_engine(source)
            self._set_state("running")
            for frame, dropped in self._frame_stream(source):
                if self._stop.is_set():
                    break
                self._process(frame, dropped)
            return None
        except Exception as exc:  # noqa: BLE001
            return exc
        finally:
            source.close()

    def _finish(self, state: WatcherState) -> None:
        if self._engine is not None:
            self._engine.finish()
        self._set_state(state)

    # -- per-frame work -----------------------------------------------------------------

    def _frame_stream(self, source: FrameSource) -> Iterator[tuple[Frame, int]]:
        """File sources yield every frame so offline runs are deterministic; live sources
        go through the latest-frame reader and may drop frames."""
        if source.latency_class == LatencyClass.FILE:
            for frame in source.frames():
                yield frame, 0
            return
        reader = LatestFrameReader(source.frames()).start()
        last_frame_at = time.monotonic()
        try:
            while not self._stop.is_set():
                ok, frame, dropped = reader.read(timeout=1.0)
                if ok and frame is not None:
                    last_frame_at = time.monotonic()
                    yield frame, dropped
                    continue
                if reader.error is not None:
                    raise reader.error
                if reader.finished:
                    return
                if time.monotonic() - last_frame_at > self.stall_timeout_s:
                    raise OSError(f"stream stalled for {self.stall_timeout_s:.0f}s")
        finally:
            reader.stop()

    def _process(self, frame: Frame, dropped: int) -> None:
        assert self._engine is not None
        self._frames_dropped += dropped
        self._frames_processed += 1
        scoreboard = None
        if (self._frames_processed - 1) % self.ocr_every_n_frames == 0:
            scoreboard = self._scoreboard_reader().read(frame.image)
        obs = Observation(frame_index=frame.index, t_pts=frame.pts_s, t_wall=frame.wall, scoreboard=scoreboard)
        self._engine.step(obs)
        self._last_t_pts, self._last_wall = frame.pts_s, frame.wall

    def _note_gap(self, error: BaseException) -> None:
        if self._engine is None:
            return
        self._engine.emit_direct(
            "stream_gap",
            team="unknown",
            evidence=Evidence(note=f"{type(error).__name__}: {error}"),
            t_pts=self._last_t_pts,
            t_wall=datetime.now(UTC),
        )

    # -- helpers --------------------------------------------------------------------------

    def _can_reconnect(self) -> bool:
        return self.spec.kind in RECONNECTABLE_KINDS or (self.spec.kind == "stub" and bool(self.spec.url))

    def _ensure_engine(self, source: FrameSource) -> None:
        if self._engine is None:
            info = SourceInfo(adapter=source.name, latency_class=source.latency_class)
            self._engine = EventEngine(
                self._plugin, self.game_id, info, StreamClock(source.latency_class),
                ttl_s=self.ttl_s, sinks=self._sinks,
            )

    def _scoreboard_reader(self) -> ScoreboardReader:
        if self._scoreboard is None:
            reader = self._text_reader
            if reader is None:
                from sports_watcher.perception.scoreboard import RapidOcrTextReader

                reader = RapidOcrTextReader()
            self._scoreboard = ScoreboardReader(reader, self.broadcaster_config, self.vote_window)
        return self._scoreboard

    def _set_state(self, state: WatcherState) -> None:
        with self._lock:
            self._state = state

    # -- read side ------------------------------------------------------------------------

    @property
    def events(self) -> list[Event]:
        return list(self._memory.events)

    def events_since(self, seq: int) -> list[Event]:
        return [e for e in self._memory.events if e.seq > seq]

    def firsts(self) -> Firsts:
        return fold_firsts(self._memory.events, self.league)

    def status(self) -> WatcherStatus:
        state = self._engine.state if self._engine is not None else None
        with self._lock:
            current = self._state
        return WatcherStatus(
            state=current,
            frames_read=self._frames_processed + self._frames_dropped,
            frames_dropped=self._frames_dropped,
            frames_processed=self._frames_processed,
            reconnects=self._reconnects,
            t_pts=self._last_t_pts,
            period=state.period if state else None,
            clock_s=state.clock_s if state else None,
            score=state.score() if state else (0, 0),
            events_emitted=len(self._memory.events),
            error=self._error,
        )
