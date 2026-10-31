"""Turns observations into events and runs the provisional → confirmed/retracted
lifecycle. One engine per game."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from sports_watcher.clock import StreamClock
from sports_watcher.events.fuser import PendingEvent, resolve_pending
from sports_watcher.events.model import Event, EventStatus, Evidence, SourceInfo
from sports_watcher.events.sinks import EventSink
from sports_watcher.sports.base import EventDraft, Observation, SportPlugin
from sports_watcher.state.game_state import GameState

RETRACT_REASONS = {"retracted": "scoreboard_disagrees", "ttl": "ttl"}


class EventEngine:
    def __init__(
        self,
        plugin: SportPlugin,
        game_id: str,
        source: SourceInfo,
        clock: StreamClock,
        ttl_s: float = 4.0,
        sinks: Sequence[EventSink] = (),
    ) -> None:
        self.plugin = plugin
        self.game_id = game_id
        self.source = source
        self.clock = clock
        self.ttl_s = ttl_s
        self.sinks = list(sinks)
        self.state = GameState()
        self.seq = 0
        self._pending: list[PendingEvent] = []
        self._finished = False

    # -- public API -----------------------------------------------------------------

    def step(self, obs: Observation) -> list[Event]:
        """Advance one observation. Returns the events emitted during this step."""
        self.clock.observe(obs.t_pts, obs.t_wall)
        prev = self.state
        new = self.plugin.update_state(prev, obs)
        self.state = new

        emitted: list[Event] = []
        resolved, self._pending = resolve_pending(self._pending, new, obs, obs.t_pts)
        for pending, resolution in resolved:
            emitted.append(self._terminal(pending, resolution))

        for rule in self.plugin.rules():
            for draft in rule(prev, new, obs):
                emitted.append(self._provisional(draft, obs))
        return emitted

    def finish(self) -> list[Event]:
        """Retract everything still pending and close the sinks."""
        if self._finished:
            return []
        self._finished = True
        out = [self._terminal(p, "stream_ended") for p in self._pending]
        self._pending = []
        for sink in self.sinks:
            sink.close()
        return out

    def emit_direct(
        self,
        type: str,
        team: str,
        evidence: Evidence,
        t_pts: float,
        t_wall: datetime,
        confidence: float = 1.0,
    ) -> Event:
        """Emit an already-confirmed event that needs no provisional stage, such as a
        ``stream_gap`` the watcher observed directly."""
        event = Event(
            game_id=self.game_id,
            seq=self._next_seq(),
            type=type,
            status=EventStatus.confirmed,
            t_pts=t_pts,
            t_wall=t_wall,
            emitted_wall=datetime.now(UTC),
            behind_live_s=self.clock.behind_live_s(),
            period=self.state.period,
            game_clock_s=self.state.clock_s,
            team=team,  # type: ignore[arg-type]
            evidence=evidence,
            confidence=confidence,
            source=self.source,
        )
        self._emit(event)
        return event

    # -- internals --------------------------------------------------------------------

    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def _provisional(self, draft: EventDraft, obs: Observation) -> Event:
        event = Event(
            game_id=self.game_id,
            seq=self._next_seq(),
            type=draft.type,
            status=EventStatus.provisional,
            t_pts=draft.t_pts,
            t_wall=self.clock.wall_for(draft.t_pts),
            emitted_wall=datetime.now(UTC),
            behind_live_s=self.clock.behind_live_s(),
            period=self.state.period,
            game_clock_s=self.state.clock_s,
            shot_clock_s=self.state.shot_clock_s,
            team=draft.team,  # type: ignore[arg-type]
            player=draft.player,
            court_xy=draft.court_xy,
            zone=draft.zone,
            evidence=draft.evidence,
            confidence=draft.confidence,
            source=self.source,
        )
        ttl = draft.ttl_s if draft.ttl_s is not None else self.ttl_s
        self._pending.append(PendingEvent(event, draft, deadline_pts=obs.t_pts + ttl))
        self._emit(event)
        return event

    def _terminal(self, pending: PendingEvent, resolution: str) -> Event:
        now = datetime.now(UTC)
        if resolution == "confirmed":
            event = pending.event.confirmed_copy(seq=self._next_seq(), emitted_wall=now)
        else:
            reason = RETRACT_REASONS.get(resolution, resolution)
            event = pending.event.retracted_copy(
                seq=self._next_seq(), emitted_wall=now, reason=reason
            )
        self._emit(event)
        return event

    def _emit(self, event: Event) -> None:
        for sink in self.sinks:
            sink.emit(event)
