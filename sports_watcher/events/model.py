"""The event record every consumer sees. Events are append-only and immutable; a
correction is a new event whose ``supersedes`` points at the one it replaces."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from sports_watcher.clock import LatencyClass

Team = Literal["home", "away", "unknown"]


class EventStatus(StrEnum):
    provisional = "provisional"
    confirmed = "confirmed"
    retracted = "retracted"


class PlayerRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    track_id: int | None = None
    name: str | None = None
    jersey: str | None = None
    confidence: float = 0.0


class CommentaryCueRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    t_pts: float
    kind: str
    text: str
    confidence: float


class Evidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    score_before: tuple[int, int] | None = None
    score_after: tuple[int, int] | None = None
    points: int | None = None
    ball_speed_mps: float | None = None
    rim_distance_m: float | None = None
    cues: list[CommentaryCueRef] = Field(default_factory=list)
    camera_state: str | None = None
    detector_confidence: float | None = None
    note: str | None = None


class SourceInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    adapter: str
    latency_class: LatencyClass


class Event(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: uuid4().hex)
    game_id: str
    seq: int
    type: str
    status: EventStatus
    supersedes: str | None = None
    reason: str | None = None
    t_pts: float
    t_wall: datetime
    emitted_wall: datetime
    behind_live_s: float | None = None
    period: int | None = None
    game_clock_s: float | None = None
    shot_clock_s: float | None = None
    team: Team = "unknown"
    player: PlayerRef | None = None
    court_xy: tuple[float, float] | None = None
    zone: str | None = None
    evidence: Evidence = Field(default_factory=Evidence)
    confidence: float = 0.0
    source: SourceInfo

    def confirmed_copy(
        self,
        seq: int,
        t_pts: float,
        t_wall: datetime,
        emitted_wall: datetime,
        confidence: float | None = None,
    ) -> Event:
        return self._successor(
            seq=seq,
            t_pts=t_pts,
            t_wall=t_wall,
            emitted_wall=emitted_wall,
            status=EventStatus.confirmed,
            reason=None,
            confidence=self.confidence if confidence is None else confidence,
        )

    def retracted_copy(
        self, seq: int, t_pts: float, t_wall: datetime, emitted_wall: datetime, reason: str
    ) -> Event:
        return self._successor(
            seq=seq,
            t_pts=t_pts,
            t_wall=t_wall,
            emitted_wall=emitted_wall,
            status=EventStatus.retracted,
            reason=reason,
            confidence=self.confidence,
        )

    def _successor(self, **changes) -> Event:
        data = self.model_dump()
        data.update(changes)
        data["id"] = uuid4().hex
        data["supersedes"] = self.id
        return Event.model_validate(data)
