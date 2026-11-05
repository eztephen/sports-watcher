"""The contract between the engine and a sport plug-in."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from sports_watcher.config import LeagueConfig
from sports_watcher.events.model import Evidence, PlayerRef
from sports_watcher.perception.scoreboard import ScoreboardState
from sports_watcher.perception.types import VisionFrame
from sports_watcher.state.game_state import GameState

# "pending" | "confirmed" | "retracted" | "retracted:<reason>" (the engine records the reason)
Verdict = str


@dataclass
class Observation:
    frame_index: int
    t_pts: float
    t_wall: datetime
    scoreboard: ScoreboardState | None = None
    camera_state: str = "main"
    vision: VisionFrame | None = None


@dataclass
class EventDraft:
    type: str
    team: str
    t_pts: float
    evidence: Evidence
    confidence: float
    confirmer: Confirmer
    ttl_s: float = 4.0
    player: PlayerRef | None = None
    court_xy: tuple[float, float] | None = None
    zone: str | None = None
    extra: dict = field(default_factory=dict)


Confirmer = Callable[[EventDraft, GameState, Observation], Verdict]
Rule = Callable[[GameState, GameState, Observation], list[EventDraft]]


class SportPlugin(Protocol):
    name: str
    league: LeagueConfig

    def update_state(self, state: GameState, obs: Observation) -> GameState: ...

    def rules(self) -> list[Rule]: ...
