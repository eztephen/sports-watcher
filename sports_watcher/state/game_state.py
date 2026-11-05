"""The game state the rules read. Sport-agnostic; sport plug-ins fill it in."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from sports_watcher.perception.types import BallObs, Box, Team, Track

ClockState = Literal["read", "assumed", "unknown"]
Side = Literal["left", "right"]


@dataclass
class Holder:
    track_id: int
    team: Team
    since_pts: float


@dataclass
class Release:
    t_pts: float
    track_id: int
    team: Team
    image_xy: tuple[float, float]
    feet_xy: tuple[float, float] | None
    zone: str


@dataclass
class RimInteraction:
    entered_pts: float | None = None
    left_pts: float | None = None
    crossed_pts: float | None = None
    above_pts: float | None = None  # last time the ball was above the rim inside its span


@dataclass
class ShotRecord:
    t_pts: float
    track_id: int
    team: Team
    zone: str
    court_xy: tuple[float, float] | None


@dataclass
class LaneFormation:
    shooter_track_id: int
    team: Team
    side: Side
    since_pts: float


@dataclass
class GameState:
    period: int | None = None
    clock_s: float | None = None
    shot_clock_s: float | None = None
    home_score: int = 0
    away_score: int = 0
    scoreboard_confidence: float = 0.0
    camera_state: str = "main"
    possession: Team = "unknown"
    t_pts: float = 0.0
    has_scoreboard: bool = False  # True once a confident scoreboard read has been applied

    # -- vision (M1) --------------------------------------------------------------------
    tracks: list[Track] = field(default_factory=list)
    ball: BallObs | None = None
    rim: Box | None = None
    homography_confidence: float = 0.0
    holder: Holder | None = None
    holder_candidate: tuple[int, float] | None = None  # track id, since pts
    last_release: Release | None = None
    ball_path: list[tuple[float, float, float]] = field(default_factory=list)  # t, x, y
    rim_interaction: RimInteraction | None = None
    last_shot: ShotRecord | None = None
    lane_formation: LaneFormation | None = None
    free_throws_pending: int = 0
    possession_since_pts: float | None = None
    clock_running: bool = False
    clock_state: ClockState = "unknown"
    last_clock_change_pts: float | None = None
    last_scoreboard_pts: float | None = None
    camera_left_main_pts: float | None = None

    def score(self) -> tuple[int, int]:
        return (self.home_score, self.away_score)
