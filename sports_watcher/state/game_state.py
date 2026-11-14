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
    clock_running: bool = True  # the clock gate when the shot was recorded
    free_throw: bool = False  # released from a lane formation
    release_pts: float | None = None  # the release it was recorded for (t_pts may be later)


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
    has_vision: bool = False  # True once any observation carried a VisionFrame
    tracks: list[Track] = field(default_factory=list)
    ball: BallObs | None = None
    rim: Box | None = None
    homography_confidence: float = 0.0
    holder: Holder | None = None
    holder_candidate: tuple[int, float] | None = None  # track id, since pts
    ball_free_since_pts: float | None = None  # start of the current run of detections out of every reach
    ball_seen_pts: float | None = None  # last detection of the ball
    last_release: Release | None = None
    ball_path: list[tuple[float, float, float]] = field(default_factory=list)  # t, x, y
    rim_interaction: RimInteraction | None = None
    last_shot: ShotRecord | None = None
    last_rim_flight_pts: float | None = None  # start of the last flight seen to reach the rim region
    release_left_reach: bool = False  # the ball was seen out of the releaser's reach since the release
    release_touch_pts: float | None = None  # last time it was seen back in his reach after that
    last_held: tuple[int, Team, float] | None = None  # track, team, last time someone was the holder
    ball_touch: tuple[int, Team, float] | None = None  # the player whose reach the ball was last seen in
    rim_low_y: float | None = None  # lowest the ball has been (largest image y) since it reached the rim
    lane_formation: LaneFormation | None = None
    free_throws_pending: int = 0
    possession_since_pts: float | None = None
    possession_locked_until_pts: float | None = None  # after a made basket: no flip on a label
    clock_running: bool = False
    clock_state: ClockState = "unknown"
    clock_running_since_pts: float | None = None  # when the gate last opened
    clock_resumed_from_stop: bool = False  # the gate last opened on a clock that had stood still
    last_clock_change_pts: float | None = None
    last_clock_change_value: float | None = None  # the clock read at that change
    last_clock_step_s: float | None = None  # how far that change moved the clock
    last_scoreboard_pts: float | None = None
    last_free_throw_pts: float | None = None  # release by the lane formation's shooter
    camera_left_main_pts: float | None = None

    def score(self) -> tuple[int, int]:
        return (self.home_score, self.away_score)
