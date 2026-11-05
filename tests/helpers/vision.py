"""Synthetic plays for the vision rules: a fixed side camera over a league court.

Plays are scripted in court metres; the scene projects them into VisionFrames through a
known homography, so rules, zones, the clock gate and the firsts folds run on
deterministic fixtures without any model.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import cv2
import numpy as np

from sports_watcher.clock import LatencyClass, StreamClock
from sports_watcher.config import LeagueConfig
from sports_watcher.events.engine import EventEngine
from sports_watcher.events.model import Event, EventStatus, SourceInfo
from sports_watcher.events.sinks import MemorySink
from sports_watcher.perception.scoreboard import ScoreboardState
from sports_watcher.perception.types import BallObs, Box, Homography, Track, VisionFrame
from sports_watcher.sports.base import Observation
from sports_watcher.sports.basketball.geometry import basket_xy
from sports_watcher.sports.basketball.plugin import BasketballPlugin
from tests.helpers.leagues import fixture_league

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
XY = tuple[float, float]
Players = dict[int, tuple[str, XY]]  # track id -> (team, feet xy in court metres)

PLAYER_W_M, PLAYER_H_M = 0.6, 1.9
BALL_R_M = 0.12
RIM_W_M, RIM_H_M, RIM_HEIGHT_M = 0.45, 0.135, 3.05
HANDS_M, HOLD_M = 2.0, 1.0


def _apply(h: np.ndarray, xy: XY) -> XY:
    v = h @ np.array([xy[0], xy[1], 1.0])
    return (float(v[0] / v[2]), float(v[1] / v[2]))


class Scene:
    """A side camera: near sideline along the bottom of a 1920x1080 frame."""

    def __init__(self, league: LeagueConfig) -> None:
        self.league = league
        length, width = league.court_length_m, league.court_width_m
        src = np.float32([(0, 0), (length, 0), (length, width), (0, width)])
        dst = np.float32([(100, 1000), (1820, 1000), (1500, 350), (420, 350)])
        self.court_to_image = cv2.getPerspectiveTransform(src, dst)
        self.image_to_court = np.linalg.inv(self.court_to_image)

    def ppm(self, court_xy: XY) -> float:
        """Pixels per metre at a court point (a 1 m step along the court length)."""
        a = _apply(self.court_to_image, court_xy)
        b = _apply(self.court_to_image, (court_xy[0] + 1.0, court_xy[1]))
        return math.dist(a, b)

    def project(self, court_xy: XY, height_m: float = 0.0) -> XY:
        x, y = _apply(self.court_to_image, court_xy)
        return (x, y - height_m * self.ppm(court_xy))

    def player(self, track_id: int, team: str, feet_xy: XY, cls: str = "player") -> Track:
        fx, fy = self.project(feet_xy)
        s = self.ppm(feet_xy)
        box: Box = (fx - PLAYER_W_M * s / 2, fy - PLAYER_H_M * s, fx + PLAYER_W_M * s / 2, fy)
        return Track(track_id, box, cls, team, feet_xy, 0.9)  # type: ignore[arg-type]

    def ball(self, floor_xy: XY, height_m: float, source: str = "detected") -> BallObs:
        return BallObs(
            self.project(floor_xy, height_m), BALL_R_M * self.ppm(floor_xy), floor_xy, source, 0.8
        )  # type: ignore[arg-type]

    def rim(self, side: str) -> Box:
        bxy = basket_xy(self.league, side)  # type: ignore[arg-type]
        cx, cy = self.project(bxy, RIM_HEIGHT_M)
        s = self.ppm(bxy)
        return (cx - RIM_W_M * s / 2, cy - RIM_H_M * s / 2, cx + RIM_W_M * s / 2, cy + RIM_H_M * s / 2)

    def homography(self, confidence: float = 0.9) -> Homography:
        return Homography(self.image_to_court, 20, 2.0, confidence)


class PlayBuilder:
    def __init__(self, scene: Scene, fps: float = 30.0, start_pts: float = 0.0) -> None:
        self.scene = scene
        self.fps = fps
        self.t = start_pts
        self.frames: list[VisionFrame] = []

    def frame(
        self,
        players: Players,
        ball: tuple[XY, float] | None,
        *,
        rim_side: str = "right",
        homography: bool = True,
        camera_hint: str = "main",
    ) -> VisionFrame:
        tracks = [self.scene.player(tid, team, xy) for tid, (team, xy) in players.items()]
        vf = VisionFrame(
            frame_index=len(self.frames),
            t_pts=round(self.t, 6),
            camera_hint=camera_hint,  # type: ignore[arg-type]
            camera_confidence=0.9,
            tracks=tracks,
            ball=self.scene.ball(*ball) if ball is not None else None,
            rim=self.scene.rim(rim_side),
            homography=self.scene.homography() if homography else None,
        )
        self.frames.append(vf)
        self.t += 1.0 / self.fps
        return vf

    def hold(self, seconds: float, players: Players, holder_id: int, **kw) -> list[VisionFrame]:
        feet = players[holder_id][1]
        return [self.frame(players, (feet, HOLD_M), **kw) for _ in range(round(seconds * self.fps))]

    def shot(
        self, shooter_id: int, players: Players, rim_side: str, flight_s: float = 1.0, outcome: str = "make"
    ) -> list[VisionFrame]:
        """A rim-bound arc from the shooter's hands; ``make`` drops through, ``miss`` falls beside."""
        start = players[shooter_id][1]
        target = basket_xy(self.scene.league, rim_side)  # type: ignore[arg-type]
        if outcome == "miss":
            # Long, off the back of the rim: the side camera sees that as an offset in
            # image x, beyond the rim's span, unlike an offset in court depth.
            direction = 1.0 if target[0] > start[0] else -1.0
            target = (target[0] + 0.4 * direction, target[1])
        out = []
        n = round(flight_s * self.fps)
        for i in range(1, n + 1):
            s = i / n
            floor = (start[0] + s * (target[0] - start[0]), start[1] + s * (target[1] - start[1]))
            height = HANDS_M + (RIM_HEIGHT_M - HANDS_M) * s + 4 * 1.5 * s * (1 - s)
            out.append(self.frame(players, (floor, height), rim_side=rim_side))
        drop = 0.5 if outcome == "make" else 1.5
        for i in range(1, 11):
            out.append(self.frame(players, (target, RIM_HEIGHT_M - drop * i / 10), rim_side=rim_side))
        return out

    def lob(
        self, shooter_id: int, catcher_id: int, players: Players, rim_side: str, flight_s: float = 0.8
    ) -> list[VisionFrame]:
        """The first 60 % of a rim-bound arc, then the ball in the catcher's hands."""
        start = players[shooter_id][1]
        target = basket_xy(self.scene.league, rim_side)  # type: ignore[arg-type]
        out = []
        n = round(flight_s * self.fps)
        for i in range(1, round(0.6 * n) + 1):
            s = i / n
            floor = (start[0] + s * (target[0] - start[0]), start[1] + s * (target[1] - start[1]))
            height = HANDS_M + (RIM_HEIGHT_M - HANDS_M) * s + 4 * 1.5 * s * (1 - s)
            out.append(self.frame(players, (floor, height), rim_side=rim_side))
        out.append(self.frame(players, (players[catcher_id][1], HOLD_M), rim_side=rim_side))
        return out

    def pass_(self, from_id: int, to_id: int, players: Players, flight_s: float = 0.5) -> list[VisionFrame]:
        a, b = players[from_id][1], players[to_id][1]
        n = round(flight_s * self.fps)
        return [
            self.frame(players, ((a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n), 1.5))
            for i in range(1, n + 1)
        ]


def observations(
    frames: list[VisionFrame],
    *,
    clock_start_s: float = 600.0,
    running: bool = True,
    resume_at: float | None = None,
    ocr_every_s: float = 0.25,
    period: int = 1,
    score: tuple[int, int] = (0, 0),
    deltas: dict[float, tuple[int, int]] | None = None,
    hidden: list[tuple[float, float]] | None = None,
) -> list[Observation]:
    """Pair frames with a confident scoreboard read every ``ocr_every_s``."""
    if not frames:
        return []
    dt = frames[1].t_pts - frames[0].t_pts if len(frames) > 1 else 1 / 30
    every = max(1, round(ocr_every_s / dt))
    changes = sorted((deltas or {}).items())
    out = []
    for i, vf in enumerate(frames):
        t = vf.t_pts
        sb = None
        is_hidden = any(start <= t < end for start, end in (hidden or []))
        if i % every == 0 and not is_hidden:
            if running:
                clock = clock_start_s - t
            elif resume_at is not None and t >= resume_at:
                clock = clock_start_s - (t - resume_at)
            else:
                clock = clock_start_s
            current = score
            for at, value in changes:
                if t >= at:
                    current = value
            sb = ScoreboardState(
                home=current[0], away=current[1], clock_s=round(max(clock, 0.0), 1), period=period,
                confidence=1.0, raw=[],
            )
        out.append(
            Observation(
                frame_index=vf.frame_index,
                t_pts=t,
                t_wall=NOW + timedelta(seconds=t),
                scoreboard=sb,
                camera_state=vf.camera_hint,
                vision=vf,
            )
        )
    return out


def drive(obs: list[Observation], league: LeagueConfig | None = None) -> list[Event]:
    """Run observations through a fresh plug-in and engine; return every emitted event."""
    plugin = BasketballPlugin(league or fixture_league("pba"))
    sink = MemorySink()
    engine = EventEngine(
        plugin, "g1", SourceInfo(adapter="scripted", latency_class=LatencyClass.FILE),
        StreamClock(LatencyClass.FILE), sinks=[sink],
    )
    for o in obs:
        engine.step(o)
    engine.finish()
    return list(sink.events)


def _of(events: list[Event], type: str, status: EventStatus) -> list[Event]:
    return [e for e in events if e.type == type and e.status == status]


def provisional(events: list[Event], type: str) -> list[Event]:
    return _of(events, type, EventStatus.provisional)


def confirmed(events: list[Event], type: str) -> list[Event]:
    return _of(events, type, EventStatus.confirmed)


def retracted(events: list[Event], type: str) -> list[Event]:
    return _of(events, type, EventStatus.retracted)
