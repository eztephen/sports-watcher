"""Rules that read the clock and the camera: timeout and tip_off (spec addendum 6.3)."""

from __future__ import annotations

from sports_watcher.config import LeagueConfig
from sports_watcher.sports.base import EventDraft, Observation
from sports_watcher.sports.basketball.confirmers import (
    base_confidence,
    clock_stopped_until,
    possession_within,
    vision_evidence,
)
from sports_watcher.sports.basketball.geometry import centre_xy, near
from sports_watcher.sports.basketball.rules import CLOCK_ZERO_EPS, full_length_s
from sports_watcher.sports.basketball.vision_state import MIN_HOMOGRAPHY_CONF
from sports_watcher.state.game_state import GameState

TIMEOUT_DRAFT_S = 20.0
TIMEOUT_CONFIRM_S = 30.0
TIMEOUT_TTL_S = 12.0
TIMEOUT_CONFIDENCE = 0.7
TIP_RADIUS_M = 2.0
TIP_RISING_OBS = 3
TIP_CONFIRM_S = 5.0
TIP_TTL_S = 10.0
TIP_CONFIDENCE = 0.6


class TimeoutRule:
    """Clock stopped and the camera off main for 20 s, once per departure from main."""

    name = "timeout"

    def __init__(self) -> None:
        self._fired_for: float | None = None

    def __call__(self, prev: GameState, new: GameState, obs: Observation) -> list[EventDraft]:
        left = new.camera_left_main_pts
        if left is None or new.clock_running or obs.camera_state == "main" or left == self._fired_for:
            return []
        if obs.t_pts - left < TIMEOUT_DRAFT_S:
            return []
        self._fired_for = left
        return [
            EventDraft(
                type="timeout",
                team="unknown",
                t_pts=left,
                evidence=vision_evidence(new, obs, note="camera_off_main"),
                confidence=TIMEOUT_CONFIDENCE,
                confirmer=clock_stopped_until(left + TIMEOUT_CONFIRM_S),
                ttl_s=TIMEOUT_TTL_S,
            )
        ]


class TipOffRule:
    """Period 1 or overtime, clock at full, two players at the centre circle and the ball
    rising from between them on three consecutive perception frames. Once per period."""

    name = "tip_off"

    def __init__(self, league: LeagueConfig) -> None:
        self._league = league
        self._fired: set[int] = set()
        self._ys: list[tuple[float, float]] = []

    def __call__(self, prev: GameState, new: GameState, obs: Observation) -> list[EventDraft]:
        p = new.period
        if p is None or p in self._fired or not (p == 1 or p > self._league.periods):
            return []
        if new.clock_s is None or abs(new.clock_s - full_length_s(self._league, p)) >= CLOCK_ZERO_EPS:
            self._ys = []
            return []
        vf = obs.vision
        if vf is None:
            return []
        if (
            obs.camera_state != "main"
            or new.homography_confidence < MIN_HOMOGRAPHY_CONF
            or vf.ball is None
            or vf.ball.source != "detected"
        ):
            self._ys = []
            return []
        centre = centre_xy(self._league)
        jumpers = [t for t in vf.tracks if t.feet_xy is not None and near(t.feet_xy, centre, TIP_RADIUS_M)]
        if len(jumpers) < 2:
            self._ys = []
            return []
        self._ys = (self._ys + [(obs.t_pts, vf.ball.xy[1])])[-TIP_RISING_OBS:]
        if len(self._ys) < TIP_RISING_OBS or not all(
            self._ys[i][1] > self._ys[i + 1][1] for i in range(TIP_RISING_OBS - 1)
        ):
            return []
        self._fired.add(p)
        t_pts = self._ys[1][0]  # the first observation that showed the ball rising
        return [
            EventDraft(
                type="tip_off",
                team="unknown",
                t_pts=t_pts,
                evidence=vision_evidence(new, obs, note=f"period {p}"),
                confidence=base_confidence(TIP_CONFIDENCE, new),
                confirmer=possession_within(t_pts, TIP_CONFIRM_S),
                ttl_s=TIP_TTL_S,
                court_xy=centre,
            )
        ]
