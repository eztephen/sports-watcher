"""Shot rules: shot_attempt, shot_made, shot_missed and the retroactive attempt
(spec addendum 6.3). Free throws join in FreeThrowRule."""

from __future__ import annotations

from dataclasses import dataclass

from sports_watcher.events.model import PlayerRef
from sports_watcher.sports.base import EventDraft, Observation
from sports_watcher.sports.basketball.confirmers import (
    always_confirm,
    base_confidence,
    delta_within,
    holder_or_late_delta,
    rim_or_delta,
    team_delta,
    vision_evidence,
)
from sports_watcher.state.game_state import GameState

ATTEMPT_CONFIDENCE = 0.7
RETROACTIVE_CONFIDENCE = 0.4
MADE_DELTA_CONFIDENCE = 0.9
MADE_VISION_CONFIDENCE = 0.6
MISSED_CONFIDENCE = 0.6
DELTA_WINDOW_S = 3.0
MISS_AFTER_LEAVE_S = 3.0
NO_RIM_MISS_S = 1.5
OPEN_ATTEMPT_MAX_S = 10.0
ZONE_POINTS = {"two": 2, "three": 3}
POINTS_ZONE = {1: "ft", 2: "two", 3: "three"}


class ShotAttemptRule:
    """Drafts once per recorded shot while the clock runs; free throws are not shots."""

    name = "shot_attempt"

    def __init__(self) -> None:
        self._drafted: set[float] = set()

    def __call__(self, prev: GameState, new: GameState, obs: Observation) -> list[EventDraft]:
        shot = new.last_shot
        if shot is None or shot.t_pts in self._drafted:
            return []
        if not new.clock_running or new.lane_formation is not None:
            return []
        self._drafted.add(shot.t_pts)
        tr = next((t for t in new.tracks if t.track_id == shot.track_id), None)
        conf = tr.confidence if tr is not None else 0.0
        return [
            EventDraft(
                type="shot_attempt",
                team=shot.team,
                t_pts=shot.t_pts,
                evidence=vision_evidence(new, obs, track_id=shot.track_id, detector_confidence=conf),
                confidence=base_confidence(ATTEMPT_CONFIDENCE, new),
                confirmer=rim_or_delta(shot.team, shot.t_pts, new.score()),
                player=PlayerRef(track_id=shot.track_id, confidence=conf),
                court_xy=shot.court_xy,
                zone=shot.zone,
            )
        ]


@dataclass
class _OpenAttempt:
    t_pts: float
    team: str
    track_id: int
    zone: str
    court_xy: tuple[float, float] | None
    score_at: tuple[int, int]
    made_points: int | None = None  # points the vision-only shot_made carries (0 = unknown zone)
    missed: bool = False
    resolved: bool = False


class ShotOutcomeRule:
    """Made and missed outcomes for open attempts, plus the retroactive attempt a score
    delta implies when vision saw none."""

    name = "shot_outcome"

    def __init__(self) -> None:
        self._open: list[_OpenAttempt] = []
        self._seen: set[float] = set()

    def __call__(self, prev: GameState, new: GameState, obs: Observation) -> list[EventDraft]:
        drafts: list[EventDraft] = []
        self._open_new(new)
        self._open = [a for a in self._open if obs.t_pts - a.t_pts <= OPEN_ATTEMPT_MAX_S]
        for team in ("home", "away"):
            d = team_delta(team, prev.score(), new.score())
            if d in (1, 2, 3):
                drafts += self._on_delta(team, d, new, obs)
        drafts += self._on_crossing(prev, new, obs)
        drafts += self._on_miss(new, obs)
        return drafts

    # -- pieces -------------------------------------------------------------------------

    def _open_new(self, new: GameState) -> None:
        shot = new.last_shot
        if shot is None or shot.t_pts in self._seen:
            return
        if not new.clock_running or new.lane_formation is not None:
            return
        self._seen.add(shot.t_pts)
        self._open.append(
            _OpenAttempt(shot.t_pts, shot.team, shot.track_id, shot.zone, shot.court_xy, new.score())
        )

    def _on_delta(self, team: str, points: int, new: GameState, obs: Observation) -> list[EventDraft]:
        if not new.has_vision:
            return []  # scoreboard-only mode behaves exactly as M0
        if points == 1:
            if new.free_throws_pending:
                return []  # FreeThrowRule owns it
            return [self._retroactive_attempt(team, points, new, obs)]
        att = next(
            (
                a
                for a in reversed(self._open)
                if a.team == team and not a.resolved and obs.t_pts - a.t_pts <= DELTA_WINDOW_S
            ),
            None,
        )
        if att is None:
            retro = self._retroactive_attempt(team, points, new, obs)
            return [retro, self._made(team, points, retro.t_pts, RETROACTIVE_CONFIDENCE, new, obs, retro.player, retro.zone, None, note="retroactive")]
        att.resolved = True
        if att.made_points is not None and att.made_points in (0, points):
            return []  # the vision-only shot_made confirms itself on this delta
        player = PlayerRef(track_id=att.track_id, confidence=0.0)
        return [self._made(team, points, att.t_pts, MADE_DELTA_CONFIDENCE, new, obs, player, att.zone, att.court_xy)]

    def _on_crossing(self, prev: GameState, new: GameState, obs: Observation) -> list[EventDraft]:
        ri = new.rim_interaction
        if ri is None or ri.crossed_pts is None:
            return []
        if prev.rim_interaction is not None and prev.rim_interaction.crossed_pts == ri.crossed_pts:
            return []
        out = []
        for att in self._open:
            if att.resolved or att.made_points is not None or ri.crossed_pts <= att.t_pts:
                continue
            points = ZONE_POINTS.get(att.zone)
            att.made_points = points if points is not None else 0
            expect = (points,) if points is not None else (2, 3)
            out.append(
                EventDraft(
                    type="shot_made",
                    team=att.team,
                    t_pts=att.t_pts,
                    evidence=vision_evidence(new, obs, track_id=att.track_id, points=points, note="rim_crossing"),
                    confidence=base_confidence(MADE_VISION_CONFIDENCE, new),
                    confirmer=delta_within(att.team, expect, att.score_at, confirmed_confidence=MADE_DELTA_CONFIDENCE),
                    player=PlayerRef(track_id=att.track_id, confidence=0.0),
                    court_xy=att.court_xy,
                    zone=att.zone,
                )
            )
        return out

    def _on_miss(self, new: GameState, obs: Observation) -> list[EventDraft]:
        ri = new.rim_interaction
        out = []
        for att in self._open:
            if att.resolved or att.missed:
                continue
            left = ri.left_pts if ri is not None and ri.left_pts is not None and ri.left_pts > att.t_pts else None
            entered = ri is not None and ri.entered_pts is not None and ri.entered_pts > att.t_pts
            if left is not None:
                miss_t, deadline = left, left + MISS_AFTER_LEAVE_S
            elif not entered:
                miss_t, deadline = att.t_pts + NO_RIM_MISS_S, att.t_pts + NO_RIM_MISS_S + MISS_AFTER_LEAVE_S
            else:
                continue
            if obs.t_pts < deadline:
                continue
            att.missed = True
            out.append(
                EventDraft(
                    type="shot_missed",
                    team=att.team,
                    t_pts=miss_t,
                    evidence=vision_evidence(new, obs, track_id=att.track_id),
                    confidence=base_confidence(MISSED_CONFIDENCE, new),
                    confirmer=holder_or_late_delta(att.team, miss_t, att.score_at),
                    player=PlayerRef(track_id=att.track_id, confidence=0.0),
                    court_xy=att.court_xy,
                    zone=att.zone,
                )
            )
        return out

    def _retroactive_attempt(self, team: str, points: int, new: GameState, obs: Observation) -> EventDraft:
        return EventDraft(
            type="shot_attempt",
            team=team,
            t_pts=obs.t_pts,
            evidence=vision_evidence(new, obs, points=points, note="retroactive"),
            confidence=RETROACTIVE_CONFIDENCE,
            confirmer=always_confirm(),
            court_xy=None,
            zone=POINTS_ZONE[points],
        )

    @staticmethod
    def _made(team, points, t_pts, confidence, new, obs, player, zone, court_xy, note=None) -> EventDraft:
        track_id = player.track_id if player is not None else None
        return EventDraft(
            type="shot_made",
            team=team,
            t_pts=t_pts,
            evidence=vision_evidence(new, obs, track_id=track_id, points=points, note=note),
            confidence=confidence if note == "retroactive" else base_confidence(confidence, new),
            confirmer=always_confirm(),
            player=player,
            court_xy=court_xy,
            zone=zone,
        )
