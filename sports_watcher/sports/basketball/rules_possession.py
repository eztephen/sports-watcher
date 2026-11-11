"""Possession rules: possession_start, rebound and turnover (spec addendum 6.3)."""

from __future__ import annotations

from sports_watcher.config import LeagueConfig
from sports_watcher.events.model import PlayerRef
from sports_watcher.perception.types import Track
from sports_watcher.sports.base import EventDraft, Observation
from sports_watcher.sports.basketball.confirmers import (
    base_confidence,
    holding,
    possession_changes_within,
    rebound_holds,
    team_holds,
    team_holds_unless_attempt,
    vision_evidence,
)
from sports_watcher.state.game_state import GameState

POSSESSION_CONFIDENCE = 0.7
REBOUND_CONFIDENCE = 0.6
TURNOVER_CONFIDENCE = 0.6
SHOT_CLOCK_TURNOVER_CONFIDENCE = 0.5
SHOT_CLOCK_TURNOVER_TTL_S = 5.0
NO_RIM_MISS_S = 1.5
RESTART_GRACE_S = 2.0  # a hold that began this long before the clock restarted still counts
DELTA_WINDOW_S = 3.0  # default scoreboard lag; a broadcaster's score_lag_s replaces it
DEFAULT_TTL_S = 4.0  # the engine's default time to live for a provisional event
LAG_TTL_MARGIN_S = 1.0  # an event that waits out the lag lives this much longer


def _track(tracks: list[Track], track_id: int) -> Track | None:
    return next((tr for tr in tracks if tr.track_id == track_id), None)


def _player(tracks: list[Track], track_id: int) -> tuple[PlayerRef, tuple[float, float] | None, float | None]:
    tr = _track(tracks, track_id)
    conf = tr.confidence if tr is not None else 0.0
    return PlayerRef(track_id=track_id, confidence=conf), (tr.feet_xy if tr else None), (conf if tr else None)


def _attempt_in_possession(state: GameState, team: str, since_pts: float | None) -> bool:
    shot = state.last_shot
    return shot is not None and shot.team == team and since_pts is not None and shot.t_pts >= since_pts


class PossessionStartRule:
    """Once per possession, as soon as the holder is known and the clock is running."""

    name = "possession_start"

    def __init__(self) -> None:
        self._drafted_since: float | None = None
        self._drafted_restart: float | None = None
        self._holds: list[float] = []  # recent hold starts of the team in possession
        self._current: tuple[str, float] | None = None  # the possession being watched
        self._held = False  # ... and whether its team was seen holding the ball
        self._before: str | None = None  # the team that had the ball before it
        self._resumed_since: float | None = None  # a return after a hold too brief to count

    def __call__(self, prev: GameState, new: GameState, obs: Observation) -> list[EventDraft]:
        h = new.holder
        if h is not None and h.since_pts not in self._holds and new.possession in ("home", "away") and h.team in (new.possession, "unknown"):
            self._holds = [s for s in self._holds if obs.t_pts - s <= 4 * RESTART_GRACE_S] + [h.since_pts]
        self._watch(new, obs.t_pts)
        if new.possession not in ("home", "away") or h is None or not new.clock_running:
            return []
        since = new.possession_since_pts
        if since is None:
            return []
        opened = new.clock_running_since_pts
        stale = opened is not None and since < opened - RESTART_GRACE_S
        if stale:
            # The team had the ball before the clock restarted. A hold carried through the
            # stoppage is not a start; a new hold around the restart of a clock that stood
            # still (an inbound, the rebound of a free throw) is.
            fresh = [s for s in self._holds if opened - RESTART_GRACE_S <= s <= opened + RESTART_GRACE_S]
            if not (new.clock_resumed_from_stop and fresh) or self._drafted_restart == opened:
                return []
            self._drafted_restart = opened
            since = min(fresh)  # the first hold around the restart
        elif since == self._drafted_since:
            return []
        else:
            self._drafted_since = since
            if since == self._resumed_since:
                return []
        player, court_xy, det = _player(new.tracks, new.holder.track_id)
        return [
            EventDraft(
                type="possession_start",
                team=new.possession,
                t_pts=since,
                evidence=vision_evidence(new, obs, track_id=new.holder.track_id, detector_confidence=det),
                confidence=base_confidence(POSSESSION_CONFIDENCE, new),
                confirmer=team_holds(new.possession),
                player=player,
                court_xy=court_xy,
            )
        ]

    def _watch(self, new: GameState, t_pts: float) -> None:
        """Follow the possession and whether its team was seen with the ball in hand. When
        the ball comes back to the team that had it before a possession nobody was seen to
        hold (its start is retracted), the old possession goes on: no new start."""
        since = new.possession_since_pts
        if since is None or new.possession not in ("home", "away"):
            return
        if self._current is None or self._current[1] != since:
            returned = self._current is not None and not self._held and self._before == new.possession
            if self._current is not None and self._held:
                self._before = self._current[0]
            self._resumed_since = since if returned else None
            self._current, self._held = (new.possession, since), returned
        if holding(new, t_pts):
            self._held = True


class ReboundRule:
    """The first holder after a miss, once per shot, tagged offensive or defensive."""

    name = "rebound"

    def __init__(self, score_lag_s: float = DELTA_WINDOW_S) -> None:
        self._lag = score_lag_s
        self._shot_t: float | None = None
        self._score_at: tuple[int, int] = (0, 0)
        self._fired: float | None = None

    def __call__(self, prev: GameState, new: GameState, obs: Observation) -> list[EventDraft]:
        shot = new.last_shot
        if shot is None:
            return []
        if self._shot_t != shot.t_pts:
            self._shot_t, self._score_at = shot.t_pts, new.score()
        if self._fired == shot.t_pts or new.score() != self._score_at or not new.clock_running:
            return []
        ri = new.rim_interaction
        if ri is not None and ri.crossed_pts is not None and ri.crossed_pts > shot.t_pts:
            self._fired = shot.t_pts  # the ball went through: nothing to rebound
            return []
        left_rim = ri is not None and ri.left_pts is not None and ri.left_pts > shot.t_pts
        never_entered = (ri is None or ri.entered_pts is None) and obs.t_pts - shot.t_pts >= NO_RIM_MISS_S
        if not (left_rim or never_entered):
            return []
        h = new.holder
        if h is None or h.since_pts <= shot.t_pts or h.team not in ("home", "away"):
            return []
        self._fired = shot.t_pts
        player, court_xy, det = _player(new.tracks, h.track_id)
        note = "offensive" if h.team == shot.team else "defensive"
        return [
            EventDraft(
                type="rebound",
                team=h.team,
                t_pts=h.since_pts,
                evidence=vision_evidence(new, obs, track_id=h.track_id, detector_confidence=det, note=note),
                confidence=base_confidence(REBOUND_CONFIDENCE, new),
                confirmer=rebound_holds(h.team, shot.team, self._score_at, not_before_pts=shot.t_pts + self._lag),
                ttl_s=max(DEFAULT_TTL_S, self._lag + LAG_TTL_MARGIN_S),
                player=player,
                court_xy=court_xy,
            )
        ]


class TurnoverRule:
    """A possession change with no attempt in between, or a shot-clock reset with the
    ball out of sight."""

    name = "turnover"

    def __init__(self, league: LeagueConfig, score_lag_s: float = DELTA_WINDOW_S) -> None:
        self._lag = score_lag_s
        self._full = float(league.shot_clock_s)

    def __call__(self, prev: GameState, new: GameState, obs: Observation) -> list[EventDraft]:
        if not new.clock_running:
            return []
        if (
            prev.possession in ("home", "away")
            and new.possession in ("home", "away")
            and new.possession != prev.possession
            and not _attempt_in_possession(new, prev.possession, prev.possession_since_pts)
        ):
            since = new.possession_since_pts if new.possession_since_pts is not None else obs.t_pts
            window = (prev.possession_since_pts or 0.0, since)
            return [
                EventDraft(
                    type="turnover",
                    team=prev.possession,
                    t_pts=since,
                    evidence=vision_evidence(new, obs, note="possession_change"),
                    confidence=base_confidence(TURNOVER_CONFIDENCE, new),
                    confirmer=team_holds_unless_attempt(
                        new.possession, prev.possession, window, new.score(), not_before_pts=since + self._lag
                    ),
                    ttl_s=max(DEFAULT_TTL_S, self._lag + LAG_TTL_MARGIN_S),
                )
            ]
        if (
            prev.shot_clock_s is not None
            and new.shot_clock_s is not None
            and prev.shot_clock_s <= self._full - 2
            and new.shot_clock_s == self._full
            and new.holder is None
            and new.possession in ("home", "away")
            and new.possession == prev.possession
            and not _attempt_in_possession(new, new.possession, new.possession_since_pts)
        ):
            return [
                EventDraft(
                    type="turnover",
                    team=new.possession,
                    t_pts=obs.t_pts,
                    evidence=vision_evidence(new, obs, note="shot_clock_reset"),
                    confidence=base_confidence(SHOT_CLOCK_TURNOVER_CONFIDENCE, new),
                    confirmer=possession_changes_within(new.possession, obs.t_pts, SHOT_CLOCK_TURNOVER_TTL_S),
                    ttl_s=SHOT_CLOCK_TURNOVER_TTL_S,
                )
            ]
        return []
