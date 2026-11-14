"""Shot rules: shot_attempt, shot_made, shot_missed and the retroactive attempt
(spec addendum 6.3). Free throws join in FreeThrowRule."""

from __future__ import annotations

from dataclasses import dataclass

from sports_watcher.config import LeagueConfig
from sports_watcher.events.model import PlayerRef
from sports_watcher.sports.base import EventDraft, Observation
from sports_watcher.sports.basketball.confirmers import (
    _held_by,
    always_confirm,
    base_confidence,
    delta_within,
    ft_resolves,
    holder_or_late_delta,
    rim_or_delta,
    team_delta,
    vision_evidence,
)
from sports_watcher.sports.basketball.geometry import free_throw_xy
from sports_watcher.state.game_state import GameState

DEFAULT_TTL_S = 4.0  # the engine's default time to live for a provisional event
ATTEMPT_CONFIDENCE = 0.7
RETROACTIVE_CONFIDENCE = 0.4
MADE_DELTA_CONFIDENCE = 0.9
MADE_VISION_CONFIDENCE = 0.6
MISSED_CONFIDENCE = 0.6
DELTA_WINDOW_S = 3.0  # default scoreboard lag; a broadcaster's score_lag_s replaces it
MADE_TTL_MARGIN_S = 1.0  # a vision-only shot_made outlives the lag by this much
MISS_AFTER_LEAVE_S = 3.0
NO_RIM_MISS_S = 1.5
OPEN_ATTEMPT_MAX_S = 10.0
FT_ATTEMPT_CONFIDENCE = 0.6
FT_MADE_CONFIDENCE = 0.9
FT_MISSED_CONFIDENCE = 0.6
FT_RESOLVE_S = 3.0
ASSUMED_FLIGHT_S = 1.0  # release to net when only the score change is known
CROSSING_WINDOW_S = 5.5  # release to the ball through the rim: the hidden-release gap plus a flight
SAME_SHOT_S = 0.5  # releases closer than this by one team are one shot (the ball changing track ids)
ZONE_POINTS = {"two": 2, "three": 3}


def _same_shot(last: tuple[float, str] | None, t_pts: float, team: str) -> bool:
    if last is None or not 0.0 <= t_pts - last[0] < SAME_SHOT_S:
        return False
    return team == last[1] or "unknown" in (team, last[1])
POINTS_ZONE = {1: "ft", 2: "two", 3: "three"}


@dataclass
class _HeldShot:
    shot: object  # the ShotRecord of a shooter whose team is unknown
    score_at: tuple[int, int]
    since: float
    rim_seen: bool = False


class ShotAttemptRule:
    """Drafts once per recorded shot while the clock runs; free throws are not shots.

    A shot by a player whose team is unknown is held back: a 2 or 3 inside the scoreboard
    lag gives it the scoring team (drafting it at once and adding the score-change twin
    put two attempts on one shot); with no score change it is drafted as it stands when
    the lag is out, if the ball reached the rim."""

    name = "shot_attempt"

    def __init__(self, score_lag_s: float = DELTA_WINDOW_S) -> None:
        self._drafted: set[float] = set()
        self._lag = score_lag_s
        self._held: list[_HeldShot] = []
        self._last: tuple[float, str] | None = None

    def __call__(self, prev: GameState, new: GameState, obs: Observation) -> list[EventDraft]:
        drafts = self._release_held(new, obs)
        shot = new.last_shot
        if shot is None or shot.t_pts in self._drafted:
            return drafts
        self._drafted.add(shot.t_pts)  # decided once, at record time: never deferred
        if not shot.clock_running or shot.free_throw:
            return drafts
        if _same_shot(self._last, shot.t_pts, shot.team):
            return drafts
        self._last = (shot.t_pts, shot.team)
        since = shot.t_pts if shot.release_pts is None else min(shot.t_pts, shot.release_pts)
        if shot.team not in ("home", "away"):
            self._held.append(_HeldShot(shot, new.score(), since))
            return drafts
        # the rim entry counts from the release: a layup dated at the shooter's last
        # touch is already in the rim area by then
        return [*drafts, self._draft(shot, shot.team, rim_or_delta(shot.team, since, new.score()), new, obs)]

    def _release_held(self, new: GameState, obs: Observation) -> list[EventDraft]:
        out, keep = [], []
        ri = new.rim_interaction
        for h in self._held:
            if ri is not None and ri.entered_pts is not None and ri.entered_pts > h.since:
                h.rim_seen = True
            scorer = next((t for t in ("home", "away") if team_delta(t, h.score_at, new.score()) in (2, 3)), None)
            if scorer is not None:
                out.append(self._draft(h.shot, scorer, always_confirm(), new, obs))
            elif obs.t_pts - h.shot.t_pts < self._lag:
                keep.append(h)
            elif h.rim_seen:
                out.append(self._draft(h.shot, h.shot.team, always_confirm(), new, obs))
        self._held = keep
        return out

    @staticmethod
    def _draft(shot, team, confirmer, new: GameState, obs: Observation) -> EventDraft:
        tr = next((t for t in new.tracks if t.track_id == shot.track_id), None)
        conf = tr.confidence if tr is not None else 0.0
        return EventDraft(
            type="shot_attempt",
            team=team,
            t_pts=shot.t_pts,
            evidence=vision_evidence(new, obs, track_id=shot.track_id, detector_confidence=conf),
            confidence=base_confidence(ATTEMPT_CONFIDENCE, new),
            confirmer=confirmer,
            player=PlayerRef(track_id=shot.track_id, confidence=conf),
            court_xy=shot.court_xy,
            zone=shot.zone,
        )


@dataclass
class _OpenAttempt:
    t_pts: float
    team: str
    track_id: int
    zone: str
    court_xy: tuple[float, float] | None
    score_at: tuple[int, int]
    since: float = 0.0  # the release: rim entries after it are this attempt's
    rim_seen: bool = False
    dead: bool = False  # retracted: the shooters held the ball again before it reached the rim
    made_points: int | None = None  # points the vision-only shot_made carries (0 = unknown zone)
    missed: bool = False
    resolved: bool = False


class ShotOutcomeRule:
    """Made and missed outcomes for open attempts, plus the retroactive attempt a score
    delta implies when vision saw none."""

    name = "shot_outcome"

    def __init__(self, score_lag_s: float = DELTA_WINDOW_S, score_delay_s: float | None = None) -> None:
        self._open: list[_OpenAttempt] = []
        self._seen: set[float] = set()
        self._lag = score_lag_s
        self._delay = score_delay_s
        self._flights_used: set[float] = set()
        self._last: tuple[float, str] | None = None
        self._miss_wait = max(MISS_AFTER_LEAVE_S, score_lag_s)  # a miss is not called before the lag is out
        self._open_max = max(OPEN_ATTEMPT_MAX_S, score_lag_s + MADE_TTL_MARGIN_S)

    def __call__(self, prev: GameState, new: GameState, obs: Observation) -> list[EventDraft]:
        drafts: list[EventDraft] = []
        self._open_new(new)
        # an attempt stays open until its miss can be called: the miss waits out the lag,
        # which on a slow scoreboard is longer than the window for a made basket
        self._open = [
            a for a in self._open
            if obs.t_pts - a.t_pts <= self._open_max + (0.0 if a.resolved or a.missed else self._miss_wait)
        ]
        ri = new.rim_interaction
        if ri is not None and ri.entered_pts is not None:
            for a in self._open:
                if ri.entered_pts > a.since and not a.dead:
                    a.rim_seen = True
        for a in self._open:
            # the attempt's own confirmer retracts it when the shooters hold the ball again
            # before it reaches the rim (a lob, a pump fake): a later rim entry is not its
            if not a.rim_seen and _held_by(new, a.team, a.since, obs):
                a.dead = True
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
        self._seen.add(shot.t_pts)
        if not shot.clock_running or shot.free_throw:
            return
        if _same_shot(self._last, shot.t_pts, shot.team):
            return
        self._last = (shot.t_pts, shot.team)
        since = shot.t_pts if shot.release_pts is None else min(shot.t_pts, shot.release_pts)
        self._open.append(
            _OpenAttempt(shot.t_pts, shot.team, shot.track_id, shot.zone, shot.court_xy, new.score(), since)
        )

    def _on_delta(self, team: str, points: int, new: GameState, obs: Observation) -> list[EventDraft]:
        if not new.has_vision:
            return []  # scoreboard-only mode behaves exactly as M0
        if points == 1:
            recent_ft = new.last_free_throw_pts is not None and obs.t_pts - new.last_free_throw_pts <= FT_RESOLVE_S
            if new.free_throws_pending or recent_ft:
                return []  # FreeThrowRule owns it
            return self._retroactive_free_throw(team, new, obs)
        # the latest attempt of the scorers (or of a shooter whose team is unknown) inside
        # the lag; one the ball reached the rim after comes first, so a release read off the
        # ball dropping out of the net does not take the points
        candidates = [
            a for a in self._open
            if a.team in (team, "unknown") and not a.resolved and not a.missed and not a.dead and obs.t_pts - a.t_pts <= self._lag
        ]
        att = next((a for a in reversed(candidates) if a.rim_seen), candidates[-1] if candidates else None)
        if att is not None:
            att.team = team
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
        before = [a for a in self._open if a.t_pts < ri.crossed_pts]
        for att in before[-1:]:  # the crossing is the latest attempt's, not every open one's
            if att.resolved or att.missed or att.dead or att.made_points is not None or att.team not in ("home", "away"):
                continue
            if ri.crossed_pts - att.since > CROSSING_WINDOW_S:
                continue  # a basket long after this attempt is another shot's (release unseen)
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
                    ttl_s=max(DEFAULT_TTL_S, self._lag + MADE_TTL_MARGIN_S),
                )
            )
        return out

    def _on_miss(self, new: GameState, obs: Observation) -> list[EventDraft]:
        ri = new.rim_interaction
        out = []
        for att in self._open:
            if att.resolved or att.missed or att.dead or att.team not in ("home", "away"):
                continue
            if not att.rim_seen and obs.t_pts - att.t_pts > DEFAULT_TTL_S:
                att.missed = True  # never reached the rim: its attempt was retracted, no miss to call
                continue
            left = ri.left_pts if ri is not None and ri.left_pts is not None and ri.left_pts > att.t_pts else None
            entered = ri is not None and ri.entered_pts is not None and ri.entered_pts > att.t_pts
            if left is not None:
                miss_t, deadline = left, left + self._miss_wait
            elif not entered:
                miss_t, deadline = att.t_pts + NO_RIM_MISS_S, att.t_pts + NO_RIM_MISS_S + self._miss_wait
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
                    # the lag is out with no score change: nothing is left to wait for
                    confirmer=always_confirm(),
                    player=PlayerRef(track_id=att.track_id, confidence=0.0),
                    court_xy=att.court_xy,
                    zone=att.zone,
                )
            )
        return out

    def _retroactive_time(self, new: GameState, obs: Observation) -> float:
        """When the shot behind a score change was released, best guess first: the start of
        a flight seen to reach the rim inside the scoreboard lag, the broadcaster's usual
        score delay, or the score change itself."""
        flight = new.last_rim_flight_pts
        if flight is not None and flight not in self._flights_used and 0.0 <= obs.t_pts - flight <= self._lag:
            self._flights_used.add(flight)
            return flight
        if self._delay is not None:
            return obs.t_pts - self._delay - ASSUMED_FLIGHT_S
        return obs.t_pts

    def _retroactive_free_throw(self, team: str, new: GameState, obs: Observation) -> list[EventDraft]:
        """One point is a free throw whether or not the lane formation was seen."""
        t = self._retroactive_time(new, obs)
        return [
            EventDraft(
                type=type_,
                team=team,
                t_pts=t,
                evidence=vision_evidence(new, obs, points=1, note="retroactive"),
                confidence=RETROACTIVE_CONFIDENCE,
                confirmer=always_confirm(),
                court_xy=None,
                zone="ft",
            )
            for type_ in ("free_throw_attempt", "free_throw_made")
        ]

    def _retroactive_attempt(self, team: str, points: int, new: GameState, obs: Observation) -> EventDraft:
        return EventDraft(
            type="shot_attempt",
            team=team,
            t_pts=self._retroactive_time(new, obs),
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


@dataclass
class _OpenFreeThrow:
    t_pts: float
    team: str
    track_id: int
    score_at: tuple[int, int]
    resolved: bool = False


class FreeThrowRule:
    """Clock stopped, a lane formation, and the shooter releasing from the line: the
    attempt on release, then made on a +1 delta within 3 s or missed at 3 s."""

    name = "free_throw"

    def __init__(self, league: LeagueConfig) -> None:
        self._league = league
        self._last_release: float | None = None
        self._open: list[_OpenFreeThrow] = []

    def __call__(self, prev: GameState, new: GameState, obs: Observation) -> list[EventDraft]:
        drafts: list[EventDraft] = []
        rel, lf = new.last_release, new.lane_formation
        if (
            rel is not None
            and rel.t_pts != self._last_release
            and lf is not None
            and rel.track_id == lf.shooter_track_id
            and not new.clock_running
        ):
            self._last_release = rel.t_pts
            self._open.append(_OpenFreeThrow(rel.t_pts, lf.team, rel.track_id, new.score()))
            drafts.append(
                EventDraft(
                    type="free_throw_attempt",
                    team=lf.team,
                    t_pts=rel.t_pts,
                    evidence=vision_evidence(new, obs, track_id=rel.track_id),
                    confidence=base_confidence(FT_ATTEMPT_CONFIDENCE, new),
                    confirmer=ft_resolves(lf.team, rel.t_pts, new.score()),
                    player=PlayerRef(track_id=rel.track_id, confidence=0.0),
                    court_xy=free_throw_xy(self._league, lf.side),
                    zone="ft",
                )
            )
        for ft in self._open:
            if ft.resolved:
                continue
            player = PlayerRef(track_id=ft.track_id, confidence=0.0)
            if team_delta(ft.team, ft.score_at, new.score()) == 1 and obs.t_pts - ft.t_pts <= FT_RESOLVE_S:
                ft.resolved = True
                drafts.append(
                    EventDraft(
                        type="free_throw_made",
                        team=ft.team,
                        t_pts=ft.t_pts,
                        evidence=vision_evidence(new, obs, track_id=ft.track_id, points=1),
                        confidence=base_confidence(FT_MADE_CONFIDENCE, new),
                        confirmer=always_confirm(),
                        player=player,
                        zone="ft",
                    )
                )
            elif obs.t_pts - ft.t_pts >= FT_RESOLVE_S:
                ft.resolved = True
                drafts.append(
                    EventDraft(
                        type="free_throw_missed",
                        team=ft.team,
                        t_pts=ft.t_pts,
                        evidence=vision_evidence(new, obs, track_id=ft.track_id),
                        confidence=base_confidence(FT_MISSED_CONFIDENCE, new),
                        confirmer=holder_or_late_delta(ft.team, ft.t_pts, ft.score_at),
                        player=player,
                        zone="ft",
                    )
                )
        self._open = [ft for ft in self._open if not ft.resolved]
        return drafts
