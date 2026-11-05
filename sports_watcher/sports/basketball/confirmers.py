"""Confirmers for the vision rules, plus the evidence and confidence helpers they share.

A confirmer runs on every later observation until it answers. Verdicts are ``pending``,
``confirmed``, ``confirmed:<confidence>`` (the terminal copy takes that confidence),
``retracted`` or ``retracted:<reason>`` (the engine records the reason).
"""

from __future__ import annotations

from sports_watcher.events.model import Evidence
from sports_watcher.perception.types import Team
from sports_watcher.sports.base import Confirmer, EventDraft, Observation, Verdict
from sports_watcher.state.game_state import GameState

HOLD_CONFIRM_S = 0.5
ASSUMED_PENALTY = 0.2
Score = tuple[int, int]


def team_delta(team: str, before: Score, after: Score) -> int:
    if team == "home":
        return after[0] - before[0]
    if team == "away":
        return after[1] - before[1]
    return 0


def other(team: str) -> str:
    return {"home": "away", "away": "home"}.get(team, "unknown")


def base_confidence(value: float, state: GameState) -> float:
    if state.clock_state == "assumed":
        return max(0.05, round(value - ASSUMED_PENALTY, 2))
    return value


def vision_evidence(state: GameState, obs: Observation, **extra) -> Evidence:
    fields = {
        "camera_state": obs.camera_state,
        "clock_state": state.clock_state,
        "homography_confidence": state.homography_confidence,
        "score_before": state.score(),
        "score_after": state.score(),
    }
    fields.update(extra)
    return Evidence(**fields)


def _rim_entered_after(state: GameState, since_pts: float) -> bool:
    ri = state.rim_interaction
    return ri is not None and ri.entered_pts is not None and ri.entered_pts > since_pts


def _held_by(state: GameState, team: str, since_pts: float, obs: Observation) -> bool:
    h = state.holder
    return (
        h is not None
        and h.team == team
        and h.since_pts > since_pts
        and obs.t_pts - h.since_pts >= HOLD_CONFIRM_S
    )


# -- confirmers -------------------------------------------------------------------------


def always_confirm() -> Confirmer:
    def confirm(draft: EventDraft, state: GameState, obs: Observation) -> Verdict:
        return "confirmed"

    return confirm


def team_holds(team: Team, hold_s: float = HOLD_CONFIRM_S) -> Confirmer:
    """Confirmed once ``team`` has held possession for ``hold_s``; retracted if the other
    team takes it first."""

    def confirm(draft: EventDraft, state: GameState, obs: Observation) -> Verdict:
        if state.possession == team and state.possession_since_pts is not None:
            if obs.t_pts - state.possession_since_pts >= hold_s:
                return "confirmed"
            return "pending"
        if state.possession == other(team):
            return "retracted:possession_flipped"
        return "pending"

    return confirm


def team_holds_unless_attempt(team: Team, lost_team: Team, window: tuple[float, float]) -> Confirmer:
    """``team_holds`` that retracts when an attempt by ``lost_team`` shows up inside ``window``."""
    inner = team_holds(team)

    def confirm(draft: EventDraft, state: GameState, obs: Observation) -> Verdict:
        shot = state.last_shot
        if shot is not None and shot.team == lost_team and window[0] <= shot.t_pts <= window[1]:
            return "retracted:attempt_found"
        return inner(draft, state, obs)

    return confirm


def rebound_holds(team: Team, shot_team: Team, score_at_shot: Score) -> Confirmer:
    """``team_holds`` that retracts when the shot turns out to have been made."""
    inner = team_holds(team)

    def confirm(draft: EventDraft, state: GameState, obs: Observation) -> Verdict:
        if team_delta(shot_team, score_at_shot, state.score()) > 0:
            return "retracted:late_delta"
        return inner(draft, state, obs)

    return confirm


def possession_changes_within(from_team: Team, since_pts: float, window_s: float) -> Confirmer:
    def confirm(draft: EventDraft, state: GameState, obs: Observation) -> Verdict:
        if (
            state.possession in ("home", "away")
            and state.possession != from_team
            and state.possession_since_pts is not None
            and state.possession_since_pts > since_pts
        ):
            return "confirmed"
        if obs.t_pts - since_pts > window_s:
            return "retracted:no_change"
        return "pending"

    return confirm


def rim_or_delta(team: Team, since_pts: float, score_before: Score) -> Confirmer:
    """A shot attempt: confirmed by the ball reaching the rim region or a 2 or 3 delta;
    retracted when the same team re-establishes a holder with no rim entry."""

    def confirm(draft: EventDraft, state: GameState, obs: Observation) -> Verdict:
        if _rim_entered_after(state, since_pts):
            return "confirmed"
        if team_delta(team, score_before, state.score()) in (2, 3):
            return "confirmed"
        if _held_by(state, team, since_pts, obs):
            return "retracted:same_team_holds"
        return "pending"

    return confirm


def delta_within(
    team: Team, points: tuple[int, ...], score_before: Score, confirmed_confidence: float | None = None
) -> Confirmer:
    """Confirmed by a delta in ``points`` (the terminal copy takes ``confirmed_confidence``
    when given, since two signals agree); a 2 or 3 outside ``points`` retracts."""

    def confirm(draft: EventDraft, state: GameState, obs: Observation) -> Verdict:
        d = team_delta(team, score_before, state.score())
        if d in points:
            return "confirmed" if confirmed_confidence is None else f"confirmed:{confirmed_confidence}"
        if d in (2, 3):
            return "retracted:points_disagree"
        return "pending"

    return confirm


def holder_or_late_delta(team: Team, since_pts: float, score_before: Score) -> Confirmer:
    """A miss: confirmed by the next holder, retracted by a late delta for the shooters."""

    def confirm(draft: EventDraft, state: GameState, obs: Observation) -> Verdict:
        if team_delta(team, score_before, state.score()) > 0:
            return "retracted:late_delta"
        h = state.holder
        if h is not None and h.since_pts > since_pts:
            return "confirmed"
        return "pending"

    return confirm


def clock_stopped_until(deadline_pts: float) -> Confirmer:
    def confirm(draft: EventDraft, state: GameState, obs: Observation) -> Verdict:
        if state.clock_running:
            return "retracted:clock_running"
        if obs.t_pts >= deadline_pts:
            return "confirmed"
        return "pending"

    return confirm


def possession_within(since_pts: float, window_s: float = 5.0) -> Confirmer:
    def confirm(draft: EventDraft, state: GameState, obs: Observation) -> Verdict:
        if (
            state.possession in ("home", "away")
            and state.possession_since_pts is not None
            and state.possession_since_pts > since_pts
            and obs.t_pts - state.possession_since_pts >= HOLD_CONFIRM_S
        ):
            return "confirmed"
        if obs.t_pts - since_pts > 2 * window_s:
            return "retracted:no_possession"
        return "pending"

    return confirm


def ft_resolves(team: Team, since_pts: float, score_before: Score) -> Confirmer:
    def confirm(draft: EventDraft, state: GameState, obs: Observation) -> Verdict:
        if _rim_entered_after(state, since_pts) or team_delta(team, score_before, state.score()) == 1:
            return "confirmed"
        return "pending"

    return confirm
