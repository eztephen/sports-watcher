"""Basketball rules that need only the scoreboard (M0). Vision rules arrive in M1.

Rules are small callables ``(prev_state, new_state, observation) -> drafts``. The ones
that must remember what they already emitted are classes with per-game memory; the
plug-in creates fresh instances per game.
"""

from __future__ import annotations

from sports_watcher.config import LeagueConfig
from sports_watcher.events.model import Evidence
from sports_watcher.sports.base import Confirmer, EventDraft, Observation, Verdict
from sports_watcher.state.game_state import GameState

MIN_SCOREBOARD_CONFIDENCE = 0.6
PERIOD_END_HOLD_S = 1.0
CLOCK_ZERO_EPS = 0.05


def scoreboard_agrees(field: str, expected, reads: int = 2) -> Confirmer:
    """Confirm after ``reads`` fresh scoreboard observations agree with ``expected``;
    retract on the first one that disagrees. Observations without a confident read
    do not vote."""
    seen = 0

    def confirm(draft: EventDraft, state: GameState, obs: Observation) -> Verdict:
        nonlocal seen
        if obs.scoreboard is None or obs.scoreboard.confidence < MIN_SCOREBOARD_CONFIDENCE:
            return "pending"
        value = state.score() if field == "score" else getattr(state, field)
        if value != expected:
            return "retracted"
        seen += 1
        return "confirmed" if seen >= reads else "pending"

    return confirm


def full_length_s(league: LeagueConfig, period: int) -> int:
    return league.period_length_s if period <= league.periods else league.overtime_length_s


class ScoreChangeRule:
    name = "score_change"

    def __call__(self, prev: GameState, new: GameState, obs: Observation) -> list[EventDraft]:
        if not prev.has_scoreboard or new.score() == prev.score():
            return []
        dh = new.home_score - prev.home_score
        da = new.away_score - prev.away_score
        if dh and da:
            team, points = "unknown", None
        elif dh:
            team, points = "home", dh
        else:
            team, points = "away", da
        note = "correction" if points is not None and points < 0 else None
        return [
            EventDraft(
                type="score_change",
                team=team,
                t_pts=obs.t_pts,
                evidence=Evidence(
                    score_before=prev.score(), score_after=new.score(), points=points, note=note
                ),
                confidence=new.scoreboard_confidence,
                confirmer=scoreboard_agrees("score", new.score(), reads=2),
            )
        ]


class PeriodStartRule:
    """Fires once per period: on the first read of a period whose clock is at full
    length, on a period increment, or on a clock reset to full length."""

    name = "period_start"

    def __init__(self, league: LeagueConfig) -> None:
        self._league = league
        self._fired: set[int] = set()

    def __call__(self, prev: GameState, new: GameState, obs: Observation) -> list[EventDraft]:
        p = new.period
        if p is None or p in self._fired:
            return []
        full = full_length_s(self._league, p)
        incremented = prev.period is not None and p > prev.period
        at_full = new.clock_s is not None and abs(new.clock_s - full) < CLOCK_ZERO_EPS
        if not (incremented or at_full):
            return []
        self._fired.add(p)
        note = "overtime" if p > self._league.periods else f"period {p}"
        return [
            EventDraft(
                type="period_start",
                team="unknown",
                t_pts=obs.t_pts,
                evidence=Evidence(score_before=new.score(), score_after=new.score(), note=note),
                confidence=new.scoreboard_confidence,
                confirmer=scoreboard_agrees("period", p, reads=1),
            )
        ]


class PeriodEndRule:
    """Fires once per period when the clock has sat at zero for PERIOD_END_HOLD_S."""

    name = "period_end"

    def __init__(self) -> None:
        self._zero_since: float | None = None
        self._ended: set[int] = set()

    def __call__(self, prev: GameState, new: GameState, obs: Observation) -> list[EventDraft]:
        if new.clock_s is None or new.period is None:
            return []
        if new.clock_s > CLOCK_ZERO_EPS:
            self._zero_since = None
            return []
        if self._zero_since is None:
            self._zero_since = obs.t_pts
            return []
        if obs.t_pts - self._zero_since < PERIOD_END_HOLD_S or new.period in self._ended:
            return []
        self._ended.add(new.period)
        return [
            EventDraft(
                type="period_end",
                team="unknown",
                t_pts=obs.t_pts,
                evidence=Evidence(
                    score_before=new.score(), score_after=new.score(), note=f"period {new.period}"
                ),
                confidence=new.scoreboard_confidence,
                confirmer=scoreboard_agrees("period", new.period, reads=1),
            )
        ]
