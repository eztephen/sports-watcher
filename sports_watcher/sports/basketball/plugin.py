"""Basketball plug-in: state updates from the scoreboard and the M0 rule set."""

from __future__ import annotations

from dataclasses import replace

from sports_watcher.config import LeagueConfig
from sports_watcher.sports.base import Observation, Rule
from sports_watcher.sports.basketball.rules import (
    MIN_SCOREBOARD_CONFIDENCE,
    PeriodEndRule,
    PeriodStartRule,
    ScoreChangeRule,
)
from sports_watcher.state.game_state import GameState

# A read where both scores changed at once is either a rejoin after a long cut or a
# graphic that happens to contain digits; only a fully agreeing window may apply it.
FULL_AGREEMENT = 1.0


class BasketballPlugin:
    name = "basketball"

    def __init__(self, league: LeagueConfig) -> None:
        self.league = league
        self._rules: list[Rule] = [ScoreChangeRule(), PeriodStartRule(league), PeriodEndRule()]

    def update_state(self, state: GameState, obs: Observation) -> GameState:
        """Apply a scoreboard read when it is confident and plausible.

        Plausibility guards (the scoreboard-only stand-in for M1's camera-state
        classifier): a read without a game clock is a graphic, not a bug, and is
        ignored; the period may only stay or advance by one; a read where both scores
        changed at once needs a fully agreeing window.
        """
        new = replace(state, t_pts=obs.t_pts, camera_state=obs.camera_state)
        sb = obs.scoreboard
        if sb is None or sb.confidence < MIN_SCOREBOARD_CONFIDENCE or sb.clock_s is None:
            return new
        if sb.period is not None and (
            state.period is None or sb.period in (state.period, state.period + 1)
        ):
            new.period = sb.period
        if sb.home is not None and sb.away is not None:
            both_changed = (
                state.has_scoreboard
                and sb.home != state.home_score
                and sb.away != state.away_score
            )
            if not both_changed or sb.confidence >= FULL_AGREEMENT:
                new.home_score, new.away_score = sb.home, sb.away
                new.has_scoreboard = True
        new.clock_s = sb.clock_s
        new.shot_clock_s = sb.shot_clock_s
        new.scoreboard_confidence = sb.confidence
        return new

    def rules(self) -> list[Rule]:
        return self._rules
