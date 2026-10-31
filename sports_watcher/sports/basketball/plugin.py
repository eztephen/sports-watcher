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


class BasketballPlugin:
    name = "basketball"

    def __init__(self, league: LeagueConfig) -> None:
        self.league = league
        self._rules: list[Rule] = [ScoreChangeRule(), PeriodStartRule(league), PeriodEndRule()]

    def update_state(self, state: GameState, obs: Observation) -> GameState:
        new = replace(state, t_pts=obs.t_pts, camera_state=obs.camera_state)
        sb = obs.scoreboard
        if sb is None or sb.confidence < MIN_SCOREBOARD_CONFIDENCE:
            return new
        if sb.home is not None:
            new.home_score = sb.home
        if sb.away is not None:
            new.away_score = sb.away
        if sb.clock_s is not None:
            new.clock_s = sb.clock_s
        if sb.period is not None:
            new.period = sb.period
        new.shot_clock_s = sb.shot_clock_s
        new.scoreboard_confidence = sb.confidence
        new.has_scoreboard = new.has_scoreboard or (sb.home is not None and sb.away is not None)
        return new

    def rules(self) -> list[Rule]:
        return self._rules
