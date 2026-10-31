"""The game state the rules read. Sport-agnostic; sport plug-ins fill it in."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class GameState:
    period: int | None = None
    clock_s: float | None = None
    shot_clock_s: float | None = None
    home_score: int = 0
    away_score: int = 0
    scoreboard_confidence: float = 0.0
    camera_state: str = "main"
    possession: Literal["home", "away", "unknown"] = "unknown"
    t_pts: float = 0.0
    has_scoreboard: bool = False  # True once a confident scoreboard read has been applied

    def score(self) -> tuple[int, int]:
        return (self.home_score, self.away_score)
