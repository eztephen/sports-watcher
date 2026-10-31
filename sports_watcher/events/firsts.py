"""The "first X" read model, folded from confirmed events only.

Provisional events count for nothing until confirmed; a retraction removes them. A
score correction (negative points) adjusts the running totals used for race-to markets
but never rewrites which score came first: the firsts record what was observed, and
retractions are the correction mechanism.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field

from sports_watcher.config import LeagueConfig
from sports_watcher.events.model import Event, EventStatus, PlayerRef

RACE_THRESHOLDS = (10, 20)
ScoreKind = Literal["ft", "two", "three", "unknown"]


class FirstRef(BaseModel):
    team: str
    player: PlayerRef | None = None
    t_pts: float
    seq: int


class FirstScore(FirstRef):
    kind: ScoreKind


class PossessionResult(BaseModel):
    period: int | None
    team: str
    result: str
    t_pts: float


class Firsts(BaseModel):
    first_possession: str | None = None
    tip_off_winner: str | None = None
    first_shot_attempt: FirstRef | None = None
    first_score: FirstScore | None = None
    first_three: FirstRef | None = None
    race_to: dict[int, dict[int, str | None]] = Field(default_factory=dict)
    possessions: list[PossessionResult] = Field(default_factory=list)


def kind_from_points(points: int | None) -> ScoreKind:
    return {1: "ft", 2: "two", 3: "three"}.get(points, "unknown")  # type: ignore[return-value]


def confirmed_in_order(events: Iterable[Event]) -> list[Event]:
    return sorted((e for e in events if e.status == EventStatus.confirmed), key=lambda e: e.seq)


def fold_firsts(events: Iterable[Event], league: LeagueConfig) -> Firsts:
    firsts = Firsts()
    race: dict[int, dict[int, str | None]] = {}
    totals: dict[int, list[int]] = {}

    def bucket(period: int) -> tuple[dict[int, str | None], list[int]]:
        return (
            race.setdefault(period, dict.fromkeys(RACE_THRESHOLDS)),
            totals.setdefault(period, [0, 0]),
        )

    for e in confirmed_in_order(events):
        period = e.period if e.period is not None else 1
        if e.type == "period_start":
            bucket(period)
            continue
        if e.type != "score_change":
            continue

        pts = e.evidence.points
        if pts is not None and pts > 0 and e.team in ("home", "away"):
            if firsts.first_score is None:
                firsts.first_score = FirstScore(
                    team=e.team, player=e.player, t_pts=e.t_pts, seq=e.seq, kind=kind_from_points(pts)
                )
            if pts == 3 and firsts.first_three is None:
                firsts.first_three = FirstRef(team=e.team, player=e.player, t_pts=e.t_pts, seq=e.seq)

        before, after = e.evidence.score_before, e.evidence.score_after
        if before is None or after is None:
            continue
        reached, tot = bucket(period)
        tot[0] += after[0] - before[0]
        tot[1] += after[1] - before[1]
        for threshold in RACE_THRESHOLDS:
            if reached[threshold] is not None:
                continue
            home_hit, away_hit = tot[0] >= threshold, tot[1] >= threshold
            if home_hit and away_hit:
                reached[threshold] = "unknown"
            elif home_hit:
                reached[threshold] = "home"
            elif away_hit:
                reached[threshold] = "away"

    firsts.race_to = race
    return firsts
