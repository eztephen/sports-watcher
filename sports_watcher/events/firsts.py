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
TIP_OFF_WINDOW_S = 5.0
SCORING_TYPES = ("shot_made", "free_throw_made")
MISS_TYPES = ("shot_missed", "free_throw_missed")
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
    _fold_vision(firsts, events)
    return firsts


def _fold_vision(firsts: Firsts, events: Iterable[Event]) -> None:
    """Possession, tip-off and first-attempt folds over confirmed events in pts order."""
    by_time = sorted((e for e in events if e.status == EventStatus.confirmed), key=lambda e: (e.t_pts, e.seq))
    starts = [e for e in by_time if e.type == "possession_start" and e.team in ("home", "away")]
    if starts:
        firsts.first_possession = starts[0].team
    tip = next((e for e in by_time if e.type == "tip_off"), None)
    if tip is not None:
        winner = next((e for e in starts if tip.t_pts <= e.t_pts <= tip.t_pts + TIP_OFF_WINDOW_S), None)
        firsts.tip_off_winner = winner.team if winner is not None else None
    attempt = next((e for e in by_time if e.type == "shot_attempt"), None)
    if attempt is not None:
        firsts.first_shot_attempt = FirstRef(
            team=attempt.team, player=attempt.player, t_pts=attempt.t_pts, seq=attempt.seq
        )
    possessions: list[PossessionResult] = []
    start_seqs = {e.seq for e in starts}
    for i, start in enumerate(starts):
        # A turnover is stamped at the instant the next possession begins, so the window
        # closes inclusively; the possession_start events themselves never count.
        end = starts[i + 1].t_pts if i + 1 < len(starts) else float("inf")
        window = [e for e in by_time if start.t_pts <= e.t_pts <= end and e.seq not in start_seqs]
        result = _possession_result(start.team, window)
        if result is None:
            continue  # still open: no later possession and nothing decided it
        possessions.append(PossessionResult(period=start.period, team=start.team, result=result, t_pts=start.t_pts))
    firsts.possessions = possessions


def _possession_result(team: str, window: list[Event]) -> str | None:
    if any(e.type in SCORING_TYPES and e.team == team for e in window):
        return "score"
    if any(e.type == "turnover" and e.team == team for e in window):
        return "turnover"
    if any(e.type in MISS_TYPES and e.team == team for e in window):
        return "miss"
    if any(e.type == "period_end" for e in window):
        return "end_period"
    return None
