from datetime import timedelta

from sports_watcher.config import load_league
from sports_watcher.events.firsts import fold_firsts
from sports_watcher.events.model import Evidence
from tests.helpers.events import NOW, make_event

PBA = load_league("pba")


class Builder:
    """Emits provisional+confirmed pairs (or provisional+retracted) like the engine does."""

    def __init__(self) -> None:
        self.events = []
        self.seq = 0

    def _next(self):
        self.seq += 1
        return self.seq

    def period_start(self, period: int, t: float):
        prov = make_event(seq=self._next(), type="period_start", team="unknown", t_pts=t, period=period,
                          evidence=Evidence(note=f"period {period}"))
        self.events += [prov, prov.confirmed_copy(self._next(), NOW)]

    def score(self, t: float, team: str, points: int, after: tuple[int, int], period: int = 1, retract=False):
        before = (after[0] - points, after[1]) if team == "home" else (after[0], after[1] - points)
        prov = make_event(seq=self._next(), type="score_change", team=team, t_pts=t, period=period,
                          evidence=Evidence(score_before=before, score_after=after, points=points,
                                            note="correction" if points < 0 else None))
        self.events.append(prov)
        if retract:
            self.events.append(prov.retracted_copy(self._next(), NOW, reason="ttl"))
        else:
            self.events.append(prov.confirmed_copy(self._next(), NOW + timedelta(seconds=t + 0.4)))
        return prov


def test_first_score_and_race_to_from_confirmed_events():
    b = Builder()
    b.period_start(1, 0.0)
    b.score(1, "home", 2, (2, 0))
    b.score(2, "away", 3, (2, 3))
    b.score(3, "home", 1, (3, 3))
    b.score(4, "away", 2, (3, 5))
    b.score(5, "home", 3, (6, 5))
    home = 6
    for i in range(10):
        home += 2
        b.score(6 + i, "home", 2, (home, 5))
    f = fold_firsts(b.events, PBA)
    assert f.first_score.team == "home" and f.first_score.kind == "two" and f.first_score.t_pts == 1
    assert f.first_three.team == "away" and f.first_three.t_pts == 2
    assert f.race_to[1][10] == "home" and f.race_to[1][20] == "home"
    assert f.first_possession is None and f.tip_off_winner is None and f.possessions == []


def test_retracted_and_provisional_scores_are_ignored():
    b = Builder()
    b.score(1, "away", 3, (0, 3), retract=True)
    pending = make_event(seq=99, type="score_change", team="home", t_pts=5,
                         evidence=Evidence(score_before=(0, 3), score_after=(2, 3), points=2))
    f = fold_firsts(b.events + [pending], PBA)
    assert f.first_score is None and f.first_three is None and f.race_to == {}


def test_correction_does_not_become_first_score_but_adjusts_totals():
    b = Builder()
    b.score(1, "home", 2, (2, 0))
    b.score(2, "home", -2, (0, 0))
    b.score(3, "away", 2, (0, 2))
    for i in range(4):
        b.score(4 + i, "home", 2, (2 + 2 * i, 2))  # home: 0 -> 8
    f = fold_firsts(b.events, PBA)
    assert f.first_score.team == "home"
    assert f.race_to[1][10] is None  # home total is 8 after the correction, away 2


def test_race_to_resets_per_period_and_kind_from_points():
    b = Builder()
    b.period_start(1, 0.0)
    for i in range(6):
        b.score(1 + i, "home", 2, (2 + 2 * i, 0))  # home 12 in period 1
    b.period_start(2, 100.0)
    b.score(101, "away", 1, (12, 1), period=2)
    b.score(102, "away", 3, (12, 4), period=2)
    f = fold_firsts(b.events, PBA)
    assert f.race_to[1][10] == "home" and f.race_to[1][20] is None
    assert f.race_to[2] == {10: None, 20: None}
    assert f.first_score.kind == "two" and f.first_three.team == "away" and f.first_three.t_pts == 102


def _c(seq, type, team, t, **kw):
    from sports_watcher.events.model import EventStatus

    return make_event(seq=seq, type=type, team=team, t_pts=t, status=EventStatus.confirmed, **kw)


def test_vision_firsts_and_possessions():
    ev = [
        _c(1, "tip_off", "unknown", 1.0), _c(2, "possession_start", "away", 2.0),
        _c(3, "shot_attempt", "away", 8.0), _c(4, "shot_missed", "away", 9.0),
        _c(5, "possession_start", "home", 10.0), _c(6, "turnover", "home", 15.0),
        _c(7, "possession_start", "away", 15.0), _c(8, "shot_made", "away", 20.0),
        _c(9, "possession_start", "home", 21.0), _c(10, "period_end", "unknown", 30.0),
        _c(11, "possession_start", "away", 40.0, period=2),
    ]
    f = fold_firsts(ev, PBA)
    assert f.first_possession == "away" and f.tip_off_winner == "away"
    assert f.first_shot_attempt.team == "away" and f.first_shot_attempt.t_pts == 8.0
    assert [(p.team, p.result) for p in f.possessions] == [
        ("away", "miss"), ("home", "turnover"), ("away", "score"), ("home", "end_period"),
    ]


def test_tip_off_winner_needs_possession_within_five_seconds():
    ev = [_c(1, "tip_off", "unknown", 1.0), _c(2, "possession_start", "home", 7.0)]
    f = fold_firsts(ev, PBA)
    assert f.tip_off_winner is None and f.first_possession == "home"
