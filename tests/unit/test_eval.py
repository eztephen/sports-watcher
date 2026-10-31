from datetime import timedelta

import pytest

from sports_watcher.eval.score import EvalReport, match_events, score
from sports_watcher.events.firsts import FirstRef, Firsts, FirstScore
from sports_watcher.events.model import Evidence
from sports_watcher.replay.truth import TruthEvent
from tests.helpers.events import NOW, make_event

_seq = 0


def _next():
    global _seq
    _seq += 1
    return _seq


def confirmed(type, t_pts, team="home", points=None, emitted_delay_s=0.3):
    t_wall = NOW + timedelta(seconds=t_pts)
    prov = make_event(seq=_next(), type=type, team=team, t_pts=t_pts, t_wall=t_wall, emitted_wall=t_wall,
                      evidence=Evidence(points=points))
    conf = prov.confirmed_copy(_next(), emitted_wall=t_wall + timedelta(seconds=emitted_delay_s))
    return [prov, conf]


def provisional_then_retracted(type, t_pts, team="home"):
    t_wall = NOW + timedelta(seconds=t_pts)
    prov = make_event(seq=_next(), type=type, team=team, t_pts=t_pts, t_wall=t_wall, emitted_wall=t_wall)
    return [prov, prov.retracted_copy(_next(), emitted_wall=t_wall + timedelta(seconds=4), reason="ttl")]


def test_score_precision_recall_and_latency():
    truth = [
        TruthEvent(t_pts=10.0, type="score_change", team="home", points=2),
        TruthEvent(t_pts=30.0, type="score_change", team="away", points=3),
    ]
    events = (
        confirmed("score_change", 10.4, "home", 2, 0.3)
        + confirmed("score_change", 55.0, "home", 2, 0.2)
        + provisional_then_retracted("score_change", 40.0)
    )
    r = score(events, truth)
    s = r.per_type["score_change"]
    assert (s.tp, s.fp, s.fn) == (1, 1, 1) and s.precision == 0.5 and s.recall == 0.5
    assert s.latency_p50_s == pytest.approx(0.3) and s.latency_p95_s == pytest.approx(0.3)
    assert r.retraction_rate == pytest.approx(1 / 3) and r.firsts_accuracy is None
    assert EvalReport.model_validate_json(r.model_dump_json()) == r


def test_match_requires_same_type_team_and_tolerance():
    truth = [TruthEvent(t_pts=10.0, type="score_change", team="home")]
    wrong_team = confirmed("score_change", 10.2, "away")
    too_late = confirmed("score_change", 11.5, "home")
    wrong_type = confirmed("period_end", 10.0, "unknown")
    pairs = match_events(wrong_team + too_late + wrong_type, truth, tolerance_s=1.0)
    assert pairs == [(truth[0], None)]
    s = score(wrong_team + too_late + wrong_type, truth).per_type["score_change"]
    assert (s.tp, s.fp, s.fn) == (0, 2, 1) and s.latency_p50_s is None


def test_each_event_matches_at_most_once_and_nearest_wins():
    truth = [TruthEvent(t_pts=10.0, type="score_change", team="home"), TruthEvent(t_pts=10.6, type="score_change", team="home")]
    events = confirmed("score_change", 10.5, "home")
    pairs = dict(match_events(events, truth))
    assert pairs[truth[1]] is not None and pairs[truth[0]] is None


def test_firsts_accuracy_compares_dotted_paths():
    firsts = Firsts(
        first_score=FirstScore(team="home", t_pts=1, seq=1, kind="two"),
        first_three=FirstRef(team="away", t_pts=2, seq=3),
    )
    truth_firsts = {"first_score.team": "home", "first_score.kind": "three", "first_three.team": "away"}
    r = score([], [], firsts=firsts, truth_firsts=truth_firsts)
    assert r.firsts_accuracy == pytest.approx(2 / 3) and r.per_type == {}
