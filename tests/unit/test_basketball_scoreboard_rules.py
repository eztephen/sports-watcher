from datetime import UTC, datetime

from sports_watcher.config import load_league
from sports_watcher.perception.scoreboard import ScoreboardState
from sports_watcher.sports.base import Observation
from sports_watcher.sports.basketball.plugin import BasketballPlugin
from sports_watcher.state.game_state import GameState

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def run(seq):
    """seq: list of (t_pts, home, away, clock_s, period) -> list[EventDraft]."""
    plugin = BasketballPlugin(load_league("pba"))
    state = GameState()
    out = []
    for i, (t, h, a, c, p) in enumerate(seq):
        sb = ScoreboardState(home=h, away=a, clock_s=c, period=p, confidence=1.0, raw=[])
        obs = Observation(frame_index=i, t_pts=t, t_wall=NOW, scoreboard=sb)
        new = plugin.update_state(state, obs)
        for rule in plugin.rules():
            out += rule(state, new, obs)
        state = new
    return out


def test_home_two_points_emits_score_change():
    d = run([(0, 0, 0, 600, 1), (1, 2, 0, 598, 1)])
    assert [x.type for x in d] == ["period_start", "score_change"]
    sc = d[-1]
    assert sc.team == "home" and sc.evidence.points == 2
    assert sc.evidence.score_before == (0, 0) and sc.evidence.score_after == (2, 0)
    assert sc.t_pts == 1 and sc.confidence == 1.0 and sc.ttl_s == 4.0


def test_away_three_points_emits_score_change():
    d = run([(0, 10, 8, 500, 2), (1, 10, 11, 499, 2)])
    sc = next(x for x in d if x.type == "score_change")
    assert sc.team == "away" and sc.evidence.points == 3


def test_negative_delta_is_a_correction():
    sc = next(x for x in run([(0, 10, 8, 500, 2), (1, 8, 8, 499, 2)]) if x.type == "score_change")
    assert sc.evidence.points == -2 and sc.evidence.note == "correction" and sc.team == "home"


def test_period_boundaries_including_overtime():
    d = run([(0, 50, 50, 0.0, 4), (2, 50, 50, 0.0, 4), (3, 50, 50, 300.0, 5)])
    assert [x.type for x in d] == ["period_end", "period_start"]
    assert d[0].evidence.note == "period 4" and d[-1].evidence.note == "overtime"


def test_period_start_fires_once_per_period_and_on_increment():
    d = run([(0, 0, 0, 600, 1), (1, 0, 0, 599, 1), (2, 0, 0, 600, 1), (600, 20, 18, 600, 2)])
    starts = [(x.type, x.evidence.note) for x in d if x.type == "period_start"]
    assert starts == [("period_start", "period 1"), ("period_start", "period 2")]
    assert [x.team for x in d if x.type == "score_change"] == ["unknown"]  # both sides jumped


def test_low_confidence_scoreboard_does_not_change_state():
    plugin = BasketballPlugin(load_league("pba"))
    state = GameState(home_score=10, away_score=8, period=2)
    sb = ScoreboardState(home=99, away=8, clock_s=400, period=2, confidence=0.4, raw=[])
    new = plugin.update_state(state, Observation(frame_index=0, t_pts=5.0, t_wall=NOW, scoreboard=sb))
    assert new.score() == (10, 8) and new.t_pts == 5.0


def test_scoreboard_agrees_confirmer():
    from sports_watcher.sports.basketball.rules import scoreboard_agrees

    confirmer = scoreboard_agrees("score", (2, 0), reads=2)
    draft = run([(0, 0, 0, 600, 1), (1, 2, 0, 598, 1)])[-1]
    agreeing = GameState(home_score=2, away_score=0, period=1)
    disagreeing = GameState(home_score=0, away_score=0, period=1)
    read = ScoreboardState(home=2, away=0, clock_s=597, period=1, confidence=1.0, raw=[])
    with_read = Observation(frame_index=0, t_pts=2.0, t_wall=NOW, scoreboard=read)
    no_read = Observation(frame_index=1, t_pts=2.1, t_wall=NOW)
    assert confirmer(draft, agreeing, with_read) == "pending"
    assert confirmer(draft, agreeing, no_read) == "pending"  # no fresh read, no vote
    assert confirmer(draft, agreeing, with_read) == "confirmed"
    fresh = scoreboard_agrees("score", (2, 0), reads=2)
    assert fresh(draft, disagreeing, with_read) == "retracted"


# --- plausibility guards: a graphic or a torn frame must not move the game state ---------


def _obs(t, h, a, c, p, conf=1.0, i=0):
    sb = ScoreboardState(home=h, away=a, clock_s=c, period=p, confidence=conf, raw=[])
    return Observation(frame_index=i, t_pts=t, t_wall=NOW, scoreboard=sb)


def test_read_without_a_clock_does_not_change_scores():
    plugin = BasketballPlugin(load_league("pba"))
    state = plugin.update_state(GameState(), _obs(0, 13, 13, 379.0, 1))
    graphic = plugin.update_state(state, _obs(1, 1, 1, None, None, conf=1.0))
    assert graphic.score() == (13, 13) and graphic.period == 1


def test_both_scores_jumping_needs_full_agreement():
    plugin = BasketballPlugin(load_league("pba"))
    state = plugin.update_state(GameState(), _obs(0, 13, 13, 379.0, 1))
    weak = plugin.update_state(state, _obs(1, 1, 1, 300.0, 1, conf=0.6))
    assert weak.score() == (13, 13)
    strong = plugin.update_state(state, _obs(2, 40, 38, 100.0, 2, conf=1.0))  # rejoined after a long cut
    assert strong.score() == (40, 38)


def test_single_side_change_is_applied_at_normal_confidence():
    plugin = BasketballPlugin(load_league("pba"))
    state = plugin.update_state(GameState(), _obs(0, 13, 13, 379.0, 1))
    new = plugin.update_state(state, _obs(1, 13, 15, 360.0, 1, conf=0.6))
    assert new.score() == (13, 15)


def test_period_can_only_advance_by_one():
    plugin = BasketballPlugin(load_league("pba"))
    state = plugin.update_state(GameState(), _obs(0, 13, 13, 379.0, 1))
    bogus = plugin.update_state(state, _obs(1, 13, 13, 379.0, 5, conf=1.0))
    assert bogus.period == 1
    nxt = plugin.update_state(state, _obs(2, 13, 13, 600.0, 2, conf=1.0))
    assert nxt.period == 2
    rejoin = plugin.update_state(GameState(), _obs(3, 50, 48, 120.0, 4, conf=1.0))  # no prior period: accept
    assert rejoin.period == 4
