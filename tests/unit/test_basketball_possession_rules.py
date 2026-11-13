from tests.helpers.leagues import fixture_league
from tests.helpers.vision import (
    PlayBuilder,
    Scene,
    confirmed,
    drive,
    observations,
    provisional,
    retracted,
)

L = fixture_league("pba")
P = {1: ("home", (10.0, 7.0)), 2: ("away", (12.0, 8.0)), 3: ("home", (16.0, 5.0))}
S = {1: ("home", (22.0, 7.5)), 2: ("away", (25.0, 6.0))}


def test_first_possession_is_drafted_then_confirmed_after_half_a_second():
    b = PlayBuilder(Scene(L))
    b.hold(1.0, P, holder_id=1)
    ev = drive(observations(b.frames))
    prov = provisional(ev, "possession_start")
    assert len(prov) == 1 and prov[0].team == "home" and prov[0].t_pts == 0.0
    assert prov[0].player.track_id == 1
    assert prov[0].confidence == 0.7 and prov[0].evidence.clock_state == "read"
    conf = confirmed(ev, "possession_start")
    assert len(conf) == 1 and conf[0].supersedes == prov[0].id and conf[0].t_pts == 0.0


def test_pass_to_teammate_is_not_a_new_possession():
    b = PlayBuilder(Scene(L))
    b.hold(0.6, P, holder_id=1)
    b.pass_(1, 3, P)
    b.hold(0.6, P, holder_id=3)
    ev = drive(observations(b.frames))
    assert len(provisional(ev, "possession_start")) == 1


def test_steal_flipping_back_within_half_a_second_is_retracted():
    b = PlayBuilder(Scene(L))
    b.hold(0.6, P, holder_id=1)
    b.hold(0.3, P, holder_id=2)
    b.hold(1.0, P, holder_id=1)
    ev = drive(observations(b.frames))
    away = [e for e in provisional(ev, "possession_start") if e.team == "away"]
    assert len(away) == 1 and len(retracted(ev, "possession_start")) == 1
    assert retracted(ev, "possession_start")[0].supersedes == away[0].id


def test_same_team_after_closeup_is_not_a_new_possession():
    b = PlayBuilder(Scene(L))
    b.hold(0.6, P, holder_id=1)
    for _ in range(30):
        b.frame(P, None, camera_hint="closeup")
    b.hold(0.6, P, holder_id=3)
    ev = drive(observations(b.frames))
    assert len(provisional(ev, "possession_start")) == 1
    b.hold(0.6, P, holder_id=2)
    ev = drive(observations(b.frames))
    assert [e.team for e in provisional(ev, "possession_start")] == ["home", "away"]


def test_no_possession_while_clock_is_stopped():
    b = PlayBuilder(Scene(L))
    b.hold(1.0, P, holder_id=1)
    assert provisional(drive(observations(b.frames, running=False)), "possession_start") == []


def test_assumed_clock_lowers_confidence():
    b = PlayBuilder(Scene(L))
    b.hold(4.0, P, holder_id=1)
    b.hold(1.0, P, holder_id=2)
    ev = drive(observations(b.frames, hidden=[(0.5, 5.0)]))
    away = next(e for e in provisional(ev, "possession_start") if e.team == "away")
    assert away.evidence.clock_state == "assumed" and away.confidence == 0.5


def test_unknown_team_holder_drafts_nothing():
    U = {1: ("unknown", (10.0, 7.0)), 2: ("away", (12.0, 8.0))}
    b = PlayBuilder(Scene(L))
    b.hold(1.0, U, holder_id=1)
    assert provisional(drive(observations(b.frames)), "possession_start") == []


def test_rebound_after_miss_is_defensive_for_the_other_team():
    b = PlayBuilder(Scene(L))
    b.hold(0.6, S, holder_id=1)
    b.shot(1, S, "right", outcome="miss")
    b.hold(4.0, S, holder_id=2)  # confirmation waits out the 3 s scoreboard window
    ev = drive(observations(b.frames))
    reb = confirmed(ev, "rebound")
    assert len(reb) == 1 and reb[0].team == "away" and reb[0].evidence.note == "defensive"


def test_turnover_when_possession_changes_without_an_attempt():
    b = PlayBuilder(Scene(L))
    b.hold(0.6, P, holder_id=1)
    b.hold(4.0, P, holder_id=2)  # confirmation waits out the 3 s scoreboard window
    ev = drive(observations(b.frames))
    to = confirmed(ev, "turnover")
    assert len(to) == 1 and to[0].team == "home"
    assert abs(to[0].t_pts - 0.6) < 0.05  # the away hold began at 0.6


def test_no_turnover_when_the_change_follows_a_shot():
    b = PlayBuilder(Scene(L))
    b.hold(0.6, S, holder_id=1)
    b.shot(1, S, "right", outcome="miss")
    b.hold(1.0, S, holder_id=2)
    assert provisional(drive(observations(b.frames)), "turnover") == []


def test_possession_held_through_a_dead_ball_is_not_drafted_at_the_restart():
    b = PlayBuilder(Scene(L))
    b.hold(6.0, P, holder_id=1)
    assert provisional(drive(observations(b.frames, running=False, resume_at=5.0)), "possession_start") == []


def test_possession_taken_just_before_the_restart_is_drafted():
    b = PlayBuilder(Scene(L))
    b.hold(2.0, P, holder_id=1)
    ev = drive(observations(b.frames, running=False, resume_at=0.5))
    prov = provisional(ev, "possession_start")
    assert len(prov) == 1 and prov[0].t_pts == 0.0


def test_made_basket_with_a_slow_delta_has_no_confirmed_rebound():
    b = PlayBuilder(Scene(L))
    b.hold(0.6, S, holder_id=1)
    b.shot(1, S, "right", outcome="make")
    b.hold(3.0, S, holder_id=2)
    ev = drive(observations(b.frames, deltas={3.0: (2, 0)}))
    assert confirmed(ev, "rebound") == [] and len(confirmed(ev, "shot_made")) == 1


def test_turnover_is_retracted_when_the_losing_team_scores():
    b = PlayBuilder(Scene(L))
    b.hold(0.6, S, holder_id=1)
    b.hold(3.5, S, holder_id=2)
    ev = drive(observations(b.frames, deltas={2.5: (2, 0)}))
    assert confirmed(ev, "turnover") == [] and len(retracted(ev, "turnover")) == 1


def test_adjacent_defender_does_not_take_possession():
    D = {1: ("home", (22.0, 7.5)), 2: ("away", (22.5, 7.5))}
    b = PlayBuilder(Scene(L))
    b.hold(1.0, D, holder_id=1)
    for _ in range(30):
        b.frame(D, ((22.3, 7.5), 1.0))
    b.hold(1.0, D, holder_id=1)
    ev = drive(observations(b.frames))
    assert provisional(ev, "turnover") == [] and len(provisional(ev, "possession_start")) == 1


def test_new_hold_at_a_dead_ball_restart_starts_the_possession():
    # Home has the ball through a stopped clock (free throws, an inbound); a teammate
    # receives it just before the clock restarts at 5 s.
    b = PlayBuilder(Scene(L))
    b.hold(4.0, P, holder_id=3)
    b.pass_(3, 1, P, flight_s=0.4)
    received_at = b.t
    b.hold(2.0, P, holder_id=1)
    ev = drive(observations(b.frames, running=False, resume_at=5.0))
    prov = provisional(ev, "possession_start")
    assert len(prov) == 1 and prov[0].team == "home"
    assert abs(prov[0].t_pts - received_at) < 0.2
    assert len(confirmed(ev, "possession_start")) == 1


def test_new_hold_after_a_hidden_scoreboard_is_not_a_possession_start():
    # The clock ran on behind a replay graphic: the same possession continues.
    b = PlayBuilder(Scene(L))
    b.hold(3.0, P, holder_id=3)
    for _ in range(round(9.0 * b.fps)):
        b.frame(P, None, camera_hint="replay")
    b.hold(2.0, P, holder_id=1)
    ev = drive(observations(b.frames, hidden=[(3.0, 12.0)]))
    assert len(provisional(ev, "possession_start")) == 1


def test_new_hold_after_a_misread_and_a_hidden_clock_is_not_a_possession_start():
    # One wrong clock read, then the scoreboard is covered while play goes on. The first
    # tick after it returns is one step down from the last read, but not from the clock
    # value at the previous change, so the clock did not stand still.
    b = PlayBuilder(Scene(L))
    b.hold(3.0, P, holder_id=3)
    b.pass_(3, 1, P, flight_s=0.4)
    b.hold(6.0, P, holder_id=1)
    b.pass_(1, 3, P, flight_s=0.4)
    b.hold(3.0, P, holder_id=3)
    obs = observations(b.frames, hidden=[(1.5, 9.0)])
    misread = next(o for o in obs if o.scoreboard is not None and o.t_pts >= 1.2)
    misread.scoreboard.clock_s = 20.0
    assert len(provisional(drive(obs), "possession_start")) == 1


def test_brief_hold_then_a_loose_ball_is_retracted_and_the_first_team_plays_on():
    # A defender's box covers the ball for a third of a second, nobody is seen with it for
    # most of a second, then the team that had it holds it again: one possession, not three.
    loose = ((20.0, 3.0), 1.0)
    b = PlayBuilder(Scene(L))
    b.hold(1.0, P, holder_id=1)
    b.hold(0.35, P, holder_id=2)
    for _ in range(24):
        b.frame(P, loose)
    b.hold(1.5, P, holder_id=3)
    ev = drive(observations(b.frames))
    away = [e for e in provisional(ev, "possession_start") if e.team == "away"]
    assert len(away) == 1
    assert [e.supersedes for e in retracted(ev, "possession_start")] == [away[0].id]
    assert len([e for e in provisional(ev, "possession_start") if e.team == "home"]) == 1


def test_ball_returning_after_a_brief_hold_is_one_turnover_draft_not_two():
    # Same play: the defender's third of a second drafts a turnover that is retracted.
    # The ball going back is not a turnover by the defender's team, which never had it.
    loose = ((20.0, 3.0), 1.0)
    b = PlayBuilder(Scene(L))
    b.hold(1.0, P, holder_id=1)
    b.hold(0.35, P, holder_id=2)
    for _ in range(24):
        b.frame(P, loose)
    b.hold(5.0, P, holder_id=3)
    ev = drive(observations(b.frames))
    drafts = provisional(ev, "turnover")
    assert [e.team for e in drafts] == ["home"]
    assert [e.supersedes for e in retracted(ev, "turnover")] == [drafts[0].id]
    assert confirmed(ev, "turnover") == []


def test_steal_and_steal_back_after_a_real_hold_are_two_turnovers():
    b = PlayBuilder(Scene(L))
    b.hold(1.0, P, holder_id=1)
    b.hold(0.8, P, holder_id=2)
    b.hold(5.0, P, holder_id=1)
    ev = drive(observations(b.frames))
    assert [e.team for e in provisional(ev, "turnover")] == ["home", "away"]


def test_steal_and_steal_back_after_a_real_hold_are_two_possessions():
    b = PlayBuilder(Scene(L))
    b.hold(1.0, P, holder_id=1)
    b.hold(0.8, P, holder_id=2)
    b.hold(1.0, P, holder_id=1)
    ev = drive(observations(b.frames))
    assert [e.team for e in confirmed(ev, "possession_start")] == ["home", "away", "home"]
