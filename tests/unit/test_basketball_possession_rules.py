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
    b.hold(1.0, S, holder_id=2)
    ev = drive(observations(b.frames))
    reb = confirmed(ev, "rebound")
    assert len(reb) == 1 and reb[0].team == "away" and reb[0].evidence.note == "defensive"


def test_turnover_when_possession_changes_without_an_attempt():
    b = PlayBuilder(Scene(L))
    b.hold(0.6, P, holder_id=1)
    b.hold(1.0, P, holder_id=2)
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
