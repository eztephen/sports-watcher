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
S = {1: ("home", (22.0, 7.5)), 2: ("away", (25.0, 6.0)), 3: ("home", (18.0, 3.0))}


def _make_two(delta_at=None):
    b = PlayBuilder(Scene(L))
    b.hold(0.6, S, holder_id=1)
    b.shot(1, S, "right", outcome="make")
    b.hold(1.5, S, holder_id=2)
    t_delta = delta_at if delta_at is not None else 0.6 + 1.0 + 0.2
    return b, observations(b.frames, deltas={t_delta: (2, 0)})


def test_attempt_drafted_at_release_with_zone_and_confirmed_by_rim():
    _, obs = _make_two()
    ev = drive(obs)
    att = provisional(ev, "shot_attempt")
    assert len(att) == 1 and att[0].team == "home" and att[0].zone == "two"
    assert att[0].player.track_id == 1
    assert abs(att[0].t_pts - 0.6) < 0.05 and att[0].court_xy is not None
    assert len(confirmed(ev, "shot_attempt")) == 1


def test_delta_after_make_emits_confirmed_shot_made():
    _, obs = _make_two()
    ev = drive(obs)
    made = confirmed(ev, "shot_made")
    assert len(made) == 1 and made[0].team == "home" and made[0].evidence.points == 2
    assert made[0].confidence == 0.9
    assert retracted(ev, "shot_missed") == [] and provisional(ev, "shot_missed") == []


def test_rim_crossing_without_delta_is_provisional_then_retracted_by_ttl():
    b = PlayBuilder(Scene(L))
    b.hold(0.6, S, holder_id=1)
    b.shot(1, S, "right", outcome="make")
    b.hold(5.0, S, holder_id=2)
    ev = drive(observations(b.frames))
    made = provisional(ev, "shot_made")
    assert len(made) == 1 and made[0].confidence == 0.6
    assert retracted(ev, "shot_made")[0].reason == "ttl"


def test_three_point_zone_from_shooter_feet():
    T = {1: ("home", (18.0, 7.5)), 2: ("away", (25.0, 6.0))}
    b = PlayBuilder(Scene(L))
    b.hold(0.6, T, holder_id=1)
    b.shot(1, T, "right", outcome="make")
    b.hold(0.5, T, holder_id=2)
    ev = drive(observations(b.frames, deltas={1.8: (3, 0)}))
    assert provisional(ev, "shot_attempt")[0].zone == "three"
    assert confirmed(ev, "shot_made")[0].evidence.points == 3


def test_zone_unknown_without_homography():
    b = PlayBuilder(Scene(L))
    b.hold(0.6, S, holder_id=1)
    b.shot(1, S, "right", outcome="make")
    for vf in b.frames:
        vf.homography = None
    ev = drive(observations(b.frames))
    att = provisional(ev, "shot_attempt")[0]
    assert att.zone == "unknown" and att.court_xy is None


def test_miss_then_rebound_emits_shot_missed_confirmed_by_next_holder():
    b = PlayBuilder(Scene(L))
    b.hold(0.6, S, holder_id=1)
    b.shot(1, S, "right", outcome="miss")
    b.hold(4.0, S, holder_id=2)
    ev = drive(observations(b.frames))
    missed = confirmed(ev, "shot_missed")
    assert len(missed) == 1 and missed[0].team == "home"


def test_shot_during_frozen_clock_is_not_drafted():
    b = PlayBuilder(Scene(L))
    b.hold(0.6, S, holder_id=1)
    b.shot(1, S, "right", outcome="make")
    b.hold(1.0, S, holder_id=2)
    ev = drive(observations(b.frames, running=False))
    assert provisional(ev, "shot_attempt") == [] and provisional(ev, "shot_made") == []


def test_pass_away_from_rim_is_not_a_shot():
    b = PlayBuilder(Scene(L))
    b.hold(0.6, S, holder_id=1)
    b.pass_(1, 3, S)
    b.hold(1.0, S, holder_id=3)
    assert provisional(drive(observations(b.frames)), "shot_attempt") == []


def test_lob_caught_by_teammate_retracts_attempt():
    U = {1: ("home", (22.0, 7.5)), 3: ("home", (25.5, 7.0)), 2: ("away", (12.0, 6.0))}
    b = PlayBuilder(Scene(L))
    b.hold(0.6, U, holder_id=1)
    b.lob(1, 3, U, "right")
    b.hold(1.0, U, holder_id=3)
    ev = drive(observations(b.frames))
    assert len(provisional(ev, "shot_attempt")) == 1
    assert retracted(ev, "shot_attempt")[0].reason in ("same_team_holds", "ttl")


def test_delta_after_lost_ball_confirms_made_without_duplicate():
    b = PlayBuilder(Scene(L))
    b.hold(0.6, S, holder_id=1)
    flight = b.shot(1, S, "right", outcome="make")
    for vf in flight[6:]:
        vf.ball = None  # ball lost after the fit window
    b.hold(3.0, S, holder_id=2)
    ev = drive(observations(b.frames, deltas={0.6 + 2.5: (2, 0)}))
    assert len(provisional(ev, "shot_attempt")) == 1 and len(confirmed(ev, "shot_made")) == 1
    assert all(e.evidence.note != "retroactive" for e in ev)


def test_delta_without_attempt_emits_retroactive_attempt():
    b = PlayBuilder(Scene(L))
    b.hold(3.0, S, holder_id=2)
    ev = drive(observations(b.frames, deltas={2.0: (0, 3)}))
    att = confirmed(ev, "shot_attempt")
    assert len(att) == 1 and att[0].confidence == 0.4 and att[0].zone == "three"
    assert att[0].team == "away"
    assert att[0].court_xy is None and att[0].evidence.note == "retroactive"
    assert confirmed(ev, "shot_made")[0].confidence == 0.4
