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
P = {1: ("home", (10.0, 7.0)), 2: ("away", (12.0, 8.0))}
LANE = {
    1: ("home", (5.8, 7.5)), 2: ("away", (3.0, 9.95)), 3: ("home", (3.0, 5.05)),
    4: ("away", (4.5, 10.1)), 5: ("home", (4.5, 4.9)), 6: ("away", (10.0, 7.5)),
}
C = {1: ("home", (13.5, 7.5)), 2: ("away", (14.5, 7.5)), 3: ("home", (16.0, 5.0))}


def test_timeout_drafted_at_twenty_seconds_and_confirmed_at_thirty():
    b = PlayBuilder(Scene(L))
    b.hold(0.5, P, holder_id=1)
    for _ in range(31 * 30):
        b.frame(P, None, camera_hint="closeup")
    ev = drive(observations(b.frames, running=False))
    to = provisional(ev, "timeout")
    assert len(to) == 1 and abs(to[0].t_pts - 0.5) < 0.05 and to[0].team == "unknown"
    assert len(confirmed(ev, "timeout")) == 1


def test_timeout_retracted_when_play_resumes():
    b = PlayBuilder(Scene(L))
    b.hold(0.5, P, holder_id=1)
    for _ in range(22 * 30):
        b.frame(P, None, camera_hint="closeup")
    b.hold(3.0, P, holder_id=1)
    ev = drive(observations(b.frames, running=False, resume_at=22.5))  # drafted at 20.5, clock runs before 30.5
    assert len(retracted(ev, "timeout")) == 1 and confirmed(ev, "timeout") == []


def test_no_timeout_while_clock_runs_during_a_closeup():
    b = PlayBuilder(Scene(L))
    for _ in range(25 * 30):
        b.frame(P, None, camera_hint="closeup")
    assert provisional(drive(observations(b.frames, running=True)), "timeout") == []


def test_tip_off_confirmed_by_first_possession():
    b = PlayBuilder(Scene(L))
    for i in range(6):
        b.frame(C, ((14.0, 7.5), 2.0 + 0.3 * i))  # ball rising at centre
    b.hold(1.0, C, holder_id=3)
    ev = drive(observations(b.frames, running=False, resume_at=6 / 30, clock_start_s=600.0, period=1))
    tip = provisional(ev, "tip_off")
    assert len(tip) == 1 and tip[0].team == "unknown" and abs(tip[0].t_pts - 1 / 30) < 0.05
    assert len(confirmed(ev, "tip_off")) == 1


def test_tip_off_retracted_without_possession():
    b = PlayBuilder(Scene(L))
    for i in range(6):
        b.frame(C, ((14.0, 7.5), 2.0 + 0.3 * i))
    for _ in range(11 * 30):
        b.frame(C, None)
    ev = drive(observations(b.frames, running=False, clock_start_s=600.0, period=1))
    assert len(retracted(ev, "tip_off")) == 1


def test_no_tip_off_mid_period():
    b = PlayBuilder(Scene(L))
    for i in range(6):
        b.frame(C, ((14.0, 7.5), 2.0 + 0.3 * i))
    assert provisional(drive(observations(b.frames, clock_start_s=431.0)), "tip_off") == []


def test_free_throw_made():
    b = PlayBuilder(Scene(L))
    b.hold(0.6, LANE, holder_id=1)
    b.shot(1, LANE, "left", outcome="make")
    b.hold(1.0, LANE, holder_id=1)
    ev = drive(observations(b.frames, running=False, deltas={0.6 + 1.2: (1, 0)}))
    att = confirmed(ev, "free_throw_attempt")
    assert len(att) == 1 and att[0].team == "home" and att[0].zone == "ft"
    assert abs(att[0].t_pts - 0.6) < 0.05
    assert len(confirmed(ev, "free_throw_made")) == 1 and provisional(ev, "shot_attempt") == []


def test_free_throw_missed():
    b = PlayBuilder(Scene(L))
    b.hold(0.6, LANE, holder_id=1)
    b.shot(1, LANE, "left", outcome="miss")
    b.hold(4.0, LANE, holder_id=2)
    ev = drive(observations(b.frames, running=False))
    assert len(confirmed(ev, "free_throw_missed")) == 1 and confirmed(ev, "free_throw_made") == []
