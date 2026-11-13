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


def test_shot_during_frozen_clock_is_not_deferred_to_the_restart():
    b = PlayBuilder(Scene(L))
    b.hold(0.6, S, holder_id=1)
    b.shot(1, S, "right", outcome="make")
    b.hold(8.0, S, holder_id=2)
    ev = drive(observations(b.frames, running=False, resume_at=4.0))
    assert provisional(ev, "shot_attempt") == [] and provisional(ev, "shot_missed") == []
    assert provisional(ev, "rebound") == []


def test_made_free_throw_is_neither_a_field_goal_nor_double_counted():
    LANE = {
        1: ("home", (5.8, 7.5)), 2: ("away", (3.0, 9.95)), 3: ("home", (3.0, 5.05)),
        4: ("away", (4.5, 10.1)), 5: ("home", (4.5, 4.9)), 6: ("away", (10.0, 7.5)),
    }
    b = PlayBuilder(Scene(L))
    b.hold(0.6, LANE, holder_id=1)
    b.shot(1, LANE, "left", outcome="make")
    b.hold(6.0, LANE, holder_id=6)  # the inbounder picks the ball up before the bug updates
    ev = drive(observations(b.frames, running=False, resume_at=5.0, deltas={2.6: (1, 0)}))
    assert len(confirmed(ev, "free_throw_made")) == 1
    assert provisional(ev, "shot_attempt") == [] and provisional(ev, "shot_missed") == []


def _late_delta_make():
    """A make whose score change shows on the bug six seconds after the release."""
    b = PlayBuilder(Scene(L))
    b.hold(0.6, S, holder_id=1)
    b.shot(1, S, "right", outcome="make")
    b.hold(9.0, S, holder_id=2)
    return observations(b.frames, deltas={6.6: (2, 0)})


def test_slow_scoreboard_keeps_the_vision_shot_when_the_lag_is_configured():
    ev = drive(_late_delta_make(), score_lag_s=8.0)
    att = confirmed(ev, "shot_attempt")
    assert len(att) == 1 and abs(att[0].t_pts - 0.6) < 0.05  # no retroactive twin at 6.6 s
    made = confirmed(ev, "shot_made")
    assert len(made) == 1 and abs(made[0].t_pts - 0.6) < 0.05 and made[0].confidence == 0.9
    assert provisional(ev, "shot_missed") == [] and retracted(ev, "shot_made") == []
    assert confirmed(ev, "rebound") == []


def test_slow_scoreboard_with_the_default_lag_backfills_a_second_attempt():
    ev = drive(_late_delta_make())
    assert len(confirmed(ev, "shot_attempt")) == 2
    assert retracted(ev, "shot_made")[0].reason == "ttl"


def _shot_with_hidden_release(outcome: str):
    """The shooter's body hides the ball for the first 80 % of the flight."""
    b = PlayBuilder(Scene(L))
    b.hold(0.6, S, holder_id=1)
    flight = b.shot(1, S, "right", flight_s=1.0, outcome=outcome)
    for vf in flight[:24]:
        vf.ball = None
    seen_at = flight[24].t_pts
    for _ in range(50):  # then lost in the crowd under the basket
        b.frame(S, None)
    b.hold(3.0, S, holder_id=2)
    return b, seen_at


def test_shot_seen_only_on_its_way_down_is_an_attempt_dated_from_where_it_was_seen():
    b, seen_at = _shot_with_hidden_release("make")
    ev = drive(observations(b.frames, deltas={3.5: (2, 0)}))
    att = provisional(ev, "shot_attempt")
    assert len(att) == 1 and att[0].team == "home" and att[0].evidence.note != "retroactive"
    assert abs(att[0].t_pts - seen_at) < 0.1
    assert len(confirmed(ev, "shot_made")) == 1


def test_missed_shot_seen_only_on_its_way_down_is_an_attempt_and_a_miss():
    b, _ = _shot_with_hidden_release("miss")
    ev = drive(observations(b.frames))
    assert len(confirmed(ev, "shot_attempt")) == 1
    assert len(provisional(ev, "shot_missed")) == 1


FT = {1: ("home", (5.8, 7.5)), 6: ("away", (10.0, 7.5))}


def test_one_point_without_a_lane_formation_is_a_free_throw_dated_from_the_flight():
    # No court model, so no lane formation: the only signs are a ball that flew into the
    # rim with the clock stopped and a +1 on a slow scoreboard five seconds later.
    b = PlayBuilder(Scene(L))
    b.hold(0.6, FT, holder_id=1, rim_side="left", homography=False)
    released_at = b.t
    for vf in b.shot(1, FT, "left", outcome="make"):
        vf.homography = None
    for _ in range(round(6.0 * b.fps)):
        b.frame(FT, None, rim_side="left", homography=False)
    ev = drive(observations(b.frames, running=False, deltas={released_at + 5.3: (1, 0)}), score_lag_s=10.0)
    att, made = confirmed(ev, "free_throw_attempt"), confirmed(ev, "free_throw_made")
    assert len(att) == 1 and len(made) == 1 and att[0].team == made[0].team == "home"
    assert abs(att[0].t_pts - released_at) < 0.3 and made[0].t_pts == att[0].t_pts
    assert att[0].zone == "ft" and att[0].evidence.note == "retroactive" and att[0].confidence == 0.4
    assert provisional(ev, "shot_attempt") == [] and provisional(ev, "shot_made") == []


def test_one_point_nobody_saw_is_dated_back_by_the_broadcasters_score_delay():
    b = PlayBuilder(Scene(L))
    for _ in range(round(10.0 * b.fps)):
        b.frame(FT, None, camera_hint="closeup")
    obs = observations(b.frames, running=False, deltas={8.0: (1, 0)})
    att = confirmed(drive(obs, score_lag_s=10.0, score_delay_s=5.3), "free_throw_attempt")
    assert len(att) == 1 and abs(att[0].t_pts - (8.0 - 5.3 - 1.0)) < 0.05
    att = confirmed(drive(obs, score_lag_s=10.0), "free_throw_attempt")
    assert len(att) == 1 and abs(att[0].t_pts - 8.0) < 0.05  # no delay configured: the delta's time



def test_attempt_dated_after_its_release_is_confirmed_by_a_rim_entry_in_between():
    # A layup: the shot is dated from the shooter's last touch, by which time the ball is
    # already in the rim area. The rim entry belongs to the release the shot was recorded
    # for, so the attempt stands and the scorer picking the ball up does not retract it.
    from sports_watcher.sports.basketball.rules_shots import ShotAttemptRule
    from sports_watcher.state.game_state import GameState, Holder, RimInteraction, ShotRecord

    b = PlayBuilder(Scene(L))
    obs = observations(b.hold(0.1, {1: ("home", (22.0, 7.5))}, holder_id=1))[-1]
    shot = ShotRecord(2.0, 1, "home", "two", None, release_pts=1.0)
    state = GameState(
        clock_running=True, last_shot=shot, rim_interaction=RimInteraction(entered_pts=1.9),
        possession="away", possession_since_pts=2.6, holder=Holder(1, "home", 2.6),
    )
    (draft,) = ShotAttemptRule()(GameState(), state, obs)
    assert draft.t_pts == 2.0
    assert draft.confirmer(draft, state, obs) == "confirmed"
