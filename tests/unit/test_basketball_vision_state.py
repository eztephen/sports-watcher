from sports_watcher.sports.basketball.plugin import BasketballPlugin
from sports_watcher.state.game_state import GameState
from tests.helpers.leagues import fixture_league
from tests.helpers.vision import PlayBuilder, Scene, observations

L = fixture_league("pba")
HOME_A, AWAY_B = (10.0, 7.0), (12.0, 8.0)


def _states(obs):
    plugin, state, out = BasketballPlugin(L), GameState(), []
    for o in obs:
        state = plugin.update_state(state, o)
        out.append(state)
    return out


def test_holder_needs_hold_time():
    b = PlayBuilder(Scene(L))
    players = {1: ("home", HOME_A), 2: ("away", AWAY_B)}
    b.hold(0.10, players, holder_id=1)  # 3 frames
    states = _states(observations(b.frames))
    assert states[-1].holder is None and states[-1].holder_candidate == (1, 0.0)
    b.hold(0.10, players, holder_id=1)  # 6 frames in total, 0.17 s reached
    states = _states(observations(b.frames))
    assert states[-1].holder is not None and states[-1].holder.track_id == 1
    assert states[-1].possession == "home" and states[-1].possession_since_pts == 0.0


def test_clock_gate_closes_two_seconds_after_the_last_clock_change():
    b = PlayBuilder(Scene(L))
    players = {1: ("home", HOME_A), 2: ("away", AWAY_B)}
    b.hold(3.0, players, holder_id=1)
    states = _states(observations(b.frames, running=False))
    assert states[-1].clock_running is False and states[-1].clock_state == "read"
    states = _states(observations(b.frames, running=True))
    assert states[-1].clock_running is True


def test_hidden_bug_assumes_running_for_five_seconds():
    b = PlayBuilder(Scene(L))
    players = {1: ("home", HOME_A), 2: ("away", AWAY_B)}
    b.hold(9.0, players, holder_id=1)
    obs = observations(b.frames, running=True, hidden=[(1.0, 9.0)])
    states = _states(obs)

    def at(t):
        return states[round(t * 30)]

    assert at(2.5).clock_running and at(2.5).clock_state == "read"  # within 2 s of the last read
    assert at(5.0).clock_running and at(5.0).clock_state == "assumed"
    assert at(7.5).clock_running and at(7.5).clock_state == "assumed"  # 2 s read + 5 s assumed
    assert not at(8.5).clock_running and at(8.5).clock_state == "unknown"


def test_closeup_clears_holder_but_keeps_possession():
    b = PlayBuilder(Scene(L))
    players = {1: ("home", HOME_A), 2: ("away", AWAY_B)}
    b.hold(0.5, players, holder_id=1)
    b.frame(players, None, camera_hint="closeup")
    states = _states(observations(b.frames))
    assert states[-1].holder is None and states[-1].possession == "home"
    assert states[-1].camera_left_main_pts == states[-1].t_pts


def test_release_and_rim_interaction_recorded_for_a_make():
    b = PlayBuilder(Scene(L))
    players = {1: ("home", (22.0, 7.5)), 2: ("away", AWAY_B)}
    b.hold(0.3, players, holder_id=1)
    b.shot(1, players, "right", outcome="make")
    states = _states(observations(b.frames))
    rel = next(s.last_release for s in states if s.last_release is not None)
    assert rel.track_id == 1 and rel.team == "home" and rel.zone == "two"
    assert states[-1].last_shot is not None and states[-1].last_shot.zone == "two"
    ri = states[-1].rim_interaction
    assert ri.entered_pts is not None and ri.crossed_pts is not None


def test_pass_records_release_but_no_shot():
    b = PlayBuilder(Scene(L))
    players = {1: ("home", HOME_A), 3: ("home", (16.0, 5.0)), 2: ("away", AWAY_B)}
    b.hold(0.3, players, holder_id=1)
    b.pass_(1, 3, players)
    b.hold(0.3, players, holder_id=3)
    states = _states(observations(b.frames))
    assert any(s.last_release is not None for s in states)
    assert states[-1].last_shot is None and states[-1].holder.track_id == 3


def test_ball_lost_keeps_holder():
    b = PlayBuilder(Scene(L))
    players = {1: ("home", HOME_A), 2: ("away", AWAY_B)}
    b.hold(0.5, players, holder_id=1)
    for _ in range(20):
        b.frame(players, None)  # ball unseen for 0.67 s
    states = _states(observations(b.frames))
    assert states[-1].holder is not None and states[-1].holder.track_id == 1


def test_lane_formation_detected_when_clock_stopped():
    b = PlayBuilder(Scene(L))
    lane = {
        1: ("home", (5.8, 7.5)), 2: ("away", (3.0, 7.5 + 2.45)), 3: ("home", (3.0, 7.5 - 2.45)),
        4: ("away", (4.5, 7.5 + 2.6)), 5: ("home", (4.5, 7.5 - 2.6)), 6: ("away", (10.0, 7.5)),
    }
    b.hold(0.5, lane, holder_id=1)
    states = _states(observations(b.frames, running=False))
    lf = states[-1].lane_formation
    assert lf is not None and lf.shooter_track_id == 1 and lf.side == "left" and lf.team == "home"
