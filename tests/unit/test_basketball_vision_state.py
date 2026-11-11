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


def test_holder_stays_while_the_ball_is_inside_their_box():
    S = {1: ("home", (22.0, 7.5)), 2: ("away", (22.5, 7.5))}
    b = PlayBuilder(Scene(L))
    b.hold(1.0, S, holder_id=1)
    for _ in range(30):
        b.frame(S, ((22.3, 7.5), 1.0))  # dribbled on the defender's side for 1 s
    b.hold(1.0, S, holder_id=1)
    states = _states(observations(b.frames))
    assert all(s.holder is None or s.holder.track_id == 1 for s in states[10:])


def test_predicted_ball_does_not_hand_the_ball_to_a_defender():
    players = {1: ("home", HOME_A), 2: ("away", AWAY_B)}
    b = PlayBuilder(Scene(L))
    b.hold(0.5, players, holder_id=1)
    for _ in range(15):  # the ball is hidden; the filter's guess drifts onto the defender
        b.frame(players, (AWAY_B, 1.0, "predicted"))
    states = _states(observations(b.frames))
    assert states[-1].holder is not None and states[-1].holder.track_id == 1
    assert states[-1].possession == "home"
    for _ in range(15):  # seen there for real: the defender has it
        b.frame(players, (AWAY_B, 1.0))
    states = _states(observations(b.frames))
    assert states[-1].holder is not None and states[-1].holder.track_id == 2



def test_ball_within_reach_of_both_teams_stays_with_the_team_in_possession():
    # A pass arrives between the receiver and his defender, nearer the defender's box centre.
    players = {1: ("home", (10.0, 7.0)), 2: ("away", (10.5, 7.0)), 3: ("home", (6.0, 7.0))}
    contested = ((10.35, 7.0), 1.0)
    b = PlayBuilder(Scene(L))
    b.hold(0.5, players, holder_id=3)
    b.pass_(3, 1, players, flight_s=0.3)
    for _ in range(15):
        b.frame(players, contested)
    states = _states(observations(b.frames))
    assert states[-1].holder is not None and states[-1].holder.track_id == 1
    assert states[-1].possession == "home"
    # The defender walks off with it: out of every home player's reach, it is his.
    stolen = {**players, 2: ("away", (13.0, 7.0))}
    b.hold(0.5, stolen, holder_id=2)
    states = _states(observations(b.frames))
    assert states[-1].holder is not None and states[-1].holder.track_id == 2
    assert states[-1].possession == "away"


def _after_shot(outcome: str):
    """Home shoots at the right basket; a player carrying a home label picks the ball up
    under that basket (the inbounder, mislabelled) and holds it."""
    shooter, picker = (22.0, 7.5), (26.5, 7.5)
    players = {1: ("home", shooter), 2: ("away", (20.0, 9.0)), 3: ("home", picker)}
    b = PlayBuilder(Scene(L))
    b.hold(0.5, players, holder_id=1)
    b.shot(1, players, "right", outcome=outcome)
    picked_up_at = b.t
    b.hold(0.5, players, holder_id=3)
    return _states(observations(b.frames)), picked_up_at


def test_after_a_made_basket_the_next_holder_starts_the_other_teams_possession():
    states, picked_up_at = _after_shot("make")
    assert states[-1].holder is not None and states[-1].holder.track_id == 3
    assert states[-1].possession == "away"
    assert abs(states[-1].possession_since_pts - picked_up_at) < 0.2


def test_after_a_miss_the_holders_own_label_decides_possession():
    states, _ = _after_shot("miss")
    assert states[-1].holder is not None and states[-1].holder.track_id == 3
    assert states[-1].possession == "home"


def test_a_dribbler_becomes_the_holder_although_the_ball_rises_on_every_bounce():
    players = {1: ("home", HOME_A), 2: ("away", (16.0, 8.0))}
    b = PlayBuilder(Scene(L))
    heights = [1.0, 0.8, 0.6, 0.4, 0.2, 0.4, 0.6, 0.8]  # four frames down, four up
    for i in range(24):
        b.frame(players, (HOME_A, heights[i % len(heights)]))
    states = _states(observations(b.frames))
    assert states[-1].holder is not None and states[-1].holder.track_id == 1
    assert states[-1].holder.since_pts < 0.1
    assert states[-1].possession == "home"


def test_holder_takes_the_team_once_his_track_gets_a_label():
    # Team labels arrive only after the assigner has seen enough players.
    b = PlayBuilder(Scene(L))
    b.hold(0.5, {1: ("unknown", HOME_A), 2: ("unknown", AWAY_B)}, holder_id=1)
    started = b.frames[0].t_pts
    b.hold(0.5, {1: ("home", HOME_A), 2: ("away", AWAY_B)}, holder_id=1)
    states = _states(observations(b.frames))
    assert states[-1].holder.team == "home"
    assert states[-1].possession == "home" and states[-1].possession_since_pts == started


def test_possession_after_a_made_basket_is_not_flipped_by_the_next_label():
    # Home scores. An away player takes the ball out of the net, then the inbounder, who
    # carries a home label, holds it half a second later: still away's ball.
    shooter, under = (22.0, 7.5), (26.5, 7.5)
    players = {1: ("home", shooter), 2: ("away", under), 3: ("home", (27.5, 6.0))}
    b = PlayBuilder(Scene(L))
    b.hold(0.5, players, holder_id=1)
    b.shot(1, players, "right", outcome="make")
    b.hold(0.4, players, holder_id=2)
    b.pass_(2, 3, players, flight_s=0.2)
    b.hold(0.5, players, holder_id=3)
    states = _states(observations(b.frames))
    assert states[-1].holder.track_id == 3 and states[-1].possession == "away"
    # Long after the basket a home hold is a steal.
    b.hold(3.0, players, holder_id=2)
    b.hold(0.5, players, holder_id=3)
    assert _states(observations(b.frames))[-1].possession == "home"

