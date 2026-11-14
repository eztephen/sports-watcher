from dataclasses import replace

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



def test_shot_is_dated_from_the_shooters_last_touch_when_the_ball_came_back_to_him():
    # A drive: the detector loses the ball in the dribbler's hands and reports a still blob
    # on the floor for 0.6 s (a release as far as the state can tell), then finds the ball
    # again in his hands on the way up. The shot leaves his hands then, not 0.8 s earlier.
    b = PlayBuilder(Scene(L))
    players = {1: ("home", (22.0, 7.5)), 2: ("away", (16.0, 12.0))}
    b.hold(1.0, players, holder_id=1)
    for _ in range(18):
        b.frame(players, ((19.5, 7.5), 1.0))  # the blob: behind him, out of everyone's reach
    for i in range(6):
        b.frame(players, ((22.0, 7.5), 1.0 + 0.2 * i))  # back in his hands, rising
    last_touch = b.frames[-1].t_pts
    b.shot(1, players, "right", flight_s=0.6)
    states = _states(observations(b.frames))
    touched = next(st for st in states if st.t_pts == last_touch)
    assert touched.release_touch_pts == last_touch  # seen back in the releaser's reach
    shot = states[-1].last_shot
    assert shot is not None and shot.team == "home" and shot.track_id == 1
    assert abs(shot.t_pts - last_touch) <= 0.2
    assert shot.release_pts is not None and shot.release_pts < 1.1  # still the same release


def test_shot_keeps_the_release_time_when_the_ball_never_came_back():
    b = PlayBuilder(Scene(L))
    players = {1: ("home", (22.0, 7.5)), 2: ("away", (16.0, 12.0))}
    b.hold(1.0, players, holder_id=1)
    release = b.frames[-1].t_pts
    b.shot(1, players, "right", flight_s=0.8)
    states = _states(observations(b.frames))
    shot = states[-1].last_shot
    assert shot is not None and abs(shot.t_pts - release) <= 0.05
    assert all(st.release_touch_pts is None for st in states)


def test_trajectory_shot_takes_the_later_touch_as_its_time():
    # The trajectory fit dates a shot at the release. When the ball was seen back in the
    # releaser's reach after that release, the later touch is when it left his hands.
    b = PlayBuilder(Scene(L))
    players = {1: ("home", (22.0, 7.5)), 2: ("away", (16.0, 12.0))}
    b.hold(1.0, players, holder_id=1)
    b.shot(1, players, "right", flight_s=0.8)
    obs = observations(b.frames)
    plugin, state, i = BasketballPlugin(L), GameState(), 0
    while state.last_shot is None:
        before, state, i = state, plugin.update_state(state, obs[i]), i + 1
    assert state.last_shot.t_pts == state.last_release.t_pts  # recorded by the fit, at the release
    touch = before.last_release.t_pts + 0.1
    touched = replace(before, release_touch_pts=touch, release_left_reach=True)
    shot = plugin.update_state(touched, obs[i - 1]).last_shot
    assert shot is not None and shot.t_pts == touch and shot.release_pts == before.last_release.t_pts


def test_a_basket_nobody_was_seen_shooting_is_not_credited_to_an_old_shot():
    # Home misses, away rebounds. Later, after a close-up, the ball drops through the rim
    # with no release seen (on footage: a free throw shot during the close-up). The shot on
    # record is the old home miss; the ball does not go to away because of it.
    from sports_watcher.sports.basketball.geometry import basket_xy
    from tests.helpers.vision import RIM_HEIGHT_M

    players = {1: ("home", (22.0, 7.5)), 2: ("away", (25.0, 6.0)), 3: ("home", (18.0, 3.0))}
    b = PlayBuilder(Scene(L))
    b.hold(0.6, players, holder_id=1)
    b.shot(1, players, "right", outcome="miss")
    b.hold(1.0, players, holder_id=2)
    for _ in range(round(6.0 * b.fps)):
        b.frame(players, None, camera_hint="closeup")
    rim = basket_xy(L, "right")
    for i in range(12):
        b.frame(players, (rim, RIM_HEIGHT_M + 0.6 - 0.1 * i))
    crossing = [s for s in _states(observations(b.frames)) if s.rim_interaction and s.rim_interaction.crossed_pts]
    assert crossing and crossing[-1].rim_interaction.crossed_pts > 7.0  # the drop was seen as a basket
    b.hold(1.0, players, holder_id=3)
    states = _states(observations(b.frames))
    assert states[-1].holder.team == "home" and states[-1].possession == "home"


def test_the_hold_timer_restarts_when_the_nearest_player_changes():
    players = {1: ("home", (12.0, 8.0)), 2: ("away", (15.0, 8.0))}
    b = PlayBuilder(Scene(L))
    for i in range(6):
        b.hold(0.1, players, holder_id=1 if i % 2 == 0 else 2)
    states = _states(observations(b.frames))
    assert all(s.holder is None for s in states)


def test_a_banner_sliding_over_the_bug_does_not_stop_the_clock():
    # One Sports: as a sponsor banner covers the score bug the OCR vote repeats the last
    # clock value for a second and a half with falling confidence. Those are not reads of
    # a stopped clock: the gate stays open through the five assumed seconds.
    b = PlayBuilder(Scene(L))
    players = {1: ("home", HOME_A), 2: ("away", AWAY_B)}
    b.hold(12.0, players, holder_id=1)
    obs = observations(b.frames)
    frozen = next(o.scoreboard.clock_s for o in obs if o.t_pts >= 3.0 and o.scoreboard is not None)
    for i, o in enumerate(obs):
        if o.scoreboard is None or o.t_pts < 3.0:
            continue
        fade = 1.0 - 0.8 * (o.t_pts - 3.0)  # 0.8, 0.6, 0.4 ... at four reads a second
        if fade > 0.0:
            sb = o.scoreboard.model_copy(update={"clock_s": frozen, "confidence": round(max(fade, 0.0), 2)})
        else:
            sb = o.scoreboard.model_copy(update={"clock_s": None, "confidence": 0.0})
        obs[i] = replace(o, scoreboard=sb)
    by_t = {round(o.t_pts, 2): s for o, s in zip(obs, _states(obs))}
    assert by_t[5.5].clock_running is True and by_t[5.5].clock_state in ("read", "assumed")
    assert by_t[8.0].clock_running is True and by_t[8.0].clock_state == "assumed"
    assert by_t[11.0].clock_running is False and by_t[11.0].clock_state == "unknown"


def test_a_clock_read_as_standing_still_closes_the_gate():
    b = PlayBuilder(Scene(L))
    players = {1: ("home", HOME_A), 2: ("away", AWAY_B)}
    b.hold(6.0, players, holder_id=1)
    obs = observations(b.frames)
    frozen = next(o.scoreboard.clock_s for o in obs if o.t_pts >= 2.0 and o.scoreboard is not None)
    obs = [replace(o, scoreboard=o.scoreboard.model_copy(update={"clock_s": frozen})) if o.scoreboard is not None and o.t_pts >= 2.0 else o for o in obs]
    states = _states(obs)
    assert states[-1].clock_running is False and states[-1].clock_state == "read"


def _rebound_through_two_track_ids(first_team: str):
    """A miss; the ball is seen for two frames in one player's hands, vanishes among the
    bodies for 0.8 s, and is then held by another track (on footage: the same rebounder
    under a new id, or the teammate he handed it to)."""
    from tests.helpers.vision import HOLD_M

    players = {1: ("home", (22.0, 7.5)), 2: (first_team, (25.0, 6.0)), 5: ("away", (21.0, 4.0)), 3: ("home", (15.0, 3.0))}
    b = PlayBuilder(Scene(L))
    b.hold(0.6, players, holder_id=1)
    b.shot(1, players, "right", outcome="miss")
    touch_at = b.t
    for _ in range(2):
        b.frame(players, (players[2][1], HOLD_M))
    for _ in range(24):
        b.frame(players, None)
    held_at = b.t
    b.hold(1.0, players, holder_id=5)
    return _states(observations(b.frames)), touch_at, held_at


def test_a_rebound_is_dated_from_the_teams_first_touch_after_the_shot():
    states, touch_at, _ = _rebound_through_two_track_ids("away")
    assert states[-1].possession == "away"
    assert abs(states[-1].possession_since_pts - touch_at) < 0.05  # not 0.8 s later, at the hold


def test_an_opponents_touch_does_not_date_the_rebound():
    states, _, held_at = _rebound_through_two_track_ids("home")
    assert states[-1].possession == "away"
    assert abs(states[-1].possession_since_pts - held_at) < 0.1
