"""Geometry facts the basketball plug-in derives from a VisionFrame into GameState.

Rules never look at pixels: they read the holder, the last release, the rim interaction,
the lane formation and the clock gate that this module maintains.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from sports_watcher.config import LeagueConfig
from sports_watcher.perception.types import BallObs, Box, Team, Track, box_center, box_height
from sports_watcher.sports.base import Observation
from sports_watcher.sports.basketball.geometry import (
    free_throw_xy,
    in_lane_edges,
    near,
    nearest_side,
    zone_for,
)
from sports_watcher.state.game_state import (
    GameState,
    Holder,
    LaneFormation,
    Release,
    RimInteraction,
    ShotRecord,
)

HOLD_S = 0.165  # five perception-frame intervals at 30 Hz (the spec's 0.17 s)
BOX_GROW = 0.25
CLOCK_RUNNING_S = 2.0
ASSUMED_MAX_S = 5.0
RESUME_MAX_STEP_S = 1.5
FREE_GAP_S = 0.75  # a longer gap in detections restarts the flight (a high arc leaves the frame for less)
RIM_SHOT_RELEASE_GAP_S = 3.0  # hidden time between the dated release and the seen flight
RIM_SHOT_FLIGHT_S = 2.5
POST_BASKET_LOCK_S = 3.0  # the inbound and the first passes after a made basket
MIN_SCOREBOARD_CONFIDENCE = 0.6
RIM_REGION_WIDTHS = 2.0
SHOT_GATE_WIDTHS = 1.5
SHOT_FIT_OBS = 4
SHOT_WINDOW_S = 1.5
BALL_PATH_MAX_S = 1.5
RIM_CROSS_WINDOW_S = 0.5
REBOUND_TOUCH_S = 2.5  # how far back a team's first touch after a shot may date its possession
TIP_BELOW_WIDTHS = 0.3  # a miss has come down this far below the rim before it can be tipped
TIP_TOUCH_S = 1.0  # from the touch to the ball in the air again
TIP_RISE_WIDTHS = 0.5  # and the tip sends it back up by this much, in rim widths
LANE_MIN_PLAYERS = 4
LANE_MARGIN_M = 1.0
FT_LINE_RADIUS_M = 1.0
MIN_HOMOGRAPHY_CONF = 0.4


# -- clock gate -------------------------------------------------------------------------


def update_clock_gate(prev: GameState, new: GameState, obs: Observation) -> None:
    """Set ``clock_running`` and ``clock_state`` from the scoreboard reads seen so far."""
    sb = obs.scoreboard
    t = obs.t_pts
    confident = sb is not None and sb.confidence >= MIN_SCOREBOARD_CONFIDENCE and sb.clock_s is not None
    if confident:
        new.last_scoreboard_pts = t
        if prev.clock_s is not None and sb.clock_s < prev.clock_s - 1e-6:
            new.last_clock_change_pts = t
            before = prev.last_clock_change_value if prev.last_clock_change_value is not None else prev.clock_s
            new.last_clock_step_s = before - sb.clock_s
            new.last_clock_change_value = sb.clock_s
    since_read = t - new.last_scoreboard_pts if new.last_scoreboard_pts is not None else float("inf")
    # The clock is running when it had changed within 2 s of the last confident read. The
    # time since that read does not count against it: as a banner slides over the bug the
    # OCR vote repeats the last value with falling confidence, and measuring from now
    # closed the gate before the assumed seconds began.
    if new.last_scoreboard_pts is not None and new.last_clock_change_pts is not None:
        running_at_read = new.last_scoreboard_pts - new.last_clock_change_pts <= CLOCK_RUNNING_S
    else:
        running_at_read = False
    if since_read <= CLOCK_RUNNING_S:
        new.clock_state = "read"
        new.clock_running = running_at_read
    elif (
        obs.camera_state == "main"
        and since_read <= CLOCK_RUNNING_S + ASSUMED_MAX_S
        and prev.clock_state in ("read", "assumed")
    ):
        new.clock_state = "assumed"
        new.clock_running = running_at_read
    else:
        new.clock_state = "unknown"
        new.clock_running = False
    if new.clock_running and not prev.clock_running:
        new.clock_running_since_pts = t
        # One tick down from the value at the previous change: the clock stood still while
        # the gate was closed (a dead ball). Any other step means it ran on unseen, behind a
        # replay or a graphic, or a read in between was wrong.
        step = new.last_clock_step_s
        new.clock_resumed_from_stop = step is not None and 0.0 < step <= RESUME_MAX_STEP_S


# -- pure geometry helpers --------------------------------------------------------------


BOX_GROW_TOP = 0.1  # hands above the head; a ball higher than that is airborne
RISING_PX = 0.5  # image-y decrease per frame that counts as rising


def grown(box: Box, factor: float) -> Box:
    """A player's reach: the box widened by ``factor`` of its height on each side and
    raised by ``BOX_GROW_TOP`` of its height (arms up), feet unchanged."""
    h = box_height(box)
    return (box[0] - h * factor, box[1] - h * BOX_GROW_TOP, box[2] + h * factor, box[3])


def _inside(xy: tuple[float, float], box: Box) -> bool:
    return box[0] <= xy[0] <= box[2] and box[1] <= xy[1] <= box[3]


def holder_candidate(tracks: list[Track], ball: BallObs | None, prefer_team: str | None = None) -> int | None:
    """The player whose grown box contains the ball centre, nearest box centre first.

    With ``prefer_team`` (the team in possession) a player of that team within reach wins
    over a nearer opponent: boxes of a ball handler and his defender overlap, and the ball
    changes sides only once it is out of reach of every player of the team that had it."""
    if ball is None:
        return None
    best: tuple[bool, float, int] | None = None
    for tr in tracks:
        if tr.cls != "player" or not _inside(ball.xy, grown(tr.box, BOX_GROW)):
            continue
        cx, cy = box_center(tr.box)
        d = (cx - ball.xy[0]) ** 2 + (cy - ball.xy[1]) ** 2
        key = (prefer_team is None or tr.team != prefer_team, d, tr.track_id)
        if best is None or key < best:
            best = key
    return best[2] if best else None


def rim_region(rim: Box) -> Box:
    cx, cy = box_center(rim)
    half = (rim[2] - rim[0]) * RIM_REGION_WIDTHS / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


def shot_path_passes_rim(path: list[tuple[float, float, float]], rim: Box) -> bool:
    """Fit x linear and y quadratic in time over the path; pass when the ball starts upward
    and the extrapolated flight comes within ``SHOT_GATE_WIDTHS`` rim widths of the rim
    centre inside the window."""
    if len(path) < SHOT_FIT_OBS:
        return False
    ts = np.array([p[0] for p in path]) - path[0][0]
    xs = np.array([p[1] for p in path])
    ys = np.array([p[2] for p in path])
    bx = np.polyfit(ts, xs, 1)
    by = np.polyfit(ts, ys, 2)
    if by[1] >= 0:
        return False  # not rising at release (image y grows downward)
    tt = np.linspace(0.0, SHOT_WINDOW_S, 60)
    px, py = np.polyval(bx, tt), np.polyval(by, tt)
    cx, cy = box_center(rim)
    gate = (rim[2] - rim[0]) * SHOT_GATE_WIDTHS
    return bool(np.min(np.hypot(px - cx, py - cy)) <= gate)


def zone_of_feet(
    feet_xy: tuple[float, float] | None, league: LeagueConfig, homography_conf: float
) -> str:
    if feet_xy is None or homography_conf < MIN_HOMOGRAPHY_CONF:
        return "unknown"
    return zone_for(feet_xy, league, nearest_side(feet_xy, league))


def lane_formation(
    tracks: list[Track], holder: Holder | None, league: LeagueConfig, t_pts: float
) -> LaneFormation | None:
    """A shooter at the free-throw line with at least four players on the lane edges."""
    if holder is None:
        return None
    shooter = next((tr for tr in tracks if tr.track_id == holder.track_id), None)
    if shooter is None or shooter.feet_xy is None:
        return None
    side = nearest_side(shooter.feet_xy, league)
    if not near(shooter.feet_xy, free_throw_xy(league, side), FT_LINE_RADIUS_M):
        return None
    on_edges = sum(
        1
        for tr in tracks
        if tr.cls == "player"
        and tr.track_id != shooter.track_id
        and tr.feet_xy is not None
        and in_lane_edges(tr.feet_xy, league, side, LANE_MARGIN_M)
    )
    if on_edges < LANE_MIN_PLAYERS:
        return None
    return LaneFormation(shooter.track_id, holder.team, side, holder.since_pts)


# -- state updates ----------------------------------------------------------------------


def freeze_vision(prev: GameState, new: GameState, obs: Observation) -> None:
    """Off the main camera: no holder, possession kept, remember when we left."""
    new.holder = None
    new.holder_candidate = None
    if prev.camera_left_main_pts is None:
        new.camera_left_main_pts = obs.t_pts


def apply_vision(prev: GameState, new: GameState, obs: Observation, league: LeagueConfig) -> None:
    vf = obs.vision
    assert vf is not None
    t = obs.t_pts
    new.camera_left_main_pts = None
    new.tracks = list(vf.tracks)
    new.ball = vf.ball
    new.rim = vf.rim
    new.homography_confidence = vf.homography.confidence if vf.homography is not None else 0.0
    new.ball_path = list(prev.ball_path)
    new.rim_interaction = replace(prev.rim_interaction) if prev.rim_interaction is not None else None

    _apply_holder(prev, new, vf, t)
    if new.holder is not None:
        new.last_held = (new.holder.track_id, new.holder.team, t)
    _apply_ball_free(new, vf, t)
    _apply_possession(new)
    _apply_release(prev, new, vf, t, league)
    _apply_release_touch(new, vf, t)
    _apply_ball_path(new, vf, t)
    _apply_rim_interaction(prev, new, vf, t)
    _apply_lane(prev, new, league, t)


def _track(tracks: list[Track], track_id: int) -> Track | None:
    return next((tr for tr in tracks if tr.track_id == track_id), None)


def _apply_holder(prev: GameState, new: GameState, vf, t: float) -> None:
    if vf.ball is None or vf.ball.source == "predicted":
        # Ball unseen: holder and candidate carry over. A predicted position is the ball
        # filter's guess while the ball is hidden, and on footage it drifts onto the nearest
        # defender; only a detection can put the ball in new hands.
        return
    if new.holder is not None:
        held = _track(vf.tracks, new.holder.track_id)
        if held is not None and _inside(vf.ball.xy, grown(held.box, BOX_GROW)):
            new.holder_candidate = None
            if new.holder.team == "unknown" and held.team in ("home", "away"):
                new.holder = replace(new.holder, team=held.team)  # the label arrived mid-hold
            return  # still in the holder's reach: a nearer defender does not take it
    prefer = new.possession if new.possession in ("home", "away") else None
    cand = holder_candidate(vf.tracks, vf.ball, prefer)
    if cand is None:
        new.holder, new.holder_candidate = None, None
        return
    if new.holder is not None and new.holder.track_id == cand:
        new.holder_candidate = None
        return
    if prev.ball is not None and vf.ball.xy[1] < prev.ball.xy[1] - RISING_PX:
        # A rising ball is airborne: it may pass over a farther player's box in 2D
        # without being in anyone's hands. It starts no candidate and makes no holder,
        # but a candidate it was already with keeps waiting: a dribble rises on every
        # bounce, and the next frame it falls counts the whole time towards the hold.
        new.holder = None
        same = prev.holder_candidate is not None and prev.holder_candidate[0] == cand
        new.holder_candidate = prev.holder_candidate if same else None
        return
    since = t
    if prev.holder_candidate is not None and prev.holder_candidate[0] == cand:
        since = prev.holder_candidate[1]
    new.holder = None
    if t - since >= HOLD_S:
        tr = _track(vf.tracks, cand)
        new.holder = Holder(cand, tr.team if tr is not None else "unknown", since)
        new.holder_candidate = None
    else:
        new.holder_candidate = (cand, since)


def _scored_on(state: GameState) -> tuple[Team, float] | None:
    """The team that takes the ball out after the last shot went through the rim, and when
    it went through; None once that possession has started or when no basket was seen."""
    shot, ri = state.last_shot, state.rim_interaction
    if shot is None or ri is None or ri.crossed_pts is None or ri.crossed_pts <= shot.t_pts:
        return None
    if shot.team not in ("home", "away"):
        return None
    since = shot.t_pts if shot.release_pts is None else min(shot.t_pts, shot.release_pts)
    if ri.crossed_pts - since > RIM_SHOT_RELEASE_GAP_S + RIM_SHOT_FLIGHT_S:
        # The ball went through long after the shot on record: a basket whose release was
        # not seen (a free throw during a close-up), not that shot. It gave the ball to
        # the team that had just scored it.
        return None
    if state.possession_since_pts is not None and state.possession_since_pts > ri.crossed_pts:
        return None
    return ("away" if shot.team == "home" else "home"), ri.crossed_pts


def _apply_ball_free(new: GameState, vf, t: float) -> None:
    """Track since when the ball has been seen in the air: detected, out of every player's
    reach, without a gap longer than ``FREE_GAP_S`` (after a gap nobody knows where it was)."""
    if vf.ball is None or vf.ball.source != "detected":
        return
    seen, new.ball_seen_pts = new.ball_seen_pts, t
    toucher = holder_candidate(vf.tracks, vf.ball)
    if toucher is not None:
        tr = _track(vf.tracks, toucher)
        team = tr.team if tr is not None else "unknown"
        new.ball_touch = (toucher, team, t)
        new.touches = [(pts, tm) for pts, tm in new.touches if t - pts <= REBOUND_TOUCH_S] + [(t, team)]
        new.ball_free_since_pts = None
    elif new.ball_free_since_pts is None or seen is None or t - seen > FREE_GAP_S:
        new.ball_free_since_pts = t


def _apply_possession(new: GameState) -> None:
    h = new.holder
    if h is None:
        return
    scored_on = _scored_on(new)
    if scored_on is not None and h.since_pts > scored_on[1]:
        # After a basket the ball belongs to the team scored on, whatever label the player
        # who picks it up carries (team labels are least reliable under the basket).
        since = _since_first_touch(new, scored_on[0], h.since_pts)
        new.possession, new.possession_since_pts = scored_on[0], since
        new.possession_locked_until_pts = h.since_pts + POST_BASKET_LOCK_S
        return
    locked = new.possession_locked_until_pts
    if locked is not None and h.since_pts < locked:
        return  # the inbound: a label does not hand the ball back to the team that scored
    if h.team in ("home", "away") and h.team != new.possession:
        new.possession = h.team
        new.possession_since_pts = _since_first_touch(new, h.team, h.since_pts)


def _since_first_touch(state: GameState, team: Team, held_since: float) -> float:
    """After a shot the ball is fought for under the rim and the tracker changes ids there:
    the hold that makes the possession official starts a second or two after the team first
    had the ball. Date it from that team's first touch since the ball reached the rim, as
    long as the other team has not touched it since."""
    ri = state.rim_interaction
    if ri is None or ri.entered_pts is None or held_since <= ri.entered_pts:
        return held_since
    since = held_since
    for pts, tm in reversed(state.touches):
        if pts >= held_since:
            continue
        if tm != team or pts <= ri.entered_pts or held_since - pts > REBOUND_TOUCH_S:
            break
        since = pts
    return since


def _apply_release(prev: GameState, new: GameState, vf, t: float, league: LeagueConfig) -> None:
    if prev.holder is None or new.holder is not None or vf.ball is None or vf.ball.source != "detected":
        return
    if new.holder_candidate is not None:
        return  # the ball went straight into another player's hands
    tr = _track(prev.tracks, prev.holder.track_id)
    feet = tr.feet_xy if tr is not None else None
    if prev.homography_confidence < MIN_HOMOGRAPHY_CONF:
        feet = None  # no trustworthy court position without a homography
    new.last_release = Release(
        t_pts=prev.t_pts,
        track_id=prev.holder.track_id,
        team=prev.holder.team,
        image_xy=vf.ball.xy,
        feet_xy=feet,
        zone=zone_of_feet(feet, league, prev.homography_confidence),
    )
    lane = prev.lane_formation
    if lane is not None and lane.shooter_track_id == prev.holder.track_id and not new.clock_running:
        new.last_free_throw_pts = prev.t_pts
    new.ball_path = [(t, vf.ball.xy[0], vf.ball.xy[1])]
    new.rim_interaction = RimInteraction()
    new.release_left_reach, new.release_touch_pts = False, None


def _apply_release_touch(new: GameState, vf, t: float) -> None:
    """Note when the ball is seen back in the releaser's reach after it had left it.

    A drive can read as a release a second before the shot: the detector loses the ball in
    the dribbler's hands and reports something else, then finds it again in his hands on
    the way up, where a rising ball makes no holder. The shot left his hands at that later
    touch, not at the release."""
    rel = new.last_release
    if rel is None or vf.ball is None or vf.ball.source != "detected":
        return
    if t <= rel.t_pts or t - rel.t_pts > RIM_SHOT_RELEASE_GAP_S:
        return
    if new.last_shot is not None and new.last_shot.release_pts == rel.t_pts:
        return
    tr = _track(vf.tracks, rel.track_id)
    if tr is None or not _inside(vf.ball.xy, grown(tr.box, BOX_GROW)):
        new.release_left_reach = True
    elif new.release_left_reach:
        new.release_touch_pts = t


def _apply_ball_path(new: GameState, vf, t: float) -> None:
    rel = new.last_release
    if rel is None or vf.ball is None or vf.ball.source != "detected":
        return
    if t - rel.t_pts > BALL_PATH_MAX_S:
        return
    if not new.ball_path or new.ball_path[-1][0] != t:
        new.ball_path.append((t, vf.ball.xy[0], vf.ball.xy[1]))
    already = new.last_shot is not None and new.last_shot.release_pts == rel.t_pts
    if already or vf.rim is None or len(new.ball_path) < SHOT_FIT_OBS:
        return
    if shot_path_passes_rim(new.ball_path, vf.rim):
        free_throw = new.last_free_throw_pts == rel.t_pts
        touch = new.release_touch_pts
        new.last_shot = ShotRecord(
            rel.t_pts if touch is None else max(rel.t_pts, touch),
            rel.track_id, rel.team, rel.zone, rel.feet_xy,
            clock_running=new.clock_running, free_throw=free_throw, release_pts=rel.t_pts,
        )


def _shot_from_rim_entry(new: GameState, t: float) -> None:
    """A ball that flies into the rim region is a shot even when the path fit never saw it
    rise: the shooter's body often hides the release, which then gets dated at the last
    hold before the ball vanished. The shot is dated from where the flight was first seen."""
    rel, free = new.last_release, new.ball_free_since_pts
    if free is None or t - free > RIM_SHOT_FLIGHT_S:
        return
    if rel is not None and rel.t_pts <= free and free - rel.t_pts <= RIM_SHOT_RELEASE_GAP_S:
        if new.last_shot is not None and new.last_shot.release_pts == rel.t_pts:
            return
        new.last_shot = ShotRecord(
            free, rel.track_id, rel.team, rel.zone, rel.feet_xy,
            clock_running=new.clock_running, free_throw=new.last_free_throw_pts == rel.t_pts,
            release_pts=rel.t_pts,
        )
        return
    # No release on record for this flight: the ball left the holder's hands past another
    # player's box, or while the detector had lost it. The last holder shot it.
    held = new.last_held
    if held is None or not 0.0 <= free - held[2] <= RIM_SHOT_RELEASE_GAP_S:
        return
    if new.last_shot is not None and new.last_shot.t_pts >= held[2]:
        return
    team = held[1] if held[1] in ("home", "away") else new.possession
    new.last_shot = ShotRecord(free, held[0], team, "unknown", None, clock_running=new.clock_running, release_pts=free)


def _apply_tip(new: GameState, vf, t: float, prev_y: float | None) -> None:
    """A miss that comes down into a player's reach and goes straight back up to the rim is
    another attempt, by the player it came off: tips and putbacks have no holder and no
    release. The ball must have dropped below the rim and risen again, so a rebound
    carried down, or a ball bouncing on the rim, is none."""
    ri, touch, free = new.rim_interaction, new.ball_touch, new.ball_free_since_pts
    if ri is None or ri.entered_pts is None or ri.crossed_pts is not None:
        return
    y, width = vf.ball.xy[1], vf.rim[2] - vf.rim[0]
    low = y if new.rim_low_y is None else max(new.rim_low_y, y)
    new.rim_low_y = low
    if free is None or touch is None or not ri.entered_pts < touch[2] <= free or free - touch[2] > TIP_TOUCH_S:
        return
    if prev_y is None or y > prev_y - RISING_PX:
        return  # not on its way up
    if not _inside(vf.ball.xy, rim_region(vf.rim)):
        return
    if low < box_center(vf.rim)[1] + TIP_BELOW_WIDTHS * width or low - y < TIP_RISE_WIDTHS * width:
        return
    team = touch[1] if touch[1] in ("home", "away") else new.possession
    new.last_shot = ShotRecord(free, touch[0], team, "two", None, clock_running=new.clock_running, release_pts=free)
    new.rim_interaction = RimInteraction(entered_pts=t)
    new.rim_low_y = y


def _apply_rim_interaction(prev: GameState, new: GameState, vf, t: float) -> None:
    if vf.rim is None or vf.ball is None or vf.ball.source != "detected":
        return
    ri = new.rim_interaction if new.rim_interaction is not None else RimInteraction()
    region = rim_region(vf.rim)
    inside = _inside(vf.ball.xy, region)
    was_inside = prev.ball is not None and prev.rim is not None and _inside(prev.ball.xy, rim_region(prev.rim))
    if inside and ri.entered_pts is None:
        ri.entered_pts = t
        new.rim_low_y = None
        free = new.ball_free_since_pts
        if free is not None and t - free <= RIM_SHOT_FLIGHT_S:
            new.last_rim_flight_pts = free
        _shot_from_rim_entry(new, t)
    if was_inside and not inside and ri.entered_pts is not None and ri.left_pts is None:
        ri.left_pts = t
    x, y = vf.ball.xy
    in_span = vf.rim[0] <= x <= vf.rim[2]
    if in_span and y < vf.rim[1]:
        ri.above_pts = t
    elif (
        in_span
        and y > vf.rim[3]
        and ri.above_pts is not None
        and t - ri.above_pts <= RIM_CROSS_WINDOW_S
        and ri.crossed_pts is None
    ):
        ri.crossed_pts = t
    new.rim_interaction = ri
    _apply_tip(new, vf, t, prev.ball.xy[1] if prev.ball is not None and prev.ball.source == "detected" else None)


def _apply_lane(prev: GameState, new: GameState, league: LeagueConfig, t: float) -> None:
    if new.clock_running:
        new.lane_formation, new.free_throws_pending = None, 0
        return
    lf = lane_formation(new.tracks, new.holder, league, t)
    if lf is None and new.holder is None and prev.lane_formation is not None:
        lf = prev.lane_formation  # the shooter just released: keep the formation
    elif lf is not None and prev.lane_formation is not None and prev.lane_formation.shooter_track_id == lf.shooter_track_id:
        lf = prev.lane_formation
    new.lane_formation = lf
    new.free_throws_pending = 1 if lf is not None else 0
