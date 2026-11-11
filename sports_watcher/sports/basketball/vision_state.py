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
MIN_SCOREBOARD_CONFIDENCE = 0.6
RIM_REGION_WIDTHS = 2.0
SHOT_GATE_WIDTHS = 1.5
SHOT_FIT_OBS = 4
SHOT_WINDOW_S = 1.5
BALL_PATH_MAX_S = 1.5
RIM_CROSS_WINDOW_S = 0.5
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
    since_read = t - new.last_scoreboard_pts if new.last_scoreboard_pts is not None else float("inf")
    since_change = (
        t - new.last_clock_change_pts if new.last_clock_change_pts is not None else float("inf")
    )
    if since_read <= CLOCK_RUNNING_S:
        new.clock_state = "read"
        new.clock_running = since_change <= CLOCK_RUNNING_S
    elif (
        obs.camera_state == "main"
        and since_read <= CLOCK_RUNNING_S + ASSUMED_MAX_S
        and prev.clock_state in ("read", "assumed")
    ):
        new.clock_state = "assumed"
        new.clock_running = prev.clock_running
    else:
        new.clock_state = "unknown"
        new.clock_running = False
    if new.clock_running and not prev.clock_running:
        new.clock_running_since_pts = t


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
    _apply_possession(new)
    _apply_release(prev, new, vf, t, league)
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
        # without being in anyone's hands. No candidate until it stops rising.
        new.holder, new.holder_candidate = None, None
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
    if state.possession_since_pts is not None and state.possession_since_pts > ri.crossed_pts:
        return None
    return ("away" if shot.team == "home" else "home"), ri.crossed_pts


def _apply_possession(new: GameState) -> None:
    h = new.holder
    if h is None:
        return
    scored_on = _scored_on(new)
    if scored_on is not None and h.since_pts > scored_on[1]:
        # After a basket the ball belongs to the team scored on, whatever label the player
        # who picks it up carries (team labels are least reliable under the basket).
        new.possession, new.possession_since_pts = scored_on[0], h.since_pts
        return
    if h.since_pts == new.possession_since_pts:
        return  # the hold that started this possession does not flip it back
    if h.team in ("home", "away") and h.team != new.possession:
        new.possession = h.team
        new.possession_since_pts = h.since_pts


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


def _apply_ball_path(new: GameState, vf, t: float) -> None:
    rel = new.last_release
    if rel is None or vf.ball is None or vf.ball.source != "detected":
        return
    if t - rel.t_pts > BALL_PATH_MAX_S:
        return
    if not new.ball_path or new.ball_path[-1][0] != t:
        new.ball_path.append((t, vf.ball.xy[0], vf.ball.xy[1]))
    already = new.last_shot is not None and new.last_shot.t_pts == rel.t_pts
    if already or vf.rim is None or len(new.ball_path) < SHOT_FIT_OBS:
        return
    if shot_path_passes_rim(new.ball_path, vf.rim):
        free_throw = new.last_free_throw_pts == rel.t_pts
        new.last_shot = ShotRecord(
            rel.t_pts, rel.track_id, rel.team, rel.zone, rel.feet_xy,
            clock_running=new.clock_running, free_throw=free_throw,
        )


def _apply_rim_interaction(prev: GameState, new: GameState, vf, t: float) -> None:
    if vf.rim is None or vf.ball is None or vf.ball.source != "detected":
        return
    ri = new.rim_interaction if new.rim_interaction is not None else RimInteraction()
    region = rim_region(vf.rim)
    inside = _inside(vf.ball.xy, region)
    was_inside = prev.ball is not None and prev.rim is not None and _inside(prev.ball.xy, rim_region(prev.rim))
    if inside and ri.entered_pts is None:
        ri.entered_pts = t
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
