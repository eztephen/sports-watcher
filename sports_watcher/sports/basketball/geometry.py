"""Court geometry in metres: baskets, zones, the key.

Court frame: x along the length from the left baseline (0 to ``court_length_m``), y along
the width from the near sideline (0 to ``court_width_m``). A ``side`` names the basket a
play is aimed at.
"""

from __future__ import annotations

import math
from typing import Literal

from sports_watcher.config import LeagueConfig

Side = Literal["left", "right"]
Zone = Literal["two", "three"]
XY = tuple[float, float]


def basket_xy(league: LeagueConfig, side: Side) -> XY:
    y = league.court_width_m / 2.0
    if side == "left":
        return (league.basket_from_baseline_m, y)
    return (league.court_length_m - league.basket_from_baseline_m, y)


def free_throw_xy(league: LeagueConfig, side: Side) -> XY:
    y = league.court_width_m / 2.0
    if side == "left":
        return (league.free_throw_line_from_baseline_m, y)
    return (league.court_length_m - league.free_throw_line_from_baseline_m, y)


def centre_xy(league: LeagueConfig) -> XY:
    return (league.court_length_m / 2.0, league.court_width_m / 2.0)


def nearest_side(court_xy: XY, league: LeagueConfig) -> Side:
    return "left" if court_xy[0] < league.court_length_m / 2.0 else "right"


def near(a: XY, b: XY, radius_m: float) -> bool:
    return math.dist(a, b) <= radius_m


def _depth_from_basket(court_xy: XY, league: LeagueConfig, side: Side) -> float:
    """Distance along the court length from the basket toward the far end (negative behind)."""
    bx, _ = basket_xy(league, side)
    return court_xy[0] - bx if side == "left" else bx - court_xy[0]


def zone_for(court_xy: XY, league: LeagueConfig, side: Side) -> Zone:
    """Two or three points for a shot from ``court_xy`` at the ``side`` basket.

    The straight corner segments sit ``three_point_corner_m`` from the basket's centre
    line and run from the baseline to where the arc meets them; beyond that depth the arc
    of ``three_point_radius_m`` applies.
    """
    bx, by = basket_xy(league, side)
    r, c = league.three_point_radius_m, league.three_point_corner_m
    corner_depth = math.sqrt(max(r * r - c * c, 0.0))
    if _depth_from_basket(court_xy, league, side) <= corner_depth:
        return "three" if abs(court_xy[1] - by) >= c else "two"
    return "three" if math.dist(court_xy, (bx, by)) >= r else "two"


def in_lane_edges(court_xy: XY, league: LeagueConfig, side: Side, margin_m: float = 1.0) -> bool:
    """Within ``margin_m`` of either long edge of the key, between baseline and free-throw line."""
    x, y = court_xy
    ft_x = free_throw_xy(league, side)[0]
    inside_depth = (0.0 <= x <= ft_x) if side == "left" else (ft_x <= x <= league.court_length_m)
    if not inside_depth:
        return False
    half = league.key_width_m / 2.0
    return abs(abs(y - league.court_width_m / 2.0) - half) <= margin_m
