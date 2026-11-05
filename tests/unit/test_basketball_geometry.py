from sports_watcher.sports.basketball.geometry import (
    basket_xy,
    centre_xy,
    free_throw_xy,
    in_lane_edges,
    nearest_side,
    zone_for,
)
from tests.helpers.leagues import fixture_league

L = fixture_league("pba")


def test_fixed_points():
    assert basket_xy(L, "left") == (1.575, 7.5) and basket_xy(L, "right") == (26.425, 7.5)
    assert free_throw_xy(L, "left") == (5.8, 7.5) and centre_xy(L) == (14.0, 7.5)
    assert nearest_side((3.0, 2.0), L) == "left" and nearest_side((20.0, 2.0), L) == "right"


def test_zones():
    assert zone_for((4.0, 7.5), L, "left") == "two"  # 2.4 m from the basket
    assert zone_for((9.0, 7.5), L, "left") == "three"  # 7.4 m straight on
    assert zone_for((8.0, 7.5), L, "left") == "two"  # 6.4 m, inside the 6.75 arc
    assert zone_for((1.575, 0.5), L, "left") == "three"  # corner: 7.0 m sideways, past 6.60
    assert zone_for((1.575, 1.5), L, "left") == "two"  # corner: 6.0 m sideways


def test_lane_edges():
    assert in_lane_edges((3.0, 7.5 + 2.45), L, "left")  # on the edge
    assert in_lane_edges((3.0, 7.5 + 2.45 + 0.9), L, "left")  # 0.9 m outside
    assert not in_lane_edges((3.0, 7.5), L, "left")  # middle of the key
    assert not in_lane_edges((7.0, 7.5 + 2.45), L, "left")  # past the free-throw line
