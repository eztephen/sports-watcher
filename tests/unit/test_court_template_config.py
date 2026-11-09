import math
from pathlib import Path

import numpy as np

from sports_watcher.config import load_league
from sports_watcher.perception.court import load_template
from tests.helpers.leagues import fixture_league

TEMPLATE = Path("configs/court_keypoints/basketball-court-detection-2.yaml")


def _by_name(league):
    names, xy = load_template(TEMPLATE, league)
    return names, dict(zip(names, map(tuple, xy.tolist()), strict=True))


def test_court_template_has_33_unique_points_inside_the_court():
    names, xy = load_template(TEMPLATE, fixture_league("pba"))
    assert len(names) == 33 and len(set(names)) == 33
    assert (xy[:, 0] >= 0).all() and (xy[:, 0] <= 28.0).all()
    assert (xy[:, 1] >= 0).all() and (xy[:, 1] <= 15.0).all()


def test_court_template_features_follow_the_league_geometry():
    # court frame: x along the length from the left baseline, y from the near sideline
    _, p = _by_name(fixture_league("pba"))
    assert p["01"] == (0.0, 15.0) and p["08"] == (0.0, 0.0)  # left corners, far and near
    assert p["34"] == (28.0, 15.0) and p["41"] == (28.0, 0.0)  # right corners
    assert p["21"] == (14.0, 7.5) and p["19"] == (14.0, 15.0) and p["23"] == (14.0, 0.0)
    assert p["09"] == (1.575, 7.5) and p["33"] == (26.425, 7.5)  # hoops
    assert p["13"] == (5.8, 7.5) and p["29"] == (22.2, 7.5)  # free-throw line midpoints
    assert p["12"] == (5.8, 7.5 + 2.45) and p["14"] == (5.8, 7.5 - 2.45)  # free-throw line ends
    assert p["04"] == (0.0, 7.5 + 2.45) and p["38"] == (28.0, 7.5 - 2.45)  # key corners on the baseline
    assert p["02"] == (0.0, 7.5 + 6.6) and p["40"] == (28.0, 7.5 - 6.6)  # three-point line ends
    assert np.allclose(p["16"], (1.575 + 6.75, 7.5)) and np.allclose(p["26"], (26.425 - 6.75, 7.5))  # arc apexes
    jx = 1.575 + math.sqrt(6.75**2 - 6.6**2)
    assert np.allclose(p["10"], (jx, 7.5 + 6.6)) and np.allclose(p["11"], (jx, 7.5 - 6.6))  # arc meets the straight segment
    assert np.allclose(p["31"], (28.0 - jx, 7.5 + 6.6)) and np.allclose(p["32"], (28.0 - jx, 7.5 - 6.6))
    assert p["15"] == (8.53, 15.0) and p["17"] == (8.53, 0.0)  # sideline marks 8.53 m (28 ft) from the baseline
    assert p["25"] == (28.0 - 8.53, 15.0) and p["27"] == (28.0 - 8.53, 0.0)


def test_court_template_scales_to_the_nba_court():
    _, p = _by_name(load_league("nba"))
    assert p["34"] == (28.65, 15.24) and p["33"] == (28.65 - 1.6, 7.62)
    assert p["12"] == (5.79, 7.62 + 2.44) and np.allclose(p["16"], (1.6 + 7.24, 7.62))


DATASET_ORDER = [
    "01", "02", "04", "05", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "19", "21",
    "23", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "37", "38", "40", "41",
]


def test_court_template_lists_the_dataset_keypoints_in_order():
    names, _ = load_template(TEMPLATE, fixture_league("pba"))
    assert names == DATASET_ORDER
