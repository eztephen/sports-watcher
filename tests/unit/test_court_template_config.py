from pathlib import Path

import pytest

from sports_watcher.perception.court import load_template
from tests.helpers.leagues import fixture_league

TEMPLATE = Path("configs/court_keypoints/basketball-court-detection-2.yaml")


def test_court_template_has_33_unique_points_inside_the_court():
    if not TEMPLATE.exists():
        pytest.skip("court keypoint template not authored yet: needs the dataset's keypoint names")
    names, xy = load_template(TEMPLATE, fixture_league("pba"))
    assert len(names) == 33 and len(set(names)) == 33
    assert (xy[:, 0] >= 0).all() and (xy[:, 0] <= 28.0).all()
    assert (xy[:, 1] >= 0).all() and (xy[:, 1] <= 15.0).all()
