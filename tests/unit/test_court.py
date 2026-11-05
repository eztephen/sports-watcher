import cv2
import numpy as np
import yaml
from sports_watcher.perception.court import (
    CourtMapper,
    court_to_image,
    image_to_court,
    load_template,
    order_template,
)

from tests.helpers.leagues import fixture_league

L = fixture_league("pba")
GRID = np.array([(x, y) for x in (0, 5.8, 14, 22.2, 28) for y in (0, 7.5, 15)], float)  # 15 points
H_TRUE = cv2.getPerspectiveTransform(
    np.float32([(0, 0), (28, 0), (28, 15), (0, 15)]),
    np.float32([(100, 1000), (1820, 1000), (1500, 350), (420, 350)]),
)
RNG = np.random.default_rng(0)


def _img(pts, noise=0.0):
    p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), H_TRUE).reshape(-1, 2)
    return p + RNG.normal(0, noise, p.shape)


def test_recovers_homography_within_twenty_centimetres():
    m = CourtMapper(GRID)
    h = m.update(_img(GRID, 1.0), np.full(15, 0.9), 0.0)
    assert h is not None and h.inliers >= 12 and h.confidence >= 0.4
    back = np.array([image_to_court(h.matrix, tuple(p)) for p in _img(GRID)])
    assert np.abs(back - GRID).max() < 0.2
    assert np.allclose(court_to_image(h.matrix, (14.0, 7.5)), _img(np.array([[14.0, 7.5]]))[0], atol=3)


def test_too_few_confident_points_is_none():
    conf = np.full(15, 0.2)
    conf[:5] = 0.9
    assert CourtMapper(GRID).update(_img(GRID), conf, 0.0) is None


def test_outliers_are_rejected():
    pts = _img(GRID, 1.0)
    pts[[1, 7, 11]] += (300, -200)
    h = CourtMapper(GRID).update(pts, np.full(15, 0.9), 0.0)
    back = np.array([image_to_court(h.matrix, tuple(p)) for p in _img(GRID)])
    keep = [i for i in range(15) if i not in (1, 7, 11)]
    assert np.abs(back[keep] - GRID[keep]).max() < 0.3


def test_smoothing_reduces_jitter_and_current_holds_two_seconds():
    raw, smooth, m = [], [], CourtMapper(GRID)
    for i in range(10):
        pts = _img(GRID, 3.0)
        conf = np.full(15, 0.9)
        raw.append(image_to_court(CourtMapper(GRID).update(pts, conf, i * 0.2).matrix, (960.0, 600.0)))
        smooth.append(image_to_court(m.update(pts, conf, i * 0.2).matrix, (960.0, 600.0)))
    assert np.std(np.array(smooth), axis=0).sum() < np.std(np.array(raw), axis=0).sum()
    assert m.current(1.8 + 1.9) is not None and m.current(1.8 + 2.1) is None


def test_template_loading_and_ordering(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump({"keypoints": [
        {"name": "a", "x_frac": 0.0, "y_frac": 0.0}, {"name": "b", "x_frac": 0.5, "y_frac": 1.0},
    ]}))
    names, xy = load_template(p, L)
    assert names == ["a", "b"] and xy.tolist() == [[0.0, 0.0], [14.0, 15.0]]
    assert order_template(names, xy, ["b", "a"]).tolist() == [[14.0, 15.0], [0.0, 0.0]]
