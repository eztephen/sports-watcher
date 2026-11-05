import numpy as np
import pytest

from sports_watcher.perception.team import (
    SiglipEmbedder,
    TeamAssigner,
    crop_color_lab,
    map_clusters,
    parse_color,
)

RED = np.full((100, 40, 3), (0, 0, 220), np.uint8)  # BGR
BLUE = np.full((100, 40, 3), (220, 0, 0), np.uint8)


def fake_embed(crop):
    """A 2-d embedding that separates by colour, with mild deterministic noise."""
    c = crop_color_lab(crop)
    rng = np.random.default_rng(int(abs(c.sum())))
    return np.array([c[1], c[2]]) / 128 + rng.normal(0, 0.01, 2)


def _assigner(**kw):
    return TeamAssigner(fake_embed, parse_color("red"), parse_color("blue"), **kw)


def test_unknown_before_fit_then_split_by_colour():
    a = _assigner(fit_after=20)
    for i in range(10):
        a.observe(i, RED, 0.0)
        a.observe(100 + i, BLUE, 0.0)
    assert a.fitted and a.team_of(3) == "home" and a.team_of(103) == "away"
    assert a.team_of(999) == "unknown"
    b = _assigner(fit_after=200)
    b.observe(1, RED, 0.0)
    assert b.fitted is False and b.team_of(1) == "unknown"


def test_samples_once_per_second_per_track():
    calls = []

    def counting(crop):
        calls.append(1)
        return np.zeros(2)

    a = TeamAssigner(counting, None, None, fit_after=10**6)
    for i in range(31):
        a.observe(1, RED, i / 30)
    assert len(calls) == 2  # first frame and the one-second mark


def test_majority_vote_survives_noisy_crops():
    a = _assigner(fit_after=20)
    for i in range(10):
        a.observe(i, RED, 0.0)
        a.observe(100 + i, BLUE, 0.0)
    for t in range(1, 6):
        a.observe(1, RED if t != 3 else BLUE, float(t))
    assert a.team_of(1) == "home"


def test_map_clusters_prefers_smaller_total_distance():
    red, blue = parse_color("red"), parse_color("blue")
    assert map_clusters(np.stack([blue, red]), red, blue) == {0: "away", 1: "home"}


def test_parse_color_hex_and_names():
    assert np.allclose(parse_color("#ff0000"), parse_color("red"))
    with pytest.raises(ValueError):
        parse_color("sparkly")


def test_siglip_preprocess_shape_and_range():
    x = SiglipEmbedder.preprocess(np.full((300, 120, 3), 255, np.uint8))
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32 and np.allclose(x, 1.0)
    assert np.allclose(SiglipEmbedder.preprocess(np.zeros((10, 10, 3), np.uint8)), -1.0)
