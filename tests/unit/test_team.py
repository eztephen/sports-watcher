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


def test_align_labels_to_previous_centroids():
    from sports_watcher.perception.team import align_labels

    prev = np.array([[0.0, 0.0], [10.0, 10.0]])
    assert align_labels(prev, np.array([[10.1, 9.9], [0.2, -0.1]])) == {0: 1, 1: 0}
    assert align_labels(prev, np.array([[0.1, 0.1], [9.0, 11.0]])) == {0: 0, 1: 1}


def test_refit_without_colours_keeps_the_mapping():
    a = TeamAssigner(fake_embed, None, None, fit_after=20, refit_every_s=30.0)
    for i in range(10):
        a.observe(i, RED, 0.0)
        a.observe(100 + i, BLUE, 0.0)
    first = (a.team_of(0), a.team_of(100))
    for t in range(1, 40):  # more samples, blue first this time, through a refit at 30 s
        a.observe(100 + (t % 10), BLUE, float(t))
        a.observe(t % 10, RED, float(t))
    assert (a.team_of(0), a.team_of(100)) == first and first[0] != first[1]


def _player_crop(jersey_bgr, rng):
    """A player-like crop: floor background, a jersey block over the torso, skin and dark shorts."""
    crop = np.full((120, 48, 3), (120, 170, 200), np.uint8)  # wooden floor
    crop[10:28, 16:32] = (120, 150, 200)  # head
    crop[28:70, 8:40] = jersey_bgr
    crop[70:90, 10:38] = jersey_bgr
    crop[90:118, 14:34] = (110, 140, 190)  # legs
    noise = rng.integers(-12, 13, crop.shape)
    return np.clip(crop.astype(int) + noise, 0, 255).astype(np.uint8)


def test_jersey_features_split_white_from_blue_kits():
    from sports_watcher.perception.team import jersey_features

    rng = np.random.default_rng(0)
    a = TeamAssigner(jersey_features, parse_color("white"), parse_color("navy"), fit_after=40)
    for i in range(20):
        a.observe(i, _player_crop((245, 245, 245), rng), 0.0)
        a.observe(100 + i, _player_crop((200, 60, 30), rng), 0.0)  # royal blue, BGR
    assert a.fitted
    assert {a.team_of(i) for i in range(20)} == {"home"}
    assert {a.team_of(100 + i) for i in range(20)} == {"away"}


WHITE_KIT, BLUE_KIT = (245, 245, 245), (200, 60, 30)  # BGR


def _side_on_crop(jersey_bgr, rng, floor=(120, 170, 200)):
    """A player seen side-on: the shirt is a narrow strip and the floor fills most of the torso window."""
    crop = np.full((120, 48, 3), floor, np.uint8)  # wooden floor unless painted
    crop[10:28, 18:30] = (120, 150, 200)  # head
    crop[28:90, 19:29] = jersey_bgr
    crop[90:118, 19:29] = (110, 140, 190)  # legs
    return np.clip(crop.astype(int) + rng.integers(-12, 13, crop.shape), 0, 255).astype(np.uint8)


def _two_kit_crop(rng):
    crop = _player_crop(WHITE_KIT, rng)
    crop[28:70, 12:24] = BLUE_KIT  # a second player across half the torso window
    return crop


def _fitted_kits(rng, **kw):
    from sports_watcher.perception.team import KitAssigner

    a = KitAssigner(parse_color("white"), parse_color("navy"), fit_after=60, **kw)
    for i in range(15):
        a.observe(i, _player_crop(WHITE_KIT, rng), 0.0)
        a.observe(100 + i, _player_crop(BLUE_KIT, rng), 0.0)
        a.observe(200 + i, _side_on_crop(WHITE_KIT, rng), 0.0)
        a.observe(300 + i, _side_on_crop(BLUE_KIT, rng), 0.0)
    assert a.fitted
    return a


def test_kit_assigner_is_unknown_before_the_fit_then_names_both_kits():
    from sports_watcher.perception.team import KitAssigner

    rng = np.random.default_rng(0)
    a = KitAssigner(parse_color("white"), parse_color("navy"), fit_after=10**6)
    a.observe(1, _player_crop(WHITE_KIT, rng), 0.0)
    assert a.fitted is False and a.team_of(1) == "unknown"
    a = _fitted_kits(rng)
    assert {a.team_of(i) for i in range(15)} == {"home"}
    assert {a.team_of(100 + i) for i in range(15)} == {"away"}


def test_floor_in_the_torso_window_does_not_read_as_the_white_kit():
    rng = np.random.default_rng(0)
    old = TeamAssigner(__import__("sports_watcher.perception.team", fromlist=["x"]).jersey_features, parse_color("white"), parse_color("navy"), fit_after=60)
    for i in range(15):
        old.observe(i, _player_crop(WHITE_KIT, rng), 0.0)
        old.observe(100 + i, _player_crop(BLUE_KIT, rng), 0.0)
        old.observe(200 + i, _side_on_crop(WHITE_KIT, rng), 0.0)
        old.observe(300 + i, _side_on_crop(BLUE_KIT, rng), 0.0)
    assert "home" in {old.team_of(300 + i) for i in range(15)}  # the median colour is the floor's
    a = _fitted_kits(np.random.default_rng(0))
    assert {a.team_of(300 + i) for i in range(15)} == {"away"}
    assert {a.team_of(200 + i) for i in range(15)} == {"home"}


def test_a_crop_with_both_kits_or_neither_does_not_vote():
    rng = np.random.default_rng(0)
    a = _fitted_kits(rng)
    a.observe(500, _two_kit_crop(rng), 10.0)
    assert a.team_of(500) == "unknown"
    a.observe(501, np.full((120, 48, 3), (120, 170, 200), np.uint8), 10.0)  # floor only
    assert a.team_of(501) == "unknown"
    a.observe(500, _player_crop(WHITE_KIT, rng), 10.1)  # the abstention did not use the sample slot
    assert a.team_of(500) == "home"


def test_kit_assigner_keeps_the_majority_and_samples_once_a_second():
    rng = np.random.default_rng(0)
    a = _fitted_kits(rng)
    for i in range(5):
        a.observe(600, _player_crop(BLUE_KIT, rng), 10.0 + i)
    a.observe(600, _player_crop(WHITE_KIT, rng), 15.0)
    assert a.team_of(600) == "away"
    a.observe(700, _player_crop(BLUE_KIT, rng), 20.0)
    a.observe(700, _player_crop(WHITE_KIT, rng), 20.5)  # inside the second: not sampled
    assert a.team_of(700) == "away"


def test_kit_assigner_reset_forgets_everything():
    a = _fitted_kits(np.random.default_rng(0))
    a.reset()
    assert a.fitted is False and a.team_of(0) == "unknown"


def test_white_court_paint_behind_a_player_is_not_his_kit():
    rng = np.random.default_rng(0)
    a = _fitted_kits(rng)
    for i in range(3):  # a blue dribbler leaning over the white centre logo
        a.observe(900, _side_on_crop(BLUE_KIT, rng, floor=WHITE_KIT), 40.0 + i)
    assert a.team_of(900) == "away"
    a.observe(901, _side_on_crop(WHITE_KIT, rng, floor=WHITE_KIT), 40.0)  # nothing tells shirt from paint
    assert a.team_of(901) == "unknown"
    a.observe(902, _side_on_crop(WHITE_KIT, rng, floor=BLUE_KIT), 40.0)  # white shirt over blue paint
    assert a.team_of(902) == "home"
