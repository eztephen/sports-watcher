from pathlib import Path

from sports_watcher.config import Settings, load_league


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.device == "cpu" and s.configs_dir.name == "configs"


def test_load_pba_league():
    pba = load_league("pba")
    assert (pba.court_length_m, pba.court_width_m) == (28.0, 15.0)
    assert pba.shot_clock_s == 24 and pba.period_length_s == 600 and pba.periods == 4
    one_sports = pba.broadcasters["one_sports"]
    assert one_sports.fields is not None and one_sports.fields.clock is not None
    assert 0 < one_sports.fields.home.y < 1 and one_sports.fields.home.y > one_sports.fields.away.y


def test_load_nba_league():
    nba = load_league("nba")
    assert nba.three_point_radius_m == 7.24 and nba.period_length_s == 720


def test_league_geometry_fields():
    pba = load_league("pba")
    assert (pba.basket_from_baseline_m, pba.free_throw_line_from_baseline_m, pba.key_width_m) == (
        1.575, 5.8, 4.9,
    )
    nba = load_league("nba")
    assert (nba.basket_from_baseline_m, nba.free_throw_line_from_baseline_m, nba.key_width_m) == (
        1.60, 5.79, 4.88,
    )


def test_settings_model_and_cache_dirs():
    s = Settings(_env_file=None)
    assert s.models_dir == Path("models") and s.cache_dir == Path("data/cache")
