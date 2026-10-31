import pytest

from sports_watcher.perception.scoreboard import parse_clock, parse_period, parse_scoreboard


@pytest.mark.parametrize(
    "s,v",
    [("10:00", 600.0), ("7:05", 425.0), ("0:59.8", 59.8), ("59.8", 59.8), (":45", 45.0),
     ("12:00", 720.0), ("78", None), ("Q3", None), ("", None)],
)
def test_parse_clock(s, v):
    assert parse_clock(s) == v


@pytest.mark.parametrize(
    "s,v",
    [("1st", 1), ("Q1", 1), ("1", 1), ("2nd", 2), ("3rd", 3), ("4TH", 4), ("OT", 5),
     ("2OT", 6), ("78", None), ("", None)],
)
def test_parse_period(s, v):
    assert parse_period(s) == v


def test_parse_scoreboard_home_left():
    boxes = [
        ("78", (10, 0, 40, 30), 0.9),
        ("65", (200, 0, 40, 30), 0.9),
        ("7:31", (100, 0, 60, 20), 0.9),
        ("3rd", (100, 30, 30, 15), 0.9),
        ("14", (300, 0, 20, 15), 0.9),
    ]
    st = parse_scoreboard(boxes, home_side="left", shot_clock_visible=True)
    assert (st.home, st.away, st.clock_s, st.period, st.shot_clock_s) == (78, 65, 451.0, 3, 14.0)
    assert st.raw == ["78", "65", "7:31", "3rd", "14"]


def test_parse_scoreboard_home_right_and_no_shot_clock():
    boxes = [("12", (10, 0, 40, 30), 0.9), ("9", (200, 0, 40, 30), 0.9), ("0:59.8", (100, 0, 60, 20), 0.9)]
    st = parse_scoreboard(boxes, home_side="right", shot_clock_visible=False)
    assert (st.home, st.away, st.clock_s, st.period, st.shot_clock_s) == (9, 12, 59.8, None, None)


def test_parse_scoreboard_with_nothing_readable():
    st = parse_scoreboard([], home_side="left", shot_clock_visible=True)
    assert st.home is None and st.away is None and st.confidence == 0.0
