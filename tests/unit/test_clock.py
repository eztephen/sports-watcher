from datetime import UTC, datetime, timedelta

from sports_watcher.clock import BEHIND_LIVE_ESTIMATE_S, LatencyClass, StreamClock


def test_wall_for_is_monotonic_across_pts_discontinuity():
    c = StreamClock(LatencyClass.HLS)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    c.observe(100.0, t0)
    c.observe(101.0, t0 + timedelta(seconds=1))
    before_jump = c.wall_for(101.0)
    c.observe(5.0, t0 + timedelta(seconds=2))  # discontinuity: pts jumped back
    assert c.wall_for(6.0) > before_jump
    assert c.wall_for(5.0) == t0 + timedelta(seconds=2)
    assert c.behind_live_s() == 15.0


def test_wall_for_interpolates_from_anchor():
    c = StreamClock(LatencyClass.FILE)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    c.observe(10.0, t0)
    assert c.wall_for(12.5) == t0 + timedelta(seconds=2.5)
    assert c.behind_live_s() is None


def test_behind_live_table_matches_spec():
    assert BEHIND_LIVE_ESTIMATE_S == {
        LatencyClass.FILE: None,
        LatencyClass.PUSH: 0.5,
        LatencyClass.PULL: 1.0,
        LatencyClass.HLS: 15.0,
        LatencyClass.CAPTURE: 0.2,
    }
