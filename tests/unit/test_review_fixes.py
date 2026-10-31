"""Tests added in the final-review fix pass. Each pins a finding from the whole-branch review."""

from __future__ import annotations

import itertools
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from sports_watcher.clock import LatencyClass, StreamClock
from sports_watcher.config import load_league
from sports_watcher.events.engine import EventEngine
from sports_watcher.events.model import EventStatus, SourceInfo
from sports_watcher.ingest.base import Frame, SourceSpec
from sports_watcher.ingest.factory import register_source
from sports_watcher.ingest.latest_frame import LatestFrameReader
from sports_watcher.ingest.url import UrlSource
from sports_watcher.perception.scoreboard import ScoreboardReader
from sports_watcher.sports.base import Observation
from sports_watcher.sports.basketball.plugin import BasketballPlugin
from sports_watcher.watcher import GameWatcher
from tests.helpers.scoreboard_images import B1, B2, BLANK, BROADCASTER, boxes
from tests.helpers.sources import FailingStubSource, StubSource
from tests.helpers.text_readers import ScriptedTextReader, SeqTextReader

PBA = load_league("pba")
NOW = datetime(2026, 1, 1, tzinfo=UTC)
GOOD = [("78", B1, 0.9), ("65", B2, 0.9)]
BAD = [("73", B1, 0.9), ("65", B2, 0.9)]


def _no_orphans(events):
    targets = {e.supersedes for e in events if e.supersedes}
    return all(e.id in targets for e in events if e.status == EventStatus.provisional)


# --- Review issue 1 (Critical), 7 and 9: voting confidence -------------------------------


def test_first_reads_alone_are_not_confident():
    r = ScoreboardReader(SeqTextReader([GOOD] * 5), broadcaster=BROADCASTER, vote_window=5)
    confs = [r.read(BLANK).confidence for _ in range(4)]
    assert confs == pytest.approx([0.2, 0.4, 0.6, 0.8])


def test_startup_misread_never_confirms_a_score_change():
    reader = ScoreboardReader(SeqTextReader([BAD] + [GOOD] * 20), broadcaster=BROADCASTER, vote_window=5)
    clock = StreamClock(LatencyClass.FILE)
    eng = EventEngine(BasketballPlugin(PBA), "g", SourceInfo(adapter="file", latency_class=LatencyClass.FILE), clock)
    emitted = []
    for i in range(21):
        state = reader.read(BLANK)
        obs = Observation(frame_index=i, t_pts=i * 0.2, t_wall=NOW + timedelta(seconds=i * 0.2), scoreboard=state)
        emitted += eng.step(obs)
    emitted += eng.finish()
    assert not [e for e in emitted if e.type == "score_change" and e.status == EventStatus.confirmed]
    assert eng.state.score() == (78, 65)


def test_unreadable_period_does_not_block_score_agreement():
    with_period = [("78", B1, 0.9), ("65", B2, 0.9), ("3rd", (100, 30, 30, 15), 0.9)]
    without = GOOD
    seq = [with_period, without, without, with_period, without]
    r = ScoreboardReader(SeqTextReader(seq), broadcaster=BROADCASTER, vote_window=5)
    states = [r.read(BLANK) for _ in seq]
    assert states[-1].confidence == 1.0 and states[-1].home == 78 and states[-1].period == 3


def test_all_blank_window_has_zero_confidence():
    r = ScoreboardReader(SeqTextReader([[]] * 5), broadcaster=BROADCASTER, vote_window=5)
    states = [r.read(BLANK) for _ in range(5)]
    assert states[-1].confidence == 0.0 and states[-1].home is None


# --- Review issue 2 (Important): live frames carry arrival time -------------------------


def test_decode_frames_uses_arrival_time_for_live_sources(sample_video):
    from sports_watcher.ingest._decode import decode_frames, open_container

    start = NOW
    with open_container(str(sample_video)) as c:
        live = list(decode_frames(c, start, live=True))
    with open_container(str(sample_video)) as c:
        offline = list(decode_frames(c, start, live=False))
    assert all(b.wall >= a.wall for a, b in itertools.pairwise(live))
    assert all(abs((f.wall - datetime.now(UTC)).total_seconds()) < 30 for f in live)
    assert offline[0].wall == start and offline[10].wall > start


def test_url_source_is_live_for_streams_only():
    assert UrlSource("rtsp://h/x").live is True and UrlSource("/tmp/a.mp4").live is False
    assert UrlSource("rtsp://h/x").open_kwargs()["timeout"] == (10.0, 10.0)
    assert "timeout" not in UrlSource("/tmp/a.mp4").open_kwargs()


# --- Review issue 3 (Important): KeyboardInterrupt still finishes the job ---------------


def test_keyboard_interrupt_retracts_pending_and_marks_failed(monkeypatch):
    register_source("stub", lambda spec: StubSource(n=1000))
    reader = ScriptedTextReader({0: boxes(0, 0, "10:00", "1st"), 5: boxes(2, 0, "9:50", "1st")})
    w = GameWatcher(SourceSpec(kind="stub"), PBA, "one_sports", game_id="ki", home="A", away="B",
                    text_reader=reader, ocr_every_n_frames=1)
    original = w._process

    def interrupting(frame, dropped):
        original(frame, dropped)
        pending = any(e.status == EventStatus.provisional and e.type == "score_change" for e in w.events)
        if pending and frame.index >= 8:
            raise KeyboardInterrupt

    monkeypatch.setattr(w, "_process", interrupting)
    with pytest.raises(KeyboardInterrupt):
        w.run()
    st = w.status()
    assert st.state == "failed" and "KeyboardInterrupt" in (st.error or "")
    assert _no_orphans(w.events) and any(e.reason == "stream_ended" for e in w.events)


# --- Review issue 4 (Important): stop waits for the producer thread ---------------------


def test_latest_frame_reader_stop_waits_for_a_slow_producer():
    def slow():
        for i in range(50):
            time.sleep(0.05)
            yield Frame(index=i, pts_s=i / 25, wall=NOW, image=BLANK)

    r = LatestFrameReader(slow()).start()
    r.read(timeout=1.0)
    r.stop(join_timeout=3.0)
    assert r.producer_alive is False


# --- Review issue 6 (Important): reconnect budget is per outage -------------------------


def test_reconnect_budget_resets_after_a_healthy_stretch():
    attempts = iter([FailingStubSource(fail_after=150, n=400) for _ in range(4)] + [StubSource(n=50)])
    register_source("stub", lambda spec: next(attempts))
    w = GameWatcher(SourceSpec(kind="stub", url="rtsp://x"), PBA, "one_sports", game_id="r1", home="A", away="B",
                    text_reader=ScriptedTextReader({}), max_reconnects=2, reconnect_backoff_s=0.01,
                    healthy_frames_to_reset=100)
    w.run()
    st = w.status()
    assert st.state == "ended" and st.reconnects == 4 and st.frames_processed == 4 * 150 + 50
    assert len([e for e in w.events if e.type == "stream_gap"]) == 4


def test_consecutive_failures_exhaust_the_budget():
    register_source("stub", lambda spec: FailingStubSource(fail_after=0, n=10))
    w = GameWatcher(SourceSpec(kind="stub", url="rtsp://x"), PBA, "one_sports", game_id="r2", home="A", away="B",
                    text_reader=ScriptedTextReader({}), max_reconnects=2, reconnect_backoff_s=0.01)
    w.run()
    st = w.status()
    assert st.state == "failed" and st.reconnects == 2 and "connection reset" in (st.error or "")


# keep the threading import honest for the slow-producer test on some platforms
assert threading
