from sports_watcher.config import load_league
from sports_watcher.events.model import EventStatus
from sports_watcher.ingest.base import SourceSpec
from sports_watcher.ingest.factory import register_source
from sports_watcher.watcher import GameWatcher
from tests.helpers.scoreboard_images import boxes
from tests.helpers.sources import FailingStubSource, StubSource
from tests.helpers.text_readers import ScriptedTextReader

PBA = load_league("pba")


def _no_orphans(events):
    terminal_targets = {e.supersedes for e in events if e.supersedes}
    return all(e.id in terminal_targets for e in events if e.status == EventStatus.provisional)


def test_watcher_emits_confirmed_score_change_and_ends(sample_video):
    reader = ScriptedTextReader({0: boxes(0, 0, "10:00", "1st"), 20: boxes(2, 0, "9:50", "1st")})
    w = GameWatcher(
        SourceSpec(kind="file", path=str(sample_video)), PBA, "one_sports",
        game_id="g1", home="TNT", away="Magnolia", text_reader=reader, ocr_every_n_frames=1,
    )
    w.run()
    types = [(e.type, e.status) for e in w.events]
    assert ("score_change", EventStatus.confirmed) in types
    assert w.status().state == "ended" and w.status().frames_processed >= 48
    assert w.status().score == (2, 0) and w.status().reconnects == 0
    assert w.firsts().first_score.team == "home"
    assert _no_orphans(w.events)
    assert w.events_since(w.events[-1].seq) == [] and w.events_since(0) == w.events


def test_watcher_reconnects_url_sources_and_emits_stream_gap():
    attempts = iter([FailingStubSource(fail_after=10, n=30), StubSource(n=20)])
    register_source("stub", lambda spec: next(attempts))
    w = GameWatcher(
        SourceSpec(kind="stub", url="rtsp://x"), PBA, "one_sports", game_id="g2",
        home="A", away="B", text_reader=ScriptedTextReader({}), max_reconnects=2,
        reconnect_backoff_s=0.01,
    )
    w.run()
    st = w.status()
    assert st.reconnects == 1 and st.state == "ended" and st.frames_processed == 30
    gaps = [e for e in w.events if e.type == "stream_gap"]
    assert len(gaps) == 1 and gaps[0].status == EventStatus.confirmed
    assert "connection reset" in (gaps[0].evidence.note or "")


def test_file_source_failure_does_not_reconnect():
    register_source("stub", lambda spec: FailingStubSource(fail_after=3, n=10))
    w = GameWatcher(
        SourceSpec(kind="stub"), PBA, "one_sports", game_id="g3", home="A", away="B",
        text_reader=ScriptedTextReader({}), reconnect_backoff_s=0.01,
    )
    w.run()
    st = w.status()
    assert st.state == "failed" and st.reconnects == 0 and "connection reset" in (st.error or "")
    assert _no_orphans(w.events)


def test_watcher_stop_ends_the_run_early():
    register_source("stub", lambda spec: StubSource(n=100000))
    w = GameWatcher(
        SourceSpec(kind="stub"), PBA, "one_sports", game_id="g4", home="A", away="B",
        text_reader=ScriptedTextReader({}),
    )
    t = w.start_in_thread()
    import time

    time.sleep(0.2)
    w.stop()
    t.join(timeout=5)
    assert not t.is_alive() and w.status().state == "ended"
