from datetime import UTC, datetime, timedelta

from sports_watcher.clock import LatencyClass, StreamClock
from sports_watcher.config import load_league
from sports_watcher.events.engine import EventEngine
from sports_watcher.events.model import EventStatus, SourceInfo
from sports_watcher.events.sinks import MemorySink
from sports_watcher.perception.scoreboard import ScoreboardState
from sports_watcher.sports.base import Observation
from sports_watcher.sports.basketball.plugin import BasketballPlugin

NOW = datetime(2026, 1, 1, tzinfo=UTC)
SOURCE = SourceInfo(adapter="file", latency_class=LatencyClass.FILE)


def make_engine(sinks=()):
    clock = StreamClock(LatencyClass.FILE)
    return EventEngine(BasketballPlugin(load_league("pba")), "g1", SOURCE, clock, ttl_s=4.0, sinks=sinks)


def obs(i, t, h, a, c, p, conf=1.0):
    sb = ScoreboardState(home=h, away=a, clock_s=c, period=p, confidence=conf, raw=[])
    return Observation(frame_index=i, t_pts=t, t_wall=NOW + timedelta(seconds=t), scoreboard=sb)


def blank(i, t):
    return Observation(frame_index=i, t_pts=t, t_wall=NOW + timedelta(seconds=t))


def test_provisional_then_confirmed_with_supersedes():
    eng = make_engine()
    eng.step(obs(0, 0.0, 0, 0, 600, 1))
    e2 = eng.step(obs(1, 1.0, 2, 0, 598, 1))
    prov = next(e for e in e2 if e.type == "score_change")
    assert prov.status == EventStatus.provisional and prov.team == "home"
    assert prov.t_wall == NOW + timedelta(seconds=1.0) and prov.behind_live_s is None
    later = eng.step(obs(2, 1.2, 2, 0, 597, 1)) + eng.step(obs(3, 1.4, 2, 0, 596, 1))
    conf = next(e for e in later if e.supersedes == prov.id)
    assert conf.status == EventStatus.confirmed and conf.seq > prov.seq
    assert conf.type == "score_change" and conf.evidence == prov.evidence and conf.t_pts == 1.0
    assert eng.state.score() == (2, 0)


def test_retracted_when_scoreboard_flips_back():
    eng = make_engine()
    eng.step(obs(0, 0.0, 0, 0, 600, 1))
    prov = next(e for e in eng.step(obs(1, 1.0, 2, 0, 598, 1)) if e.type == "score_change")
    out = eng.step(obs(2, 1.2, 0, 0, 597, 1))
    retr = next(e for e in out if e.supersedes == prov.id)
    assert retr.status == EventStatus.retracted and retr.reason == "scoreboard_disagrees"


def test_ttl_expiry_retracts():
    eng = make_engine()
    eng.step(obs(0, 0.0, 0, 0, 600, 1))
    prov = next(e for e in eng.step(obs(1, 1.0, 2, 0, 598, 1)) if e.type == "score_change")
    assert eng.step(blank(2, 2.0)) == [] and eng.step(blank(3, 4.9)) == []
    out = eng.step(blank(4, 5.1))
    retr = next(e for e in out if e.supersedes == prov.id)
    assert retr.status == EventStatus.retracted and retr.reason == "ttl"


def test_finish_retracts_pending_with_stream_ended():
    eng = make_engine()
    eng.step(obs(0, 0.0, 0, 0, 600, 1))
    prov = next(e for e in eng.step(obs(1, 1.0, 2, 0, 598, 1)) if e.type == "score_change")
    out = eng.finish()
    assert [e.reason for e in out] == ["stream_ended"] and out[0].supersedes == prov.id
    assert eng.finish() == []


def test_period_start_is_confirmed_by_the_next_read():
    eng = make_engine()
    first = eng.step(obs(0, 0.0, 0, 0, 600, 1))
    assert [(e.type, e.status) for e in first] == [("period_start", EventStatus.provisional)]
    second = eng.step(obs(1, 0.2, 0, 0, 599.8, 1))
    assert [(e.type, e.status) for e in second] == [("period_start", EventStatus.confirmed)]


def test_seq_strictly_increasing_and_every_event_reaches_sinks(tmp_path):
    from sports_watcher.events.sinks import JsonlSink

    mem, path = MemorySink(), tmp_path / "events.jsonl"
    eng = make_engine(sinks=[mem, JsonlSink(path)])
    for i, (t, h, a, c) in enumerate([(0, 0, 0, 600), (1, 2, 0, 598), (1.2, 2, 0, 597), (1.4, 2, 0, 596), (2, 2, 3, 590)]):
        eng.step(obs(i, t, h, a, c, 1))
    eng.finish()
    seqs = [e.seq for e in mem.events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs) and len(seqs) >= 5
    lines = path.read_text().strip().splitlines()
    assert len(lines) == len(mem.events)
    assert all(e.game_id == "g1" and e.source == SOURCE for e in mem.events)
