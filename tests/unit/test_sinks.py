import queue

from sports_watcher.events.model import Event
from sports_watcher.events.sinks import Broadcaster, JsonlSink, MemorySink, QueueSink
from tests.helpers.events import make_event


def test_memory_and_queue_sinks_keep_order():
    mem, q = MemorySink(), queue.Queue()
    qs = QueueSink(q)
    e1, e2 = make_event(seq=1), make_event(seq=2)
    for s in (mem, qs):
        s.emit(e1)
        s.emit(e2)
    assert [e.seq for e in mem.events] == [1, 2]
    assert q.get_nowait() is e1 and q.get_nowait() is e2


def test_jsonl_sink_appends_one_json_line_per_event(tmp_path):
    path = tmp_path / "e.jsonl"
    sink = JsonlSink(path)
    sink.emit(make_event(seq=1))
    sink.emit(make_event(seq=2))
    sink.close()
    lines = path.read_text().splitlines()
    assert [Event.model_validate_json(line).seq for line in lines] == [1, 2]


def test_broadcaster_fans_out_and_drops_when_a_subscriber_is_full():
    b = Broadcaster(maxsize=1)
    q1, q2 = b.subscribe(), b.subscribe()
    b.emit(make_event(seq=1))
    b.emit(make_event(seq=2))  # q1 and q2 are full: dropped, no exception
    assert q1.get_nowait().seq == 1 and q2.get_nowait().seq == 1
    b.unsubscribe(q2)
    b.emit(make_event(seq=3))
    assert q1.get_nowait().seq == 3 and q2.empty()
