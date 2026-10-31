from sports_watcher.events.model import Event, EventStatus
from tests.helpers.events import NOW, make_event


def test_confirmed_copy_supersedes_and_roundtrips():
    e = make_event(type="score_change", status=EventStatus.provisional)
    c = e.confirmed_copy(seq=e.seq + 1, emitted_wall=e.t_wall)
    assert c.supersedes == e.id and c.id != e.id and c.status == EventStatus.confirmed
    assert c.seq == e.seq + 1 and c.type == e.type and c.evidence == e.evidence
    assert c.t_pts == e.t_pts and c.t_wall == e.t_wall
    assert Event.model_validate_json(c.model_dump_json()) == c


def test_retracted_copy_carries_reason():
    r = make_event().retracted_copy(seq=2, emitted_wall=NOW, reason="ttl")
    assert r.status == EventStatus.retracted and r.reason == "ttl"


def test_events_are_frozen():
    e = make_event()
    try:
        e.seq = 99  # type: ignore[misc]
    except Exception:  # noqa: BLE001 - pydantic raises its own frozen error type
        return
    raise AssertionError("Event must be immutable")
