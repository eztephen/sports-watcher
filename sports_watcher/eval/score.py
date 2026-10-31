"""Precision, recall and latency per event type, plus retraction rate and firsts accuracy.

Only confirmed events are scored. A truth event matches a confirmed event of the same
type (and team, when the truth names one, and points, when both carry them) within the
tolerance; the nearest pair wins and each event is used once. Latency is the time from
the frame the event belongs to (``t_wall``) to when its confirmation was emitted.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from pydantic import BaseModel

from sports_watcher.events.firsts import Firsts
from sports_watcher.events.model import Event, EventStatus
from sports_watcher.replay.truth import TruthEvent


class TypeScore(BaseModel):
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    latency_p50_s: float | None
    latency_p95_s: float | None


class EvalReport(BaseModel):
    per_type: dict[str, TypeScore]
    retraction_rate: float
    firsts_accuracy: float | None


def _compatible(truth: TruthEvent, event: Event, tolerance_s: float) -> bool:
    if event.type != truth.type or abs(event.t_pts - truth.t_pts) > tolerance_s:
        return False
    if truth.team != "unknown" and event.team != truth.team:
        return False
    both_have_points = truth.points is not None and event.evidence.points is not None
    return not (both_have_points and truth.points != event.evidence.points)


def _confirmed(events: Iterable[Event]) -> list[Event]:
    return sorted((e for e in events if e.status == EventStatus.confirmed), key=lambda e: e.seq)


def match_events(
    events: list[Event], truth: list[TruthEvent], tolerance_s: float = 1.0
) -> list[tuple[TruthEvent, Event | None]]:
    confirmed = _confirmed(events)
    candidates = sorted(
        (abs(e.t_pts - t.t_pts), ti, ei)
        for ti, t in enumerate(truth)
        for ei, e in enumerate(confirmed)
        if _compatible(t, e, tolerance_s)
    )
    assigned: dict[int, Event] = {}
    used: set[int] = set()
    for _dt, ti, ei in candidates:
        if ti in assigned or ei in used:
            continue
        assigned[ti] = confirmed[ei]
        used.add(ei)
    return [(t, assigned.get(i)) for i, t in enumerate(truth)]


def _lookup(data: dict[str, Any], dotted: str) -> Any:
    node: Any = data
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def firsts_accuracy(firsts: Firsts, truth_firsts: dict[str, Any]) -> float | None:
    if not truth_firsts:
        return None
    data = firsts.model_dump()
    hits = sum(1 for path, expected in truth_firsts.items() if _lookup(data, path) == expected)
    return hits / len(truth_firsts)


def score(
    events: list[Event],
    truth: list[TruthEvent],
    tolerance_s: float = 1.0,
    firsts: Firsts | None = None,
    truth_firsts: dict[str, Any] | None = None,
) -> EvalReport:
    pairs = match_events(events, truth, tolerance_s)
    confirmed = _confirmed(events)
    matched_ids = {e.id for _, e in pairs if e is not None}
    types = sorted({t.type for t in truth} | {e.type for e in confirmed})
    per_type: dict[str, TypeScore] = {}
    for typ in types:
        matched = [e for t, e in pairs if t.type == typ and e is not None]
        tp = len(matched)
        fn = sum(1 for t, e in pairs if t.type == typ and e is None)
        fp = sum(1 for e in confirmed if e.type == typ and e.id not in matched_ids)
        latencies = [(e.emitted_wall - e.t_wall).total_seconds() for e in matched]
        per_type[typ] = TypeScore(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=tp / (tp + fp) if tp + fp else 0.0,
            recall=tp / (tp + fn) if tp + fn else 0.0,
            latency_p50_s=float(np.percentile(latencies, 50)) if latencies else None,
            latency_p95_s=float(np.percentile(latencies, 95)) if latencies else None,
        )
    provisional = sum(1 for e in events if e.status == EventStatus.provisional)
    retracted = sum(1 for e in events if e.status == EventStatus.retracted)
    return EvalReport(
        per_type=per_type,
        retraction_rate=retracted / provisional if provisional else 0.0,
        firsts_accuracy=firsts_accuracy(firsts, truth_firsts) if firsts is not None and truth_firsts else None,
    )
