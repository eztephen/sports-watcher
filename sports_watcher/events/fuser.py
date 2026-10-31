"""Pending provisional events and how they resolve: by their confirmer or by TTL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sports_watcher.events.model import Event
from sports_watcher.sports.base import EventDraft, Observation, Verdict
from sports_watcher.state.game_state import GameState

Resolution = Literal["confirmed", "retracted", "ttl"]


@dataclass
class PendingEvent:
    event: Event
    draft: EventDraft
    deadline_pts: float


def resolve_pending(
    pending: list[PendingEvent], state: GameState, obs: Observation, now_pts: float
) -> tuple[list[tuple[PendingEvent, Resolution]], list[PendingEvent]]:
    """Ask each pending event's confirmer for a verdict, then apply the TTL.

    Returns ``(resolved, still_pending)``; resolution is ``confirmed``, ``retracted`` or
    ``ttl`` (expired while still pending).
    """
    resolved: list[tuple[PendingEvent, Resolution]] = []
    remaining: list[PendingEvent] = []
    for p in pending:
        verdict: Verdict = p.draft.confirmer(p.draft, state, obs)
        if verdict == "pending" and now_pts >= p.deadline_pts:
            resolved.append((p, "ttl"))
        elif verdict == "pending":
            remaining.append(p)
        else:
            resolved.append((p, verdict))
    return resolved, remaining
