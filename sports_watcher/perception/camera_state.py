"""Camera state: the per-frame hint from perception, held steady over time.

``CameraStateTracker`` turns the perceiver's hint (main, replay, closeup, graphics) into
the state the plug-in sees, switching only when a new hint has persisted for ``hold_s``.
The probe that produces the hint lives further down this module (Task 16).
"""

from __future__ import annotations

from sports_watcher.perception.types import CameraHint

DEFAULT_HOLD_S = 0.5
DEFAULT_MIN_CONFIDENCE = 0.3


class CameraStateTracker:
    def __init__(self, hold_s: float = DEFAULT_HOLD_S, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> None:
        self.hold_s = hold_s
        self.min_confidence = min_confidence
        self._state: str = "main"
        self._seen_any = False
        self._pending: tuple[str, float] | None = None  # hint, since pts

    @property
    def state(self) -> str:
        return self._state

    def reset(self) -> None:
        self._state, self._seen_any, self._pending = "main", False, None

    def update(self, hint: CameraHint, confidence: float, t_pts: float) -> str:
        if confidence < self.min_confidence:
            return self._state
        if not self._seen_any:
            self._seen_any = True
            self._state = hint
            return self._state
        if hint == self._state:
            self._pending = None
            return self._state
        if self._pending is None or self._pending[0] != hint:
            self._pending = (hint, t_pts)
        elif t_pts - self._pending[1] >= self.hold_s:
            self._state = hint
            self._pending = None
        return self._state
