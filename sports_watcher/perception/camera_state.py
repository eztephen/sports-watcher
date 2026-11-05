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


# -- the probe --------------------------------------------------------------------------

import json
from pathlib import Path

import numpy as np

KEYPOINTS_TOTAL = 33.0
PLAYERS_TOTAL = 12.0


def frame_features(inliers: int, n_players: int, median_box_h_frac: float) -> np.ndarray:
    """Three cheap cues next to the embedding: homography inliers, player count and the
    median player height as a fraction of the frame, each scaled to about 0..1."""
    return np.array(
        [inliers / KEYPOINTS_TOTAL, n_players / PLAYERS_TOTAL, median_box_h_frac], dtype=np.float32
    )


class CameraProbe:
    """A linear classifier over ``concat(embedding, features)``; softmax gives the hint."""

    def __init__(self, coef: np.ndarray, intercept: np.ndarray, classes: list[str]) -> None:
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = np.asarray(intercept, dtype=np.float64)
        self.classes = list(classes)

    @property
    def embedding_dim(self) -> int:
        return self.coef.shape[1] - 3

    def predict(self, embedding: np.ndarray, features: np.ndarray) -> tuple[CameraHint, float]:
        x = np.concatenate([np.asarray(embedding, dtype=np.float64).ravel(), np.asarray(features, dtype=np.float64).ravel()])
        logits = self.coef @ x + self.intercept
        logits = logits - logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        best = int(probs.argmax())
        return self.classes[best], float(probs[best])  # type: ignore[return-value]

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(
                {
                    "coef": self.coef.tolist(),
                    "intercept": self.intercept.tolist(),
                    "classes": self.classes,
                    "embedding_dim": self.embedding_dim,
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> CameraProbe:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.array(d["coef"]), np.array(d["intercept"]), d["classes"])
