"""Player tracking in the OC-SORT style and a ball filter.

Players: one SORT Kalman filter per track, IoU plus an observation-centric momentum cost,
Hungarian assignment, and a velocity re-seed when a track is re-associated after a gap.
Ball: a constant-velocity filter in image space that predicts through dropouts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from sports_watcher.perception.types import BallObs, Box


@dataclass(slots=True)
class Detection:
    box: Box
    cls: str
    confidence: float


@dataclass(slots=True)
class TrackState:
    track_id: int
    box: Box
    cls: str
    confidence: float
    hits: int
    time_since_update_s: float


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


# -- SORT box <-> state ------------------------------------------------------------------


def _to_z(box: Box) -> np.ndarray:
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, w * h, w / h if h > 0 else 1.0], dtype=float)


def _to_box(x: np.ndarray) -> Box:
    cx, cy, s, r = float(x[0]), float(x[1]), max(float(x[2]), 1e-6), max(float(x[3]), 1e-6)
    w = math.sqrt(s * r)
    h = s / w
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _new_kf(box: Box) -> KalmanFilter:
    kf = KalmanFilter(dim_x=7, dim_z=4)
    kf.F = np.eye(7)
    kf.F[0, 4] = kf.F[1, 5] = kf.F[2, 6] = 1.0
    kf.H = np.eye(4, 7)
    kf.R[2:, 2:] *= 10.0
    kf.P[4:, 4:] *= 1000.0
    kf.P *= 10.0
    kf.Q[-1, -1] *= 0.01
    kf.Q[4:, 4:] *= 0.01
    kf.x[:4, 0] = _to_z(box)
    return kf


@dataclass
class _Track:
    track_id: int
    kf: KalmanFilter
    cls: str
    confidence: float
    last_obs: Box
    last_update_pts: float
    hits: int = 1
    published: bool = False
    history: list[tuple[float, Box]] = field(default_factory=list)  # (pts, observed box)

    def predicted_box(self) -> Box:
        return _to_box(self.kf.x[:, 0])

    def direction(self, delta: int = 3) -> tuple[float, float] | None:
        """Unit direction of the last observed motion, OC-SORT's momentum term."""
        if len(self.history) < 2:
            return None
        a = self.history[max(0, len(self.history) - 1 - delta)][1]
        b = self.history[-1][1]
        dx = (b[0] + b[2]) / 2 - (a[0] + a[2]) / 2
        dy = (b[1] + b[3]) / 2 - (a[1] + a[3]) / 2
        n = math.hypot(dx, dy)
        return (dx / n, dy / n) if n > 1e-6 else None


class Tracker:
    def __init__(
        self,
        max_age_s: float = 1.0,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        velocity_weight: float = 0.2,
    ) -> None:
        self.max_age_s = max_age_s
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.velocity_weight = velocity_weight
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._last_pts: float | None = None

    def reset(self) -> None:
        self._tracks, self._next_id, self._last_pts = [], 1, None

    def update(self, dets: list[Detection], t_pts: float) -> list[TrackState]:
        for tr in self._tracks:
            tr.kf.predict()
        matches, unmatched_dets = self._associate(dets)
        for ti, di in matches:
            self._update_track(self._tracks[ti], dets[di], t_pts)
        for di in unmatched_dets:
            d = dets[di]
            self._tracks.append(
                _Track(self._next_id, _new_kf(d.box), d.cls, d.confidence, d.box, t_pts, history=[(t_pts, d.box)])
            )
            self._next_id += 1
        self._tracks = [tr for tr in self._tracks if t_pts - tr.last_update_pts <= self.max_age_s]
        self._last_pts = t_pts
        out = []
        for tr in self._tracks:
            updated_now = tr.last_update_pts == t_pts
            if tr.hits >= self.min_hits:
                tr.published = True
            if tr.published and updated_now:
                out.append(
                    TrackState(tr.track_id, tr.last_obs, tr.cls, tr.confidence, tr.hits, 0.0)
                )
        return out

    # -- internals ------------------------------------------------------------------------

    def _associate(self, dets: list[Detection]) -> tuple[list[tuple[int, int]], list[int]]:
        if not self._tracks or not dets:
            return [], list(range(len(dets)))
        cost = np.zeros((len(self._tracks), len(dets)))
        gate = np.zeros_like(cost, dtype=bool)
        for ti, tr in enumerate(self._tracks):
            pred = tr.predicted_box()
            direction = tr.direction()
            last_c = ((tr.last_obs[0] + tr.last_obs[2]) / 2, (tr.last_obs[1] + tr.last_obs[3]) / 2)
            for di, d in enumerate(dets):
                overlap = iou(pred, d.box)
                c = 1.0 - overlap
                if direction is not None:
                    dc = ((d.box[0] + d.box[2]) / 2 - last_c[0], (d.box[1] + d.box[3]) / 2 - last_c[1])
                    n = math.hypot(*dc)
                    if n > 1e-6:
                        c -= self.velocity_weight * (dc[0] * direction[0] + dc[1] * direction[1]) / n
                cost[ti, di] = c
                gate[ti, di] = overlap >= self.iou_threshold and tr.cls == d.cls
        cost[~gate] = 1e6
        rows, cols = linear_sum_assignment(cost)
        matches = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if gate[r, c]]
        matched_dets = {c for _, c in matches}
        return matches, [i for i in range(len(dets)) if i not in matched_dets]

    def _update_track(self, tr: _Track, d: Detection, t_pts: float) -> None:
        gap = t_pts - tr.last_update_pts
        if self._last_pts is not None and gap > 1.5 * max(t_pts - self._last_pts, 1e-6):
            # Re-associated after missed frames: re-seed position and velocity along the
            # straight line from the last observation (OC-SORT's observation-centric
            # re-update, in one step).
            frames = max(1.0, round(gap / max(t_pts - self._last_pts, 1e-6)))
            z_old, z_new = _to_z(tr.last_obs), _to_z(d.box)
            tr.kf.x[:4, 0] = z_new
            tr.kf.x[4:7, 0] = (z_new[:3] - z_old[:3]) / frames
        else:
            tr.kf.update(_to_z(d.box))
        tr.last_obs = d.box
        tr.confidence = d.confidence
        tr.last_update_pts = t_pts
        tr.hits += 1
        tr.history.append((t_pts, d.box))
        del tr.history[:-10]


# -- ball -------------------------------------------------------------------------------


class BallFilter:
    """Constant-velocity Kalman filter on the ball centre, variable time step."""

    def __init__(self, drop_after_s: float = 0.5, gate_px: float = 60.0) -> None:
        self.drop_after_s = drop_after_s
        self.gate_px = gate_px
        self._kf: KalmanFilter | None = None
        self._last_t: float | None = None
        self._radius = 0.0
        self._confidence = 0.0

    def reset(self) -> None:
        self._kf, self._last_t = None, None

    @staticmethod
    def _f(dt: float) -> np.ndarray:
        f = np.eye(4)
        f[0, 2] = f[1, 3] = dt
        return f

    def predict(self, t_pts: float) -> tuple[float, float] | None:
        if self._kf is None or self._last_t is None:
            return None
        dt = t_pts - self._last_t
        x = self._f(dt) @ self._kf.x
        return (float(x[0, 0]), float(x[1, 0]))

    def update(self, xy: tuple[float, float], radius_px: float, confidence: float, t_pts: float) -> bool:
        if self._kf is None or self._last_t is None:
            kf = KalmanFilter(dim_x=4, dim_z=2)
            kf.H = np.eye(2, 4)
            kf.x = np.array([[xy[0]], [xy[1]], [0.0], [0.0]])
            kf.P = np.diag([1.0, 1.0, 1e4, 1e4])
            kf.R = np.eye(2) * 1.0
            kf.Q = np.diag([0.1, 0.1, 10.0, 10.0])
            self._kf = kf
        else:
            pred = self.predict(t_pts)
            assert pred is not None
            if math.dist(pred, xy) > max(self.gate_px, 3.0 * radius_px):
                return False
            dt = t_pts - self._last_t
            self._kf.F = self._f(dt)
            self._kf.predict()
            self._kf.update(np.array([[xy[0]], [xy[1]]]))
        self._last_t = t_pts
        self._radius = radius_px
        self._confidence = confidence
        return True

    def current(self, t_pts: float) -> BallObs | None:
        if self._kf is None or self._last_t is None or t_pts - self._last_t > self.drop_after_s:
            return None
        if t_pts == self._last_t:
            return BallObs((float(self._kf.x[0, 0]), float(self._kf.x[1, 0])), self._radius, None, "detected", self._confidence)
        pred = self.predict(t_pts)
        assert pred is not None
        return BallObs(pred, self._radius, None, "predicted", self._confidence)
