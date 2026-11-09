"""Court keypoints to a homography (image pixels to court metres), smoothed over time."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from sports_watcher.config import LeagueConfig
from sports_watcher.perception.types import Homography

XY = tuple[float, float]


def load_template(path: Path, league: LeagueConfig) -> tuple[list[str], np.ndarray]:
    """Keypoint names and their court positions in metres from a template yaml.

    An entry names a court ``feature`` resolved from the league's geometry (the yaml's
    ``sport`` picks the resolver; ``basketball`` is the only one), or gives ``x_frac``
    and ``y_frac`` of the court length and width for an arbitrary point."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    resolver = None
    if any("feature" in k for k in data["keypoints"]):
        sport = data.get("sport", "basketball")
        if sport != "basketball":
            raise ValueError(f"no court feature resolver for sport {sport!r}")
        from sports_watcher.sports.basketball.geometry import resolve_feature

        resolver = resolve_feature
    names, points = [], []
    for k in data["keypoints"]:
        names.append(str(k["name"]))
        if "feature" in k:
            assert resolver is not None
            points.append(resolver(k, league))
        else:
            points.append((float(k["x_frac"]) * league.court_length_m, float(k["y_frac"]) * league.court_width_m))
    return names, np.array(points, dtype=float)


def order_template(names: list[str], xy: np.ndarray, manifest_keypoints: list[str]) -> np.ndarray:
    """Rows of ``xy`` in the model's keypoint order; a missing name raises ``KeyError``."""
    index = {n: i for i, n in enumerate(names)}
    try:
        rows = [index[n] for n in manifest_keypoints]
    except KeyError as exc:
        raise KeyError(f"court template lacks keypoint {exc}") from exc
    return xy[rows]


def image_to_court(h: np.ndarray, xy: XY) -> XY:
    v = h @ np.array([xy[0], xy[1], 1.0])
    return (float(v[0] / v[2]), float(v[1] / v[2]))


def court_to_image(h: np.ndarray, xy: XY) -> XY:
    return image_to_court(np.linalg.inv(h), xy)


class CourtMapper:
    def __init__(
        self,
        template_xy: np.ndarray,
        min_point_conf: float = 0.5,
        min_inliers: int = 6,
        ransac_px: float = 8.0,
        smooth_s: float = 1.0,
        min_confidence: float = 0.4,
        hold_s: float = 2.0,
    ) -> None:
        self.template = np.asarray(template_xy, dtype=np.float32)
        self.min_point_conf = min_point_conf
        self.min_inliers = min_inliers
        self.ransac_px = ransac_px
        self.smooth_s = smooth_s
        self.min_confidence = min_confidence
        self.hold_s = hold_s
        self._recent: list[tuple[float, np.ndarray, float]] = []  # pts, matrix, confidence
        self._last: tuple[float, Homography] | None = None

    def reset(self) -> None:
        self._recent, self._last = [], None

    def update(self, points_xy: np.ndarray, conf: np.ndarray, t_pts: float) -> Homography | None:
        keep = np.asarray(conf) >= self.min_point_conf
        if int(keep.sum()) < max(4, self.min_inliers):
            return None
        image_pts = np.asarray(points_xy, dtype=np.float32)[keep]
        court_pts = self.template[keep]
        # Fit court -> image so RANSAC's threshold and the error are in pixels, then invert.
        c2i, mask = cv2.findHomography(court_pts, image_pts, cv2.RANSAC, self.ransac_px)
        if c2i is None or mask is None or abs(np.linalg.det(c2i)) < 1e-12:
            return None
        inlier = mask.ravel().astype(bool)
        inliers = int(inlier.sum())
        if inliers < self.min_inliers:
            return None
        proj = cv2.perspectiveTransform(court_pts[inlier].reshape(-1, 1, 2), c2i).reshape(-1, 2)
        err = float(np.mean(np.linalg.norm(proj - image_pts[inlier], axis=1)))
        confidence = min(1.0, inliers / 12.0) * max(0.0, 1.0 - err / self.ransac_px)
        if confidence < self.min_confidence:
            return None
        h = np.linalg.inv(c2i)
        h = h / h[2, 2]
        self._recent = [(t, m, c) for t, m, c in self._recent if t_pts - t <= self.smooth_s]
        self._recent.append((t_pts, h, confidence))
        weights = np.array([c for _, _, c in self._recent])
        smoothed = sum(m * w for (_, m, _), w in zip(self._recent, weights, strict=True)) / weights.sum()
        smoothed = smoothed / smoothed[2, 2]
        result = Homography(smoothed, inliers, err, confidence)
        self._last = (t_pts, result)
        return result

    def current(self, t_pts: float) -> Homography | None:
        if self._last is None or t_pts - self._last[0] > self.hold_s:
            return None
        return self._last[1]
