"""Ball detection at native resolution: a crop around the predicted ball position."""

from __future__ import annotations

import numpy as np

from sports_watcher.perception.tracker import Detection

DEFAULT_CROP_PX = 384


def crop_window(center_xy: tuple[float, float], crop_px: int, image_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    """``(x0, y0, x1, y1)`` of a ``crop_px`` square centred on ``center_xy``, shifted to stay
    inside the image (smaller only when the image itself is smaller)."""
    h, w = image_hw
    size_x, size_y = min(crop_px, w), min(crop_px, h)
    x0 = round(center_xy[0] - size_x / 2)
    y0 = round(center_xy[1] - size_y / 2)
    x0 = min(max(x0, 0), w - size_x)
    y0 = min(max(y0, 0), h - size_y)
    return (x0, y0, x0 + size_x, y0 + size_y)


class BallFinder:
    def __init__(self, detector, crop_px: int = DEFAULT_CROP_PX, ball_class: str = "ball") -> None:
        self._detector = detector
        self.crop_px = crop_px
        self.ball_class = ball_class

    def find(
        self,
        image_bgr: np.ndarray,
        predicted_xy: tuple[float, float] | None,
        full_dets: list[Detection],
    ) -> Detection | None:
        if predicted_xy is not None:
            x0, y0, x1, y1 = crop_window(predicted_xy, self.crop_px, image_bgr.shape[:2])
            crop = np.ascontiguousarray(image_bgr[y0:y1, x0:x1])
            balls = [d for d in self._detector.detect(crop) if d.cls == self.ball_class]
            if balls:
                best = max(balls, key=lambda d: d.confidence)
                b = best.box
                return Detection((b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0), best.cls, best.confidence)
        balls = [d for d in full_dets if d.cls == self.ball_class]
        return max(balls, key=lambda d: d.confidence) if balls else None
