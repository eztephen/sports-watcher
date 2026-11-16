"""Ball detection at native resolution: a crop around the predicted ball position.

The crop pass costs a whole detector run (250 of 531 ms a frame on a laptop CPU), and it
used to run on every frame, also minutes after the ball was last seen, around a position
the filter kept extrapolating. It now runs only while the ball is being tracked.

Tried and dropped: taking a confident full-frame ball without the crop pass. On three
minutes of footage the two passes were 3.4 px apart at the median where both saw a ball,
but trusting the full frame cost a made basket's date on one clip and added false
possession starts on two (the crop's ball is the better one for the holder test)."""

from __future__ import annotations

import numpy as np

from sports_watcher.perception.tracker import Detection

DEFAULT_CROP_PX = 384
MAX_RADIUS_FRAC = 0.04  # a ball is 15 to 25 px in a 1080p frame; anything past 43 px is not one


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
    def __init__(
        self,
        detector,
        crop_px: int = DEFAULT_CROP_PX,
        ball_class: str = "ball",
        max_radius_frac: float = MAX_RADIUS_FRAC,
    ) -> None:
        self._detector = detector
        self.crop_px = crop_px
        self.ball_class = ball_class
        self.max_radius_frac = max_radius_frac

    def _plausible(self, d: Detection, image_h: int) -> bool:
        radius = max(d.box[2] - d.box[0], d.box[3] - d.box[1]) / 2.0
        return d.cls == self.ball_class and radius <= self.max_radius_frac * image_h

    def find(
        self,
        image_bgr: np.ndarray,
        predicted_xy: tuple[float, float] | None,
        full_dets: list[Detection],
        tracking: bool = True,
    ) -> Detection | None:
        """``tracking`` is False once the ball has not been seen for a while: the predicted
        position is then worthless and no crop is searched."""
        image_h = image_bgr.shape[0]
        if predicted_xy is not None and tracking:
            x0, y0, x1, y1 = crop_window(predicted_xy, self.crop_px, image_bgr.shape[:2])
            crop = np.ascontiguousarray(image_bgr[y0:y1, x0:x1])
            found = [d for d in self._detector.detect(crop) if self._plausible(d, image_h)]
            if found:
                best = max(found, key=lambda d: d.confidence)
                b = best.box
                return Detection((b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0), best.cls, best.confidence)
        balls = [d for d in full_dets if self._plausible(d, image_h)]
        return max(balls, key=lambda d: d.confidence) if balls else None
