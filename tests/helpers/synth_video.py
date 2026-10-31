"""Write a short video with a drawn scoreboard so end-to-end tests need no footage."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sports_watcher.config import Roi
from tests.helpers.scoreboard_images import render_scoreboard

FRAME_SIZE = (1280, 720)
BAR_SIZE = (640, 80)
BAR_ORIGIN = (0, 640)  # x, y of the scoreboard bar inside the frame


def write_scoreboard_video(
    path: Path,
    timeline: list[tuple[float, int, int, str, str]],
    duration_s: float,
    fps: int = 25,
) -> Roi:
    """``timeline`` entries are ``(t_s, home, away, clock, period)`` taking effect from
    ``t_s``. Returns the ROI (fractions of the frame) that contains the scoreboard."""
    w, h = FRAME_SIZE
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open " + str(path))
    entries = sorted(timeline)
    n = round(duration_s * fps)
    for i in range(n):
        t = i / fps
        current = entries[0]
        for entry in entries:
            if entry[0] <= t:
                current = entry
        _, home, away, clock, period = current
        frame = np.full((h, w, 3), (40, 90, 40), np.uint8)  # court green
        bar = render_scoreboard(home, away, clock, period, size=BAR_SIZE)
        x0, y0 = BAR_ORIGIN
        frame[y0 : y0 + BAR_SIZE[1], x0 : x0 + BAR_SIZE[0]] = bar
        writer.write(frame)
    writer.release()
    return Roi(x=BAR_ORIGIN[0] / w, y=BAR_ORIGIN[1] / h, w=BAR_SIZE[0] / w, h=BAR_SIZE[1] / h)
