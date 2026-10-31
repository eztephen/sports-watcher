"""Synthetic scoreboard text boxes and rendered images for scoreboard tests."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sports_watcher.config import BroadcasterConfig, Roi

# Box layout used by the parse tests: scores are the tallest boxes, left and right.
B1 = (10, 0, 40, 30)
B2 = (200, 0, 40, 30)
B_CLOCK = (100, 0, 60, 20)
B_PERIOD = (100, 30, 30, 15)
B_SHOT = (300, 0, 20, 15)

BLANK = np.zeros((80, 640, 3), np.uint8)
BROADCASTER = BroadcasterConfig(
    scoreboard_roi=Roi(x=0.0, y=0.0, w=1.0, h=1.0), home_side="left", shot_clock_visible=True
)


def boxes(home: int, away: int, clock: str, period: str, shot_clock: int | None = None):
    out = [
        (str(home), B1, 0.9),
        (str(away), B2, 0.9),
        (clock, B_CLOCK, 0.9),
        (period, B_PERIOD, 0.9),
    ]
    if shot_clock is not None:
        out.append((str(shot_clock), B_SHOT, 0.9))
    return out


def render_scoreboard(
    home: int, away: int, clock: str, period: str, size: tuple[int, int] = (640, 80)
) -> np.ndarray:
    """White text on a dark bar, in the same left-to-right order as the box layout."""
    w, h = size
    img = Image.new("RGB", (w, h), (20, 20, 30))
    draw = ImageDraw.Draw(img)
    try:
        big = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 40)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 26)
    except OSError:
        big = ImageFont.load_default(size=40)
        small = ImageFont.load_default(size=26)
    draw.text((20, 15), str(home), font=big, fill=(255, 255, 255))
    draw.text((w - 110, 15), str(away), font=big, fill=(255, 255, 255))
    draw.text((w // 2 - 50, 8), clock, font=small, fill=(255, 255, 255))
    draw.text((w // 2 - 20, 44), period, font=small, fill=(255, 255, 255))
    rgb = np.asarray(img)
    return rgb[:, :, ::-1].copy()  # BGR for OpenCV conventions
