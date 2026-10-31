"""Read the broadcast scoreboard bug: scores, game clock, period and shot clock.

The reader crops a configured region, runs a text recogniser, assigns roles to the text
boxes by geometry, and majority-votes the stable fields over a short window so a single
misread never changes the game state.
"""

from __future__ import annotations

import re
from collections import Counter, deque
from typing import Protocol

import numpy as np
from pydantic import BaseModel

from sports_watcher.config import BroadcasterConfig, Roi

TextBox = tuple[str, tuple[int, int, int, int], float]  # text, (x, y, w, h), confidence

_INT = re.compile(r"^\d{1,3}$")
_CLOCK_MMSS = re.compile(r"^(\d{1,2})[:.](\d{2})(?:\.(\d))?$")
_CLOCK_SECONDS = re.compile(r"^:(\d{1,2})(?:\.(\d))?$")
_CLOCK_TENTHS = re.compile(r"^(\d{1,2})\.(\d)$")
_PERIOD_Q = re.compile(r"^Q(\d)$")
_PERIOD_ORD = re.compile(r"^(\d)(ST|ND|RD|TH)?$")
_PERIOD_OT = re.compile(r"^(\d?)OT$")
SHOT_CLOCK_MAX = 24


class ScoreboardState(BaseModel):
    home: int | None = None
    away: int | None = None
    clock_s: float | None = None
    period: int | None = None
    shot_clock_s: float | None = None
    confidence: float = 0.0
    raw: list[str] = []


class TextReader(Protocol):
    def read(self, image_bgr: np.ndarray) -> list[TextBox]: ...


class RapidOcrTextReader:
    """RapidOCR (Apache-2.0, ONNX models bundled) as a TextReader."""

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self._ocr = RapidOCR()

    def read(self, image_bgr: np.ndarray) -> list[TextBox]:
        result, _elapsed = self._ocr(image_bgr)
        out: list[TextBox] = []
        for box, text, conf in result or []:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x, y = int(min(xs)), int(min(ys))
            out.append((str(text).strip(), (x, y, int(max(xs) - x), int(max(ys) - y)), float(conf)))
        return out


def crop_roi(image: np.ndarray, roi: Roi) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = max(0, round(roi.x * w))
    y0 = max(0, round(roi.y * h))
    x1 = min(w, round((roi.x + roi.w) * w))
    y1 = min(h, round((roi.y + roi.h) * h))
    return image[y0:y1, x0:x1]


def parse_clock(text: str) -> float | None:
    t = text.strip().replace(" ", "")
    if m := _CLOCK_MMSS.match(t):
        minutes, seconds, tenths = m.groups()
        return int(minutes) * 60 + int(seconds) + (int(tenths) / 10 if tenths else 0.0)
    if m := _CLOCK_SECONDS.match(t):
        seconds, tenths = m.groups()
        return int(seconds) + (int(tenths) / 10 if tenths else 0.0)
    if m := _CLOCK_TENTHS.match(t):
        seconds, tenths = m.groups()
        return int(seconds) + int(tenths) / 10
    return None


def parse_period(text: str) -> int | None:
    t = text.strip().upper().replace(" ", "")
    if m := _PERIOD_Q.match(t):
        return int(m.group(1))
    if m := _PERIOD_OT.match(t):
        return 4 + (int(m.group(1)) if m.group(1) else 1)
    if m := _PERIOD_ORD.match(t):
        return int(m.group(1))
    return None


def parse_scoreboard(boxes: list[TextBox], home_side: str, shot_clock_visible: bool) -> ScoreboardState:
    raw = [b[0] for b in boxes]
    ints = [b for b in boxes if _INT.match(b[0].strip())]
    others = [b for b in boxes if not _INT.match(b[0].strip())]

    home = away = None
    used: list[TextBox] = []
    if len(ints) >= 2:
        tallest = sorted(ints, key=lambda b: b[1][3], reverse=True)[:2]
        left, right = sorted(tallest, key=lambda b: b[1][0])
        home_box, away_box = (left, right) if home_side == "left" else (right, left)
        home, away = int(home_box[0]), int(away_box[0])
        used = [left, right]

    remaining_ints = [b for b in ints if b not in used]
    shot_clock = None
    if shot_clock_visible:
        for b in remaining_ints:
            if int(b[0]) <= SHOT_CLOCK_MAX:
                shot_clock = float(int(b[0]))
                remaining_ints.remove(b)
                break

    clock = next((v for b in others if (v := parse_clock(b[0])) is not None), None)
    period = next((v for b in others if (v := parse_period(b[0])) is not None), None)
    if period is None:
        period = next((int(b[0]) for b in remaining_ints if 1 <= int(b[0]) <= 9), None)

    confidence = sum(b[2] for b in used) / len(used) if used else 0.0
    return ScoreboardState(
        home=home, away=away, clock_s=clock, period=period, shot_clock_s=shot_clock,
        confidence=confidence, raw=raw,
    )


def _mode(values: list) -> tuple[object, int]:
    """Most common non-None value; ties go to the most recent. Returns (value, count)."""
    present = [v for v in values if v is not None]
    if not present:
        return None, 0
    counts = Counter(present)
    best = max(counts.values())
    for v in reversed(present):
        if counts[v] == best:
            return v, best
    return None, 0  # pragma: no cover


class ScoreboardReader:
    VOTED_FIELDS = ("home", "away", "period")
    LATEST_FIELDS = ("clock_s", "shot_clock_s")

    def __init__(self, reader: TextReader, broadcaster: BroadcasterConfig, vote_window: int = 5) -> None:
        self._reader = reader
        self._broadcaster = broadcaster
        self._window: deque[ScoreboardState] = deque(maxlen=vote_window)

    def read(self, frame_bgr: np.ndarray) -> ScoreboardState:
        crop = crop_roi(frame_bgr, self._broadcaster.scoreboard_roi)
        boxes = self._reader.read(crop) if crop.size else []
        parsed = parse_scoreboard(
            boxes, self._broadcaster.home_side, self._broadcaster.shot_clock_visible
        )
        self._window.append(parsed)
        return self._vote(parsed)

    def _vote(self, latest: ScoreboardState) -> ScoreboardState:
        window = list(self._window)
        voted = {f: _mode([getattr(s, f) for s in window])[0] for f in self.VOTED_FIELDS}
        winning = tuple(voted[f] for f in self.VOTED_FIELDS)
        agree = sum(1 for s in window if tuple(getattr(s, f) for f in self.VOTED_FIELDS) == winning)
        latest_vals = {
            f: next((getattr(s, f) for s in reversed(window) if getattr(s, f) is not None), None)
            for f in self.LATEST_FIELDS
        }
        return ScoreboardState(
            **voted, **latest_vals, confidence=agree / len(window), raw=latest.raw
        )
