"""Scripted TextReader doubles."""

from __future__ import annotations

import numpy as np


class SeqTextReader:
    """Returns the scripted box lists in order; repeats the last one afterwards."""

    def __init__(self, seq: list[list]) -> None:
        self._seq = list(seq)
        self.calls = 0

    def read(self, image_bgr: np.ndarray) -> list:
        idx = min(self.calls, len(self._seq) - 1)
        self.calls += 1
        return list(self._seq[idx]) if self._seq else []


class ScriptedTextReader:
    """Returns the entry whose key is the largest not greater than the call count so far."""

    def __init__(self, by_frame: dict[int, list]) -> None:
        self._by_frame = dict(sorted(by_frame.items()))
        self.calls = 0

    def read(self, image_bgr: np.ndarray) -> list:
        current = self.calls
        self.calls += 1
        chosen: list = []
        for key, value in self._by_frame.items():
            if key <= current:
                chosen = value
            else:
                break
        return list(chosen)
