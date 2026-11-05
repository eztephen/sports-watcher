"""Ground truth for a clip: one JSON object per line."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class TruthEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    t_pts: float
    type: str
    team: str
    player: str | None = None
    points: int | None = None
    zone: str | None = None


class CameraSegment(BaseModel):
    """A stretch of a clip in one camera state: ``[start, end)`` seconds of pts."""

    model_config = ConfigDict(frozen=True)

    start: float
    end: float
    state: str


def load_camera_truth(path: str | Path) -> list[CameraSegment]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [CameraSegment.model_validate_json(line) for line in lines if line.strip()]


def load_truth(path: str | Path) -> list[TruthEvent]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [TruthEvent.model_validate_json(line) for line in lines if line.strip()]


def save_truth(path: str | Path, events: list[TruthEvent]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(e.model_dump_json() + "\n" for e in events), encoding="utf-8")
