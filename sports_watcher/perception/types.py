"""What the perceiver hands the sport plug-in for one frame."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
Team = Literal["home", "away", "unknown"]
TrackClass = Literal["player", "referee"]
CameraHint = Literal["main", "replay", "closeup", "graphics"]
BallSource = Literal["detected", "predicted"]


@dataclass(slots=True)
class Track:
    track_id: int
    box: Box
    cls: TrackClass
    team: Team
    feet_xy: tuple[float, float] | None  # court metres, None without a homography
    confidence: float


@dataclass(slots=True)
class BallObs:
    xy: tuple[float, float]  # image pixels, centre
    radius_px: float
    floor_xy: tuple[float, float] | None  # the centre projected onto the floor plane
    source: BallSource
    confidence: float


@dataclass(slots=True)
class Homography:
    matrix: np.ndarray  # 3x3, image pixels -> court metres
    inliers: int
    reprojection_error_px: float
    confidence: float  # 0..1 from inliers and error


@dataclass(slots=True)
class VisionFrame:
    frame_index: int
    t_pts: float
    camera_hint: CameraHint
    camera_confidence: float
    tracks: list[Track] = field(default_factory=list)
    ball: BallObs | None = None
    rim: Box | None = None
    homography: Homography | None = None


def empty_vision(frame_index: int, t_pts: float) -> VisionFrame:
    return VisionFrame(frame_index=frame_index, t_pts=t_pts, camera_hint="main", camera_confidence=0.0)


def box_center(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def box_height(box: Box) -> float:
    return box[3] - box[1]


def feet_of(box: Box) -> tuple[float, float]:
    """Bottom centre of a box: where a standing player touches the floor."""
    return ((box[0] + box[2]) / 2.0, box[3])


# -- JSON shape for the perception cache ------------------------------------------------


def _tuple(v: Any) -> Any:
    return tuple(v) if v is not None else None


def vision_to_dict(vf: VisionFrame) -> dict:
    d: dict[str, Any] = {
        "frame_index": vf.frame_index,
        "t_pts": vf.t_pts,
        "camera_hint": vf.camera_hint,
        "camera_confidence": vf.camera_confidence,
        "tracks": [asdict(t) for t in vf.tracks],
        "ball": asdict(vf.ball) if vf.ball is not None else None,
        "rim": list(vf.rim) if vf.rim is not None else None,
        "homography": None,
    }
    if vf.homography is not None:
        h = vf.homography
        d["homography"] = {
            "matrix": [float(x) for x in np.asarray(h.matrix, dtype=float).reshape(9)],
            "inliers": h.inliers,
            "reprojection_error_px": h.reprojection_error_px,
            "confidence": h.confidence,
        }
    return d


def vision_from_dict(d: dict) -> VisionFrame:
    tracks = [
        Track(
            track_id=t["track_id"],
            box=_tuple(t["box"]),
            cls=t["cls"],
            team=t["team"],
            feet_xy=_tuple(t["feet_xy"]),
            confidence=t["confidence"],
        )
        for t in d.get("tracks", [])
    ]
    ball = None
    if d.get("ball") is not None:
        b = d["ball"]
        ball = BallObs(
            xy=_tuple(b["xy"]),
            radius_px=b["radius_px"],
            floor_xy=_tuple(b["floor_xy"]),
            source=b["source"],
            confidence=b["confidence"],
        )
    homography = None
    if d.get("homography") is not None:
        h = d["homography"]
        homography = Homography(
            matrix=np.asarray(h["matrix"], dtype=float).reshape(3, 3),
            inliers=h["inliers"],
            reprojection_error_px=h["reprojection_error_px"],
            confidence=h["confidence"],
        )
    return VisionFrame(
        frame_index=d["frame_index"],
        t_pts=d["t_pts"],
        camera_hint=d["camera_hint"],
        camera_confidence=d["camera_confidence"],
        tracks=tracks,
        ball=ball,
        rim=_tuple(d.get("rim")),
        homography=homography,
    )
