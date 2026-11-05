"""The real perceiver: ONNX detector, ball crop, tracker, court mapper, team assigner and
camera probe composed into one VisionFrame per frame. Optional models degrade: no court
model means no homography, no SigLIP means no teams and no probe, no probe means hint
``main`` at confidence 0.5."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from sports_watcher.config import LeagueConfig, Settings
from sports_watcher.ingest.base import Frame
from sports_watcher.perception.ball import BallFinder
from sports_watcher.perception.camera_state import CameraProbe, frame_features
from sports_watcher.perception.court import (
    CourtMapper,
    image_to_court,
    load_template,
    order_template,
)
from sports_watcher.perception.detector import OnnxDetector, OnnxKeypointModel
from sports_watcher.perception.models import ModelSet
from sports_watcher.perception.team import SiglipEmbedder, TeamAssigner, parse_color
from sports_watcher.perception.tracker import BallFilter, Detection, Tracker
from sports_watcher.perception.types import (
    BallObs,
    Box,
    Homography,
    Track,
    VisionFrame,
    box_center,
    box_height,
    feet_of,
)

RIM_HOLD_S = 2.0
RIM_MEDIAN_S = 1.0
NO_PROBE_HINT = ("main", 0.5)


class OnnxPerceiver:
    def __init__(
        self,
        models: ModelSet,
        league: LeagueConfig,
        home_color: str | None,
        away_color: str | None,
        template_path: Path | None = None,
        keypoint_every_s: float = 0.2,
        probe_every_s: float = 0.2,
        providers: list[str] | None = None,
    ) -> None:
        self.version = models.version
        self.keypoint_every_s = keypoint_every_s
        self.probe_every_s = probe_every_s
        self.timings: dict[str, float] = {}
        self._counts: dict[str, int] = {}

        self._detector = OnnxDetector(models.detector, models.detector_manifest, providers)
        self._ball_finder = BallFinder(self._detector)
        self._tracker = Tracker()
        self._ball = BallFilter()

        self._keypoints: OnnxKeypointModel | None = None
        self._court: CourtMapper | None = None
        if models.court is not None and models.court_manifest is not None:
            manifest = models.court_manifest
            if template_path is None:
                dataset = Path(manifest.dataset or "court").name
                template_path = Path(Settings().configs_dir) / "court_keypoints" / f"{dataset}.yaml"
            names, xy = load_template(template_path, league)
            self._court = CourtMapper(order_template(names, xy, manifest.keypoints))
            self._keypoints = OnnxKeypointModel(models.court, manifest, providers)

        self._embed: SiglipEmbedder | None = SiglipEmbedder(models.siglip, providers) if models.siglip else None
        self._team: TeamAssigner | None = None
        if self._embed is not None:
            self._team = TeamAssigner(
                self._embed,
                parse_color(home_color) if home_color else None,
                parse_color(away_color) if away_color else None,
            )
        self._probe: CameraProbe | None = None
        if models.camera_probe is not None and self._embed is not None:
            self._probe = CameraProbe.load(models.camera_probe)
        self.reset()

    def reset(self) -> None:
        self._tracker.reset()
        self._ball.reset()
        if self._court is not None:
            self._court.reset()
        if self._team is not None:
            self._team.reset()
        self._last_kp_pts: float | None = None
        self._last_probe_pts: float | None = None
        self._hint: tuple[str, float] = NO_PROBE_HINT
        self._rims: list[tuple[float, Box]] = []
        self._rim: Box | None = None
        self._rim_seen_pts: float | None = None

    # -- per frame ----------------------------------------------------------------------

    def perceive(self, frame: Frame) -> VisionFrame:
        t, img = frame.pts_s, frame.image
        dets = self._timed("detector", self._detector.detect, img)
        people = [d for d in dets if d.cls in ("player", "referee")]
        raw_tracks = self._timed("tracker", self._tracker.update, people, t)
        rim = self._update_rim([d for d in dets if d.cls == "rim"], t)
        ball = self._update_ball(img, dets, t)
        homography = self._update_court(img, t)
        tracks = self._build_tracks(img, raw_tracks, homography, t)
        if ball is not None and homography is not None:
            ball.floor_xy = image_to_court(homography.matrix, ball.xy)
        hint, conf = self._update_hint(img, tracks, homography, t)
        return VisionFrame(frame.index, t, hint, conf, tracks, ball, rim, homography)  # type: ignore[arg-type]

    # -- stages -------------------------------------------------------------------------

    def _timed(self, name: str, fn, *args):
        t0 = time.perf_counter()
        out = fn(*args)
        dt = time.perf_counter() - t0
        n = self._counts.get(name, 0)
        self.timings[name] = (self.timings.get(name, 0.0) * n + dt) / (n + 1)
        self._counts[name] = n + 1
        return out

    def _update_rim(self, rims: list[Detection], t: float) -> Box | None:
        if rims:
            best = max(rims, key=lambda d: d.confidence)
            self._rims.append((t, best.box))
            self._rim_seen_pts = t
        self._rims = [(ts, b) for ts, b in self._rims if t - ts <= RIM_MEDIAN_S]
        if self._rim_seen_pts is None or t - self._rim_seen_pts > RIM_HOLD_S:
            self._rim = None
        elif self._rims:
            arr = np.array([b for _, b in self._rims])
            self._rim = tuple(float(v) for v in np.median(arr, axis=0))  # type: ignore[assignment]
        return self._rim

    def _update_ball(self, img: np.ndarray, dets: list[Detection], t: float) -> BallObs | None:
        pred = self._ball.predict(t)
        det = self._timed("ball", self._ball_finder.find, img, pred, dets)
        if det is not None:
            cx, cy = box_center(det.box)
            radius = max(det.box[2] - det.box[0], det.box[3] - det.box[1]) / 2.0
            self._ball.update((cx, cy), radius, det.confidence, t)
        return self._ball.current(t)

    def _update_court(self, img: np.ndarray, t: float) -> Homography | None:
        if self._court is None or self._keypoints is None:
            return None
        if self._last_kp_pts is None or t - self._last_kp_pts >= self.keypoint_every_s:
            kp_xy, kp_conf = self._timed("keypoints", self._keypoints.keypoints, img)
            self._court.update(kp_xy, kp_conf, t)
            self._last_kp_pts = t
        return self._court.current(t)

    def _build_tracks(self, img: np.ndarray, raw, homography: Homography | None, t: float) -> list[Track]:
        tracks: list[Track] = []
        h, w = img.shape[:2]
        for ts in raw:
            feet = image_to_court(homography.matrix, feet_of(ts.box)) if homography is not None else None
            team = "unknown"
            if ts.cls == "player" and self._team is not None:
                x1, y1, x2, y2 = (int(max(0, ts.box[0])), int(max(0, ts.box[1])), int(min(w, ts.box[2])), int(min(h, ts.box[3])))
                if x2 > x1 + 1 and y2 > y1 + 1:
                    crop = img[y1:y2, x1:x2]
                    self._timed("team", self._team.observe, ts.track_id, crop, t)
                team = self._team.team_of(ts.track_id)
            tracks.append(Track(ts.track_id, ts.box, ts.cls, team, feet, ts.confidence))  # type: ignore[arg-type]
        return tracks

    def _update_hint(self, img: np.ndarray, tracks: list[Track], homography: Homography | None, t: float) -> tuple[str, float]:
        if self._probe is None or self._embed is None:
            return NO_PROBE_HINT
        if self._last_probe_pts is None or t - self._last_probe_pts >= self.probe_every_s:
            players = [tr for tr in tracks if tr.cls == "player"]
            median_h = float(np.median([box_height(tr.box) for tr in players]) / img.shape[0]) if players else 0.0
            feats = frame_features(homography.inliers if homography is not None else 0, len(players), median_h)
            emb = self._timed("probe", self._embed, img)
            self._hint = self._probe.predict(emb, feats)
            self._last_probe_pts = t
        return self._hint
