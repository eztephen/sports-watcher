"""Team assignment: a per-crop feature clustered into two teams, mapped to home and away by
the operator's jersey colours. Referees never enter: the detector names them.

The perceiver clusters the torso colour (``jersey_features``). SigLIP embeddings of whole
crops were tried first and split the players of a PBA broadcast at random with respect to
the kit (51 % agreement with the jersey colour on 1,098 crops), while the torso colour
separated the two kits cleanly."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans

from sports_watcher.perception.types import Team

Embedder = Callable[[np.ndarray], np.ndarray]

NAMED_COLORS: dict[str, tuple[int, int, int]] = {  # RGB
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 128, 0),
    "yellow": (255, 215, 0),
    "orange": (255, 140, 0),
    "purple": (128, 0, 128),
    "navy": (0, 0, 128),
    "maroon": (128, 0, 0),
    "gold": (212, 175, 55),
    "teal": (0, 128, 128),
    "grey": (128, 128, 128),
}
MAX_SAMPLES = 2000
SIGLIP_SIZE = 224


def _rgb_to_lab(rgb: tuple[int, int, int]) -> np.ndarray:
    pixel = np.array([[rgb[::-1]]], dtype=np.uint8)  # BGR for OpenCV
    return cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0].astype(float)


def parse_color(text: str) -> np.ndarray:
    """Lab colour from ``#rrggbb`` or a colour name."""
    t = text.strip().lower()
    if t.startswith("#") and len(t) == 7:
        return _rgb_to_lab((int(t[1:3], 16), int(t[3:5], 16), int(t[5:7], 16)))
    if t in NAMED_COLORS:
        return _rgb_to_lab(NAMED_COLORS[t])
    raise ValueError(f"unknown colour {text!r}: use #rrggbb or one of {sorted(NAMED_COLORS)}")


def crop_color_lab(crop_bgr: np.ndarray) -> np.ndarray:
    """Median Lab colour of the torso window (rows 25 to 55 %, columns 25 to 75 %)."""
    h, w = crop_bgr.shape[:2]
    window = crop_bgr[int(h * 0.25) : max(int(h * 0.55), int(h * 0.25) + 1), int(w * 0.25) : max(int(w * 0.75), int(w * 0.25) + 1)]
    lab = cv2.cvtColor(np.ascontiguousarray(window), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return np.median(lab, axis=0).astype(float)


def jersey_features(crop_bgr: np.ndarray) -> np.ndarray:
    """The clustering feature for team assignment: the torso's median Lab colour."""
    return crop_color_lab(crop_bgr)


def align_labels(prev_centroids: np.ndarray, new_centroids: np.ndarray) -> dict[int, int]:
    """New cluster label to previous label, by the closer total centroid distance."""
    straight = np.linalg.norm(new_centroids[0] - prev_centroids[0]) + np.linalg.norm(new_centroids[1] - prev_centroids[1])
    swapped = np.linalg.norm(new_centroids[0] - prev_centroids[1]) + np.linalg.norm(new_centroids[1] - prev_centroids[0])
    return {0: 0, 1: 1} if straight <= swapped else {0: 1, 1: 0}


def map_clusters(cluster_lab: np.ndarray, home_lab: np.ndarray, away_lab: np.ndarray) -> dict[int, Team]:
    """Assign the two clusters to home and away with the smaller total Lab distance."""
    straight = np.linalg.norm(cluster_lab[0] - home_lab) + np.linalg.norm(cluster_lab[1] - away_lab)
    swapped = np.linalg.norm(cluster_lab[0] - away_lab) + np.linalg.norm(cluster_lab[1] - home_lab)
    return {0: "home", 1: "away"} if straight <= swapped else {0: "away", 1: "home"}


class TeamAssigner:
    def __init__(
        self,
        embed: Embedder,
        home_lab: np.ndarray | None,
        away_lab: np.ndarray | None,
        fit_after: int = 200,
        refit_every_s: float = 30.0,
        sample_every_s: float = 1.0,
    ) -> None:
        self._embed = embed
        self._home, self._away = home_lab, away_lab
        self.fit_after = fit_after
        self.refit_every_s = refit_every_s
        self.sample_every_s = sample_every_s
        self.reset()

    def reset(self) -> None:
        self._embeddings: list[np.ndarray] = []
        self._colors: list[np.ndarray] = []
        self._track_of: list[int] = []
        self._labels: list[int] = []
        self._last_sample: dict[int, float] = {}
        self._model: KMeans | None = None
        self._mapping: dict[int, Team] = {}
        self._centroids: np.ndarray | None = None
        self._last_fit_pts: float | None = None
        self._fit_size = 0

    @property
    def fitted(self) -> bool:
        return self._model is not None

    def observe(self, track_id: int, crop_bgr: np.ndarray, t_pts: float) -> None:
        last = self._last_sample.get(track_id)
        if last is not None and t_pts - last < self.sample_every_s:
            return
        self._last_sample[track_id] = t_pts
        self._embeddings.append(np.asarray(self._embed(crop_bgr), dtype=float).ravel())
        self._colors.append(crop_color_lab(crop_bgr))
        self._track_of.append(track_id)
        if len(self._embeddings) > MAX_SAMPLES:
            drop = len(self._embeddings) - MAX_SAMPLES
            del self._embeddings[:drop], self._colors[:drop], self._track_of[:drop], self._labels[:drop]
        if self._model is not None:
            self._labels.append(int(self._model.predict(self._embeddings[-1][None, :])[0]))
        else:
            self._labels.append(-1)
        n = len(self._embeddings)
        due = self._last_fit_pts is not None and t_pts - self._last_fit_pts >= self.refit_every_s
        if (self._model is None and n >= self.fit_after) or (due and n > self._fit_size):
            self._fit(t_pts)

    def team_of(self, track_id: int) -> Team:
        if self._model is None:
            return "unknown"
        votes = [lab for tid, lab in zip(self._track_of, self._labels, strict=True) if tid == track_id and lab >= 0]
        if not votes:
            return "unknown"
        label = max(set(votes), key=votes.count)
        return self._mapping.get(label, "unknown")

    def _fit(self, t_pts: float) -> None:
        x = np.stack(self._embeddings)
        model = KMeans(n_clusters=2, n_init=4, random_state=0).fit(x)
        labels = model.labels_.astype(int)
        colors = np.stack(self._colors)
        cluster_lab = np.stack([colors[labels == k].mean(axis=0) if (labels == k).any() else colors.mean(axis=0) for k in (0, 1)])
        if self._home is not None and self._away is not None:
            mapping = map_clusters(cluster_lab, self._home, self._away)
        elif self._centroids is not None:
            # No operator colours: keep the labels of the previous fit by centroid proximity.
            aligned = align_labels(self._centroids, model.cluster_centers_)
            mapping = {new: self._mapping[old] for new, old in aligned.items()}
        else:
            mapping = {0: "home", 1: "away"}
        self._model, self._mapping = model, mapping
        self._centroids = model.cluster_centers_.copy()
        self._labels = [int(v) for v in labels]
        self._last_fit_pts = t_pts
        self._fit_size = len(self._embeddings)


class SiglipEmbedder:
    """The SigLIP vision tower exported to ONNX: a crop in, an L2-normalised vector out."""

    def __init__(self, onnx_path: Path, providers: list[str] | None = None) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(onnx_path), providers=providers or ort.get_available_providers())
        self._input = self._session.get_inputs()[0].name

    @staticmethod
    def preprocess(crop_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop_bgr, (SIGLIP_SIZE, SIGLIP_SIZE), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalised = (rgb - 0.5) / 0.5
        return np.ascontiguousarray(normalised.transpose(2, 0, 1)[None])

    def __call__(self, crop_bgr: np.ndarray) -> np.ndarray:
        out = self._session.run(None, {self._input: self.preprocess(crop_bgr)})[0]
        vec = np.asarray(out, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
