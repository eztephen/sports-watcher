"""RF-DETR ONNX wrappers: the box detector and the court keypoint model."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sports_watcher.perception.models import ModelManifest
from sports_watcher.perception.tracker import Detection

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_THRESHOLD = 0.3


def default_providers() -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    return [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider") if p in available] or available


def preprocess(image_bgr: np.ndarray, resolution: int) -> np.ndarray:
    """Squash to a square, RGB, [0, 1], ImageNet normalisation, NCHW float32."""
    resized = cv2.resize(image_bgr, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalised = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalised.transpose(2, 0, 1)[None])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode_detections(
    boxes: np.ndarray,
    logits: np.ndarray,
    image_hw: tuple[int, int],
    classes: dict[int, str],
    threshold: float = DEFAULT_THRESHOLD,
    max_dets: int = 100,
) -> list[Detection]:
    """``boxes`` are (Q, 4) cxcywh normalised to the input; ``logits`` are (Q, C). The score
    is the sigmoid of the best class per query; indices absent from ``classes`` are dropped."""
    h, w = image_hw
    probs = _sigmoid(np.asarray(logits, dtype=np.float64))
    best = probs.argmax(axis=1)
    scores = probs[np.arange(len(best)), best]
    order = np.argsort(-scores)
    out: list[Detection] = []
    for q in order:
        score = float(scores[q])
        if score < threshold or len(out) >= max_dets:
            break
        name = classes.get(int(best[q]))
        if name is None:
            continue
        cx, cy, bw, bh = (float(v) for v in boxes[q])
        x1, y1 = max(0.0, (cx - bw / 2) * w), max(0.0, (cy - bh / 2) * h)
        x2, y2 = min(float(w), (cx + bw / 2) * w), min(float(h), (cy + bh / 2) * h)
        if x2 <= x1 or y2 <= y1:
            continue
        out.append(Detection((x1, y1, x2, y2), name, score))
    return out


class _Session:
    def __init__(self, model_path: Path, providers: list[str] | None) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=providers or default_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def run(self, x: np.ndarray) -> dict[str, np.ndarray]:
        outputs = self.session.run(None, {self.input_name: x})
        return dict(zip(self.output_names, outputs, strict=True))


class OnnxDetector:
    def __init__(self, model_path: Path, manifest: ModelManifest, providers: list[str] | None = None) -> None:
        self.manifest = manifest
        self.resolution = int(manifest.resolution or 512)
        self.threshold = DEFAULT_THRESHOLD
        self._session = _Session(model_path, providers)

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        out = self._session.run(preprocess(image_bgr, self.resolution))
        boxes = np.asarray(out[self.manifest.output_boxes])[0]
        logits = np.asarray(out[self.manifest.output_logits])[0]
        return decode_detections(boxes, logits, image_bgr.shape[:2], self.manifest.classes, self.threshold)


class OnnxKeypointModel:
    """The court keypoint model: the keypoints of the most confident court query."""

    def __init__(self, model_path: Path, manifest: ModelManifest, providers: list[str] | None = None) -> None:
        self.manifest = manifest
        self.resolution = int(manifest.resolution or 576)
        self._session = _Session(model_path, providers)

    def keypoints(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(K, 2)`` pixel positions and ``(K,)`` confidences."""
        h, w = image_bgr.shape[:2]
        out = self._session.run(preprocess(image_bgr, self.resolution))
        logits = np.asarray(out[self.manifest.output_logits])[0]
        kps = np.asarray(out[self.manifest.output_keypoints])[0]
        k = len(self.manifest.keypoints) or kps.shape[-2]
        kps = kps.reshape(kps.shape[0], k, -1)
        best = int(_sigmoid(logits).max(axis=1).argmax())
        chosen = kps[best]
        xy = chosen[:, :2].astype(float) * np.array([w, h], dtype=float)
        conf = chosen[:, 2].astype(float) if chosen.shape[1] > 2 else np.ones(k)
        if conf.min() < 0 or conf.max() > 1:
            conf = _sigmoid(conf)
        return xy, conf
