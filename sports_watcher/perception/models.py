"""Model manifests and the set of weights a perceiver runs with.

Layout under ``models/`` (gitignored):

    detector/model.onnx + manifest.json   required
    court/model.onnx + manifest.json      optional: no court model means no homography
    siglip/vision.onnx                    optional: no SigLIP means no teams and no camera probe
    camera/probe.json                     optional: no probe means camera hint "main"
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


class ModelManifest(BaseModel):
    name: str
    family: str  # rfdetr | rfdetr-keypoint | siglip | linear-probe
    size: str | None = None
    resolution: int | None = None
    classes: dict[int, str] = {}
    keypoints: list[str] = []
    dataset: str | None = None
    dataset_version: str | None = None
    trained_on: str | None = None
    licence: str
    attribution: str | None = None
    sha256: str
    output_boxes: str = "dets"
    output_logits: str = "labels"
    output_keypoints: str = "keypoints"


def load_manifest(path: Path) -> ModelManifest:
    return ModelManifest.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class ModelSet:
    detector: Path
    detector_manifest: ModelManifest
    court: Path | None = None
    court_manifest: ModelManifest | None = None
    siglip: Path | None = None
    camera_probe: Path | None = None

    @classmethod
    def load(cls, models_dir: Path, verify: bool = False) -> ModelSet | None:
        """The weights under ``models_dir``, or ``None`` without a detector. With
        ``verify`` each manifest's sha256 is checked against its weight file."""
        root = Path(models_dir)
        detector = root / "detector" / "model.onnx"
        detector_manifest = root / "detector" / "manifest.json"
        if not (detector.is_file() and detector_manifest.is_file()):
            return None
        court = root / "court" / "model.onnx"
        court_manifest = root / "court" / "manifest.json"
        has_court = court.is_file() and court_manifest.is_file()
        siglip = root / "siglip" / "vision.onnx"
        probe = root / "camera" / "probe.json"
        ms = cls(
            detector=detector,
            detector_manifest=load_manifest(detector_manifest),
            court=court if has_court else None,
            court_manifest=load_manifest(court_manifest) if has_court else None,
            siglip=siglip if siglip.is_file() else None,
            camera_probe=probe if probe.is_file() else None,
        )
        if verify:
            for path, manifest in ((ms.detector, ms.detector_manifest), (ms.court, ms.court_manifest)):
                if path is not None and manifest is not None and _sha256(path) != manifest.sha256:
                    raise ValueError(f"{path}: sha256 does not match its manifest")
        return ms

    @property
    def version(self) -> str:
        """Twelve hex digits that change whenever any present weight changes."""
        parts = [f"detector:{self.detector_manifest.sha256}"]
        if self.court_manifest is not None:
            parts.append(f"court:{self.court_manifest.sha256}")
        if self.siglip is not None:
            parts.append(f"siglip:{_sha256(self.siglip)}")
        if self.camera_probe is not None:
            parts.append(f"camera:{_sha256(self.camera_probe)}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def settings_version(base: str, **settings: object) -> str:
    """Twelve hex digits over a weights version and the settings that shape a cache."""
    parts = [base] + [f"{k}={settings[k]!r}" for k in sorted(settings)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
