#!/usr/bin/env python
"""Train the camera-state probe from labelled clip segments.

  ./.venv/bin/python -m training.train_camera_probe --labels data/labels/camera_state.jsonl \
      --clips data/clips --models models --out models/camera/probe.json --fps 2 \
      --holdout pba_ros_cvg_2500_2700.mp4

Label lines: {"clip": "<file name>", "start": <s>, "end": <s>, "state": "main|replay|closeup|graphics"}.
Frames inside each segment are sampled at --fps, embedded with the SigLIP export, and
joined with the three frame features from the detector and the court model. Without a
detector the probe trains on the embedding alone: the feature columns are zero in
training and their weights are zeroed in the saved probe, so inference ignores them.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run by path or as a module

import argparse
import json
from collections import Counter

import numpy as np

CLASSES = ["main", "replay", "closeup", "graphics"]


def sample_frames(clip: Path, segments: list[dict], fps: float):
    """Yield (image_bgr, state) for frames inside the clip's segments at ``fps``."""
    import av

    step = 1.0 / fps
    next_t = 0.0
    with av.open(str(clip)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
            if t < next_t:
                continue
            next_t = t + step
            state = next((s["state"] for s in segments if s["start"] <= t < s["end"]), None)
            if state is None:
                continue
            yield frame.to_ndarray(format="bgr24"), state


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=Path("data/labels/camera_state.jsonl"))
    ap.add_argument("--clips", type=Path, default=Path("data/clips"))
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--out", type=Path, default=Path("models/camera/probe.json"))
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--holdout", default=None, help="clip file name kept out of training")
    args = ap.parse_args(argv)

    from sklearn.linear_model import LogisticRegression

    from sports_watcher.config import Settings, load_league
    from sports_watcher.perception.camera_state import CameraProbe, frame_features
    from sports_watcher.perception.court import CourtMapper, load_template, order_template
    from sports_watcher.perception.detector import OnnxDetector, OnnxKeypointModel
    from sports_watcher.perception.models import ModelSet
    from sports_watcher.perception.team import SiglipEmbedder
    from sports_watcher.perception.types import box_height

    siglip_path = args.models / "siglip" / "vision.onnx"
    if not siglip_path.is_file():
        raise SystemExit(f"need {siglip_path} (python -m training.export_siglip)")
    embed = SiglipEmbedder(siglip_path)
    models = ModelSet.load(args.models)
    detector = OnnxDetector(models.detector, models.detector_manifest) if models is not None else None
    if detector is None:
        print("no detector under models/: training on embeddings only, feature weights zeroed")
    court = None
    if models is not None and models.court is not None and models.court_manifest is not None:
        league = load_league("pba")
        template_path = Path(Settings().configs_dir) / "court_keypoints" / f"{Path(models.court_manifest.dataset or 'court').name}.yaml"
        names, xy = load_template(template_path, league)
        court = (OnnxKeypointModel(models.court, models.court_manifest), CourtMapper(order_template(names, xy, models.court_manifest.keypoints)))

    by_clip: dict[str, list[dict]] = {}
    for line in args.labels.read_text(encoding="utf-8").splitlines():
        if line.strip():
            seg = json.loads(line)
            by_clip.setdefault(seg["clip"], []).append(seg)

    xs: dict[str, list[np.ndarray]] = {"train": [], "holdout": []}
    ys: dict[str, list[str]] = {"train": [], "holdout": []}
    for clip_name, segments in by_clip.items():
        split = "holdout" if clip_name == args.holdout else "train"
        for image, state in sample_frames(args.clips / clip_name, segments, args.fps):
            if detector is not None:
                dets = detector.detect(image)
                players = [d for d in dets if d.cls == "player"]
                inliers = 0
                if court is not None:
                    kp_xy, kp_conf = court[0].keypoints(image)
                    h = court[1].update(kp_xy, kp_conf, 0.0)
                    inliers = h.inliers if h is not None else 0
                median_h = float(np.median([box_height(d.box) for d in players]) / image.shape[0]) if players else 0.0
                feats = frame_features(inliers, len(players), median_h)
            else:
                feats = np.zeros(3, dtype=np.float32)
            xs[split].append(np.concatenate([embed(image), feats]))
            ys[split].append(state)
        print(f"{clip_name}: sampled into {split}")
    if not xs["train"]:
        raise SystemExit("no training frames: check --labels and --clips")
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(np.stack(xs["train"]), ys["train"])
    classes = list(clf.classes_)
    coef = np.zeros((len(CLASSES), clf.coef_.shape[1]))
    intercept = np.zeros(len(CLASSES))
    for i, c in enumerate(classes):
        coef[CLASSES.index(c)] = clf.coef_[i] if len(classes) > 2 else (clf.coef_[0] * (1 if i == 1 else -1))
        intercept[CLASSES.index(c)] = clf.intercept_[i] if len(classes) > 2 else (clf.intercept_[0] * (1 if i == 1 else -1))
    if detector is None:
        coef[:, -3:] = 0.0
    probe = CameraProbe(coef, intercept, CLASSES)
    probe.save(args.out)
    print(f"trained on {len(ys['train'])} frames {dict(Counter(ys['train']))}; wrote {args.out}")
    if xs["holdout"]:
        pred = [probe.predict(x[:-3], x[-3:])[0] for x in xs["holdout"]]
        per_class = {c: round(np.mean([p == y for p, y in zip(pred, ys["holdout"], strict=True) if y == c]), 3) for c in set(ys["holdout"])}
        print(f"holdout {args.holdout}: accuracy {np.mean([p == y for p, y in zip(pred, ys['holdout'], strict=True)]):.3f} per class {per_class}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
