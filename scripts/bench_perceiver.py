#!/usr/bin/env python
"""Time each perception stage on real frames and count frames with a detected ball.

  ./.venv/bin/python scripts/bench_perceiver.py --models models --source data/clips/x.mp4 --start 70 --frames 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sports_watcher.config import load_league
from sports_watcher.ingest.base import SourceSpec
from sports_watcher.ingest.factory import open_source
from sports_watcher.perception.models import ModelSet


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--source", required=True)
    ap.add_argument("--league", default="pba")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the clip")
    ap.add_argument("--frames", type=int, default=100, help="perception frames to time")
    ap.add_argument("--every", type=int, default=2, help="perceive every N frames")
    ap.add_argument("--home-color", default=None)
    ap.add_argument("--away-color", default=None)
    args = ap.parse_args(argv)

    models = ModelSet.load(args.models)
    if models is None:
        raise SystemExit(f"no detector under {args.models}")
    from sports_watcher.perception.onnx_perceiver import OnnxPerceiver

    perceiver = OnnxPerceiver(models, load_league(args.league), args.home_color, args.away_color)
    source = open_source(SourceSpec(kind="file", path=args.source))
    source.open()
    seen = balls = 0
    t0 = time.perf_counter()
    try:
        for i, frame in enumerate(source.frames()):
            if frame.pts_s < args.start or i % args.every:
                continue
            vf = perceiver.perceive(frame)
            seen += 1
            balls += vf.ball is not None and vf.ball.source == "detected"
            if seen >= args.frames:
                break
    finally:
        source.close()
    wall = time.perf_counter() - t0
    report = {
        "models": models.version,
        "detector": {"size": models.detector_manifest.size, "resolution": models.detector_manifest.resolution},
        "frames": seen,
        "ms_per_frame": round(1000 * wall / max(seen, 1), 1),
        "stage_ms": {k: round(1000 * v, 1) for k, v in perceiver.timings.items()},
        "frames_with_detected_ball": balls,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
