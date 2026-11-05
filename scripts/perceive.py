#!/usr/bin/env python
"""Run perception once over a clip and write the cache the watcher and golden tests read.

  ./.venv/bin/python scripts/perceive.py --source data/golden/x.mp4 --home-color white --away-color navy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sports_watcher.config import Settings, load_league
from sports_watcher.ingest.base import SourceSpec
from sports_watcher.perception.cache import cache_path
from sports_watcher.perception.models import ModelSet
from sports_watcher.perception.offline import build_cache


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="a clip on disk")
    ap.add_argument("--league", default="pba")
    ap.add_argument("--models", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="cache directory (default SW_CACHE_DIR)")
    ap.add_argument("--home-color", default=None)
    ap.add_argument("--away-color", default=None)
    ap.add_argument("--every", type=int, default=None, help="perceive every N frames (default: about 30 Hz)")
    args = ap.parse_args(argv)

    settings = Settings()
    models = ModelSet.load(args.models or settings.models_dir)
    if models is None:
        raise SystemExit(f"no detector weights under {args.models or settings.models_dir}")
    from sports_watcher.perception.onnx_perceiver import OnnxPerceiver

    perceiver = OnnxPerceiver(models, load_league(args.league), args.home_color, args.away_color)
    clip = Path(args.source)
    path = cache_path(args.out or settings.cache_dir, clip, perceiver.version)
    print(f"writing {path}", file=sys.stderr)
    stats = build_cache(
        SourceSpec(kind="file", path=str(clip)),
        perceiver,
        path,
        every_n=args.every,
        progress=lambda n: print(f"\r{n} frames", end="", file=sys.stderr) if n % 50 == 0 else None,
    )
    print(file=sys.stderr)
    print(json.dumps({
        "cache": str(path),
        "frames_read": stats.frames_read,
        "frames_perceived": stats.frames_perceived,
        "seconds": round(stats.seconds, 1),
        "stage_ms": {k: round(1000 * v, 1) for k, v in perceiver.timings.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
