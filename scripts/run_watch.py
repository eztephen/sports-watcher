#!/usr/bin/env python
"""Watch one game from a file, URL or YouTube link and write events as JSONL.

Examples:
  ./.venv/bin/python scripts/run_watch.py --source data/game.mp4 --home TNT --away Magnolia
  ./.venv/bin/python scripts/run_watch.py --source rtsp://localhost:8554/game --home A --away B
  ./.venv/bin/python scripts/run_watch.py --source youtube:https://www.youtube.com/watch?v=... --home A --away B
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from sports_watcher.config import load_league
from sports_watcher.events.sinks import JsonlSink
from sports_watcher.ingest.base import SourceSpec
from sports_watcher.watcher import GameWatcher


def source_spec(value: str, realtime: bool) -> SourceSpec:
    if value.startswith("youtube:"):
        return SourceSpec(kind="youtube", url=value[len("youtube:"):], realtime=realtime)
    if "://" in value:
        return SourceSpec(kind="url", url=value, realtime=realtime)
    return SourceSpec(kind="file", path=value, realtime=realtime)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="file path, URL, or youtube:<watch url>")
    ap.add_argument("--league", default="pba")
    ap.add_argument("--broadcaster", default="one_sports")
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--out", default="tmp/events.jsonl")
    ap.add_argument("--realtime", action="store_true", help="pace a file at its own frame rate")
    ap.add_argument("--ocr-every", type=int, default=5, help="run scoreboard OCR every N frames")
    ap.add_argument("--game-id", default=None)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    watcher = GameWatcher(
        source_spec(args.source, args.realtime),
        load_league(args.league),
        args.broadcaster,
        game_id=args.game_id or uuid.uuid4().hex[:8],
        home=args.home,
        away=args.away,
        sinks=[JsonlSink(out)],
        ocr_every_n_frames=args.ocr_every,
    )
    thread = watcher.start_in_thread()
    try:
        while thread.is_alive():
            thread.join(timeout=0.5)
    except KeyboardInterrupt:
        print("stopping...", file=sys.stderr)
        watcher.stop()
        thread.join(timeout=30)
    status = watcher.status()
    print(json.dumps({"status": status.model_dump(), "firsts": watcher.firsts().model_dump()}, indent=2))
    print(f"events written to {out}", file=sys.stderr)
    return 0 if status.state == "ended" else 1


if __name__ == "__main__":
    raise SystemExit(main())
