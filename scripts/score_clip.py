#!/usr/bin/env python
"""Run the watcher on a golden clip exactly as `tests/golden` does and print its scores.

  ./.venv/bin/python scripts/score_clip.py data/golden/pba_ros_cvg_q4.mp4

Reads the clip's `.meta.json` (teams, colours, broadcaster) and `.truth.jsonl`, writes the
events to `tmp/events_<clip>.jsonl` and prints true / false / missed per event type. With
`--events`, also lists the terminal events of the named types.

Refuses to run when the perception cache for the current code is missing or has no
completion marker: the watcher would rebuild it inline (an hour for three minutes of
clip), and starting while a builder is still writing truncates the builder's file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sports_watcher.eval.score import score
from sports_watcher.replay.truth import load_truth
from tests.golden.test_golden_clips import _meta, _run


def cache_problem(clip: Path, meta: dict) -> str | None:
    """Why the clip cannot be scored from its cache, or None when it can."""
    from sports_watcher.config import Settings, load_league
    from sports_watcher.perception.cache import CachedPerceiver
    from sports_watcher.perception.perceiver import build_perceiver

    perceiver = build_perceiver(
        Settings(), load_league(meta.get("league", "pba")), clip, meta.get("home_color"), meta.get("away_color")
    )
    if perceiver is None:
        return None  # scoreboard only: nothing to build
    if isinstance(perceiver, CachedPerceiver) and not perceiver.complete:
        return (
            f"the perception cache for {clip.name} is missing or incomplete for this code; build it with "
            f"scripts/perceive_memo.py --source {clip} --home-color ... --away-color ... and wait for it to finish"
        )
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--tolerance", type=float, default=1.0)
    ap.add_argument("--events", nargs="*", default=None, help="event types to list (no value: all)")
    ap.add_argument("--out", type=Path, default=None, help="events JSONL (default tmp/events_<clip>.jsonl)")
    args = ap.parse_args(argv)

    clip = args.clip.resolve()
    meta = _meta(clip)
    problem = cache_problem(clip, meta)
    if problem is not None:
        print(problem, file=sys.stderr)
        return 2
    watcher = _run(clip, meta)
    out = args.out or Path("tmp") / f"events_{clip.stem}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(e.model_dump_json() + "\n" for e in watcher.events))

    truth_path = clip.with_suffix(".truth.jsonl")
    if truth_path.exists():
        report = score(watcher.events, load_truth(truth_path), tolerance_s=args.tolerance)
        print(f"{'type':<22}{'true':>6}{'false':>7}{'missed':>8}")
        for event_type, s in report.per_type.items():
            print(f"{event_type:<22}{s.tp:>6}{s.fp:>7}{s.fn:>8}")
        print(f"retraction rate {report.retraction_rate:.2f}")
    else:
        print(f"no truth file at {truth_path}: events written, nothing scored")
    if args.events is not None:
        for e in watcher.events:
            if e.status != "provisional" and (not args.events or e.type in args.events):
                print(f"{e.t_pts:8.2f} {e.type:<20} {e.status:<10} {e.team:<8} {e.reason or ''}")
    print(f"events written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
