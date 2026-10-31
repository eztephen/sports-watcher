#!/usr/bin/env python
"""Score an events JSONL against a truth JSONL and print a per-type table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sports_watcher.eval.score import score
from sports_watcher.events.firsts import Firsts
from sports_watcher.events.model import Event
from sports_watcher.replay.truth import load_truth


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", required=True, help="events.jsonl written by the watcher")
    ap.add_argument("--truth", required=True, help="truth.jsonl (TruthEvent per line)")
    ap.add_argument("--firsts", help="firsts JSON as returned by the API or CLI")
    ap.add_argument("--truth-firsts", help='JSON of dotted paths, e.g. {"first_score.team": "home"}')
    ap.add_argument("--tolerance", type=float, default=1.0)
    ap.add_argument("--out", default="tmp/report.json")
    args = ap.parse_args(argv)

    events = [Event.model_validate_json(l) for l in Path(args.events).read_text().splitlines() if l.strip()]
    firsts = Firsts.model_validate_json(Path(args.firsts).read_text()) if args.firsts else None
    truth_firsts = json.loads(Path(args.truth_firsts).read_text()) if args.truth_firsts else None
    report = score(events, load_truth(args.truth), args.tolerance, firsts, truth_firsts)

    print(f"{'type':<18}{'tp':>5}{'fp':>5}{'fn':>5}{'P':>7}{'R':>7}{'p50 s':>8}{'p95 s':>8}")
    for typ, s in report.per_type.items():
        p50 = "-" if s.latency_p50_s is None else f"{s.latency_p50_s:.2f}"
        p95 = "-" if s.latency_p95_s is None else f"{s.latency_p95_s:.2f}"
        print(f"{typ:<18}{s.tp:>5}{s.fp:>5}{s.fn:>5}{s.precision:>7.2f}{s.recall:>7.2f}{p50:>8}{p95:>8}")
    print(f"retraction rate {report.retraction_rate:.2f}; firsts accuracy {report.firsts_accuracy}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2))
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
