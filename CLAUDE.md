# CLAUDE.md

Sports Watcher is a Python 3.12 service that reads a basketball broadcast and emits
game events. Spec: `docs/superpowers/specs/2026-10-31-sports-watcher-v1-design.md`.
Plan being executed: `docs/superpowers/plans/2026-10-31-sports-watcher-m0-skeleton.md`.

## Commands

```bash
./.venv/bin/pip install -e ".[dev,youtube]"          # install (venv at .venv)
./.venv/bin/python -m pytest -q                      # whole suite (unit + integration + golden)
./.venv/bin/python -m pytest tests/unit -q           # fast unit tests only
./.venv/bin/ruff check .                             # lint
./.venv/bin/uvicorn sports_watcher.api.main:create_app --factory --port 8000
./.venv/bin/python scripts/run_watch.py --source <file|url|youtube:url> --home A --away B
./.venv/bin/python scripts/dump_frame.py --source <file> --t 120 --roi x,y,w,h
./.venv/bin/python scripts/evaluate.py --events tmp/events.jsonl --truth data/golden/<clip>.truth.jsonl
```

Always use `./.venv/bin/python`. `tests/integration` runs real RapidOCR (about 35 s);
`tests/golden` needs clips in `data/golden/` and skips otherwise.

## Layout

- `sports_watcher/ingest/` — `FrameSource` adapters (file, url, youtube) behind
  `factory.open_source(SourceSpec)`; `LatestFrameReader` for live sources only.
- `sports_watcher/perception/scoreboard.py` — ROI crop, RapidOCR, clock/period parsing,
  majority vote over 5 reads.
- `sports_watcher/state/`, `sports_watcher/sports/` — `GameState`, the `SportPlugin`
  contract (`update_state`, `rules`), and `sports/basketball` (M0: score and period rules).
- `sports_watcher/events/` — `Event` model, `EventEngine` (provisional → confirmed /
  retracted, 4 s TTL), sinks, `firsts.fold_firsts`.
- `sports_watcher/watcher.py` — one game: source → OCR → engine → sinks, with
  reconnects for URL sources (`stream_gap` events).
- `sports_watcher/api/` — FastAPI job API and websocket.
- `sports_watcher/replay/`, `sports_watcher/eval/` — replay rig, truth files, scoring.
- `configs/leagues/*.yaml` — court dimensions, period lengths, broadcaster scoreboard ROI.

## Rules

- Only Apache-2.0, MIT or BSD runtime dependencies. No Ultralytics, no boxmot, no GPL.
- Events are only appended. A correction is a `retracted` event that supersedes the
  provisional one; nothing is edited in place. Terminal copies keep the original
  `t_pts`/`t_wall`; only `emitted_wall` changes.
- File sources process every frame (deterministic); only live sources drop frames.
- Secrets live in `.env` only. Weights in `models/`, footage in `data/`; both gitignored.
- Test-first: write the failing test, watch it fail, then implement.
