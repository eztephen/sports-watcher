# CLAUDE.md

Sports Watcher is a Python 3.12 service that reads a basketball broadcast and emits
game events. Spec: `docs/superpowers/specs/2026-10-31-sports-watcher-v1-design.md` plus
the M1 addendum `docs/superpowers/specs/2026-10-31-sports-watcher-m1-vision-design.md`.
Plans: `docs/superpowers/plans/2026-10-31-sports-watcher-m0-skeleton.md` (done),
`docs/superpowers/plans/2026-10-31-sports-watcher-m1-vision.md` (vision).

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
./.venv/bin/python scripts/perceive.py --source <clip> --home-color white --away-color navy   # build the perception cache
./.venv/bin/python scripts/bench_perceiver.py --models models --source <clip> --start 70 --frames 100
./.venv/bin/python scripts/download_models.py                                           # weights from a GitHub release
./.venv/bin/pip install -e ".[train]"   # training extra (rfdetr, torch, transformers, roboflow); see training/README.md
```

Always use `./.venv/bin/python`. `tests/integration` runs real RapidOCR (about 35 s) and
the ONNX perceiver when `models/` has weights; `tests/golden` needs clips in
`data/golden/` and skips otherwise (the vision checks also need weights or a perception
cache in `data/cache/`, and take a few minutes).

## Layout

- `sports_watcher/ingest/` — `FrameSource` adapters (file, url, youtube) behind
  `factory.open_source(SourceSpec)`; `LatestFrameReader` for live sources only.
- `sports_watcher/perception/scoreboard.py` — ROI crop, RapidOCR, clock/period parsing,
  majority vote over 5 reads.
- `sports_watcher/perception/` (M1) — `types.py` (`VisionFrame`, `Track`, `BallObs`,
  `Homography`), `perceiver.py` (`Perceiver` protocol, `ScriptedPerceiver`,
  `build_perceiver`), `cache.py` (per-clip JSONL cache), `onnx_perceiver.py` (the real
  one), `detector.py` + `ball.py` (RF-DETR ONNX, ball crop), `tracker.py` (OC-SORT style
  tracker, ball filter), `team.py` (SigLIP crops → KMeans → operator colours),
  `court.py` (keypoints → homography), `camera_state.py` (probe + hysteresis tracker),
  `models.py` (manifests, `ModelSet`), `offline.py` (cache builder).
- `sports_watcher/state/`, `sports_watcher/sports/` — `GameState`, the `SportPlugin`
  contract (`update_state`, `rules`), and `sports/basketball`: `rules.py` (score and
  period), `vision_state.py` (clock gate, holder, release, rim interaction, lane
  formation), `geometry.py`, `confirmers.py`, `rules_possession.py`, `rules_shots.py`,
  `rules_clock.py`.
- `training/` — Colab/Mac scripts that produce the ONNX weights and manifests;
  `configs/court_keypoints/` maps the court model's keypoints to court metres.
- `sports_watcher/events/` — `Event` model, `EventEngine` (provisional → confirmed /
  retracted, 4 s TTL), sinks, `firsts.fold_firsts`.
- `sports_watcher/watcher.py` — one game: source → OCR → engine → sinks, with
  reconnects for URL sources (`stream_gap` events).
- `sports_watcher/api/` — FastAPI job API and websocket.
- `sports_watcher/replay/`, `sports_watcher/eval/` — replay rig, truth files, scoring.
- `configs/leagues/*.yaml` — court dimensions, period lengths, broadcaster scoreboard boxes
  (NBA Cup and One Sports PBA are calibrated). Pipeline tests use `tests/fixtures/configs`
  (detection ROI, scripted readers) via `tests/helpers/leagues.py`, never the production calibration.

## Rules

- Only Apache-2.0, MIT or BSD runtime dependencies. No Ultralytics, no boxmot, no GPL.
- Events are only appended. A correction is a `retracted` event that supersedes the
  provisional one; nothing is edited in place. Terminal copies keep the original
  `t_pts`/`t_wall`; only `emitted_wall` changes.
- File sources process every frame (deterministic); only live sources drop frames.
- Secrets live in `.env` only. Weights in `models/`, footage in `data/`, the perception
  cache in `data/cache/`, labels in `data/labels/`; all gitignored.
- Vision rules draft only while the game clock runs (OCR clock changed within 2 s);
  free throws and the tip-off are the exceptions. Without weights the watcher behaves
  exactly as M0.
- Test-first: write the failing test, watch it fail, then implement. Rules are tested on
  synthetic plays from `tests/helpers/vision.py`, never on footage.
