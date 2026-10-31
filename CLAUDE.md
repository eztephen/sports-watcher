# CLAUDE.md

Sports Watcher is a Python 3.12 service that reads a basketball broadcast and emits
game events (see `docs/superpowers/specs/2026-10-31-sports-watcher-v1-design.md`).

## Commands

```bash
./.venv/bin/pip install -e ".[dev]"      # install (venv at .venv)
./.venv/bin/python -m pytest -q          # unit tests
./.venv/bin/ruff check .                 # lint
```

Always use `./.venv/bin/python`. Tests under `tests/integration` need the OCR package;
tests under `tests/golden` need clips in `data/golden/` and skip otherwise.

## Rules

- Only Apache-2.0, MIT or BSD runtime dependencies. No Ultralytics, no boxmot, no GPL.
- Events are only appended. A correction is a `retracted` event that supersedes the
  provisional one; nothing is edited in place.
- Secrets live in `.env` only. Weights in `models/`, footage in `data/`; both gitignored.
