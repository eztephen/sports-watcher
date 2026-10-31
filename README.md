# Sports Watcher

Watches a basketball broadcast (file, URL or YouTube live) and emits a timestamped
stream of game events: score changes, period boundaries and, in later milestones,
possession, shots, rebounds and turnovers, with the commentators' voices as a third
evidence channel. The event stream feeds micro-betting markets ("first team to score",
"race to 20", "next possession result") on the GT/APS sportsbook.

Design: `docs/superpowers/specs/2026-10-31-sports-watcher-v1-design.md`.
Research: `docs/research/2026-10-31-sports-watcher-research.md`.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
./.venv/bin/python -m pytest -q
```

Settings come from `.env` (copy `.env.example`); league and broadcaster settings from
`configs/leagues/*.yaml`.
