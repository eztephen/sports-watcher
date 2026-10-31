# Sports Watcher

Watches a basketball broadcast (file, URL or YouTube live) and emits a timestamped
stream of game events. M0 reads the scoreboard: score changes, period boundaries, and
the "firsts" they imply (first team to score, race to 10 and 20). Later milestones add
possession, shots, rebounds and turnovers from vision, and the commentators' voices as
a third evidence channel. The event stream is meant to feed micro-betting markets on
the GT/APS sportsbook.

- Design: `docs/superpowers/specs/2026-10-31-sports-watcher-v1-design.md`
- Research: `docs/research/2026-10-31-sports-watcher-research.md`
- M0 plan: `docs/superpowers/plans/2026-10-31-sports-watcher-m0-skeleton.md`

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev,youtube]"
./.venv/bin/python -m pytest -q
```

Settings come from `.env` (copy `.env.example`). League and broadcaster settings,
including the scoreboard region, live in `configs/leagues/*.yaml`.

## Run the API

```bash
./.venv/bin/uvicorn sports_watcher.api.main:create_app --factory --port 8000
```

```bash
curl -s -X POST localhost:8000/api/v1/games -H 'content-type: application/json' \
  -d '{"source": {"kind": "file", "path": "data/game.mp4"}, "league": "pba", "broadcaster": "one_sports", "home": "TNT", "away": "Magnolia"}'
```

Then `GET /api/v1/games/{id}` for status, `GET /api/v1/games/{id}/events?since_seq=0`
to backfill, `WS /api/v1/games/{id}/events` for live events (retractions included),
`GET /api/v1/games/{id}/firsts` for the read model, `POST /api/v1/games/{id}/stop`.

## Run on a recording from the command line

```bash
./.venv/bin/python scripts/run_watch.py --source data/game.mp4 --league pba --home TNT --away Magnolia --out tmp/events.jsonl
```

`--source` also takes `rtsp://`, `srt://`, `udp://`, an HLS manifest, or
`youtube:<watch url>` (development only; resolved with yt-dlp).

## Calibrate the scoreboard region

```bash
./.venv/bin/python scripts/dump_frame.py --source data/game.mp4 --t 120 --out tmp/frame.png --roi 0.02,0.86,0.40,0.10
```

Open `tmp/frame.png` and `tmp/frame_roi.png`, adjust the fractions until the crop holds
both scores, the clock and the period, then record them under
`broadcasters.<name>.scoreboard_roi` in the league YAML.

## Evaluate against a truth file

```bash
./.venv/bin/python scripts/evaluate.py --events tmp/events.jsonl --truth data/golden/clip.truth.jsonl
```

See `data/golden/README.md` for the clip and truth-file layout. Golden clips present in
`data/golden/` are scored automatically by `tests/golden`.

## Replay rig (latency tests)

Start mediamtx (`scripts/replay_push.py` prints a docker compose snippet), then:

```bash
./.venv/bin/python scripts/replay_push.py --input data/game.mp4 --url rtsp://localhost:8554/game
```

and watch `rtsp://localhost:8554/game` with the API or the CLI. Events carry
`behind_live_s` from the adapter's latency class and `emitted_wall` minus `t_wall` is
the pipeline's own latency. That latency only means something on a live source or a
file run with `--realtime`; a plain file run finishes faster than real time and shows
negative values.

## Scoreboard layouts

Two ways to describe a broadcaster's scoreboard in the league YAML:

- `scoreboard_roi` alone: one box, text detection finds the fields, `home_side` says
  which score is the home team. Works when every field is a word or multi-digit number.
- `fields` (preferred): one box per field (`home`, `away`, `clock`, `period`,
  `shot_clock`), each read with recognition only. Needed when scores are large single
  digits, which text detectors miss. Both shipped broadcasters are calibrated this way:
  `configs/leagues/nba.yaml` from the NBA Cup broadcast and `configs/leagues/pba.yaml`
  from the One Sports PBA feed (top row away, bottom row home).

Reads are also gated for plausibility in the basketball plug-in: a read without a game
clock (a full-screen graphic) is ignored, the period may only advance by one, and a read
where both scores jumped at once needs a fully agreeing vote window.

## Golden clips

`data/golden/` holds the first golden clip's truth and metadata (committed) for
`pba_ros_cvg_q1.mp4` (not committed). Re-create the video with:

```bash
./.venv/bin/yt-dlp -f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]" --merge-output-format mp4 \
  --download-sections "*00:25:00-00:27:00" -o "data/golden/pba_ros_cvg_q1.%(ext)s" \
  "https://www.youtube.com/watch?v=TyYUFM0h-6c"
```

## Event lifecycle

Every event is appended, never edited. A rule emits a `provisional` event at once; a
confirmer turns it into a `confirmed` or `retracted` successor whose `supersedes` is the
provisional's id. Provisionals expire after 4 s of stream time (`reason: ttl`); a stream
that ends retracts everything pending (`reason: stream_ended`). Consumers key on
`(game_id, seq)`.
