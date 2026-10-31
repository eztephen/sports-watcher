# Golden clips

Short recordings with hand-made truth files that the golden tests score against.
Clips are never committed (this directory is gitignored except for this file).

Layout per clip:

- `<clip>.mp4` — 30 to 120 s of broadcast with the scoreboard visible.
- `<clip>.truth.jsonl` — one `TruthEvent` per line: `{"t_pts": 12.4, "type": "score_change", "team": "home", "points": 2}`.
  `team` is `home`, `away` or `unknown`; `points` and `player` are optional.
- `<clip>.meta.json` (optional) — `{"league": "pba", "broadcaster": "one_sports", "home": "TNT", "away": "Magnolia", "ocr_every_n_frames": 5}`.
  Defaults: league `pba`, broadcaster `one_sports`.

Make a clip from a recording:

```bash
ffmpeg -ss 00:12:00 -t 90 -i data/game.mp4 -c:v libx264 -crf 20 -an data/golden/q1_first_scores.mp4
```

Find the scoreboard ROI with `scripts/dump_frame.py`, then record it in the league's
`configs/leagues/<league>.yaml` under `broadcasters.<name>.scoreboard_roi`.

The golden test asserts precision and recall of at least 0.9 on `score_change`.
