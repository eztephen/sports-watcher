# Golden clips

Short recordings with hand-made truth files that the golden tests score against.
Clips are never committed (this directory is gitignored except for this file, the truth
files, the meta files and the camera segments).

Layout per clip:

- `<clip>.mp4` — 30 s to 3 min of broadcast with the scoreboard visible.
- `<clip>.truth.jsonl` — one `TruthEvent` per line:
  `{"t_pts": 12.4, "type": "score_change", "team": "home", "points": 2}`.
  `team` is `home`, `away` or `unknown`; `points`, `player` and `zone` are optional
  (`zone` is `two`, `three` or `ft` for `shot_attempt`).
- `<clip>.meta.json` (optional) — league, broadcaster, team names, jersey colours
  (`home_color`, `away_color`), `ocr_every_n_frames`, and `checks`: the event types to
  score with their precision and recall floors, for example
  `{"score_change": 0.9, "possession_start": 0.8, "shot_attempt": 0.8}`.
  Defaults: league `pba`, broadcaster `one_sports`, checks `{"score_change": 0.9}`.
- `<clip>.camera.jsonl` (optional) — camera-state segments
  `{"start": 4.5, "end": 6.5, "state": "closeup"}` with states `main`, `replay`,
  `closeup`, `graphics`. The camera golden test asserts frame accuracy of 0.9 or better
  and no confirmed vision event inside a non-main segment.

Vision checks need a perceiver: weights under `models/` or a perception cache for the
clip under `data/cache/` (build it once with `scripts/perceive.py --source <clip>`).
Without either, the vision types are skipped and only the scoreboard checks run.

Make a clip from a recording:

```bash
ffmpeg -ss 00:12:00 -t 90 -i data/game.mp4 -c:v libx264 -crf 18 -an data/golden/q1_first_scores.mp4
```

Find the scoreboard ROI with `scripts/dump_frame.py`, then record it in the league's
`configs/leagues/<league>.yaml` under `broadcasters.<name>`.

Current clips:

- `pba_ros_cvg_q1` — the M0 clip (One Sports, minutes 25 to 27 of the Rain or Shine vs
  Converge quarterfinal): `score_change` plus the camera-state segments.
- `pba_ros_cvg_vision` — minutes 28 to 31 of the same game: `score_change`,
  `possession_start` and `shot_attempt`, drafted from contact sheets and checked by hand.
