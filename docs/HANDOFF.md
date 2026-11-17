# Handoff (2026-11-17)

Written so a new Claude account, or a new person, can pick this project up without the
old chat history. Everything a session needs is in this repo or listed below.

## Where the project stands

- `main` at `e7d50fc` holds M0 (scoreboard events) and M1 (vision events). 302 unit
  tests, ruff clean, golden tests pass apart from the camera-state check on
  `pba_ros_cvg_q1`.
- Scores per clip and everything parked or open: the "State of M1" section and the
  sections after it in `docs/superpowers/plans/2026-10-31-sports-watcher-m1-vision.notes.md`.
- Solid on PBA file replays: `score_change`, `shot_made`, `shot_attempt`, free throws,
  `timeout`. Weak: `possession_start`, `rebound`, `turnover`.
- Live on the M1 Pro Mac only the scoreboard side keeps up. Perception takes about
  531 ms a frame.

## Open items

These two need Stephen:

1. GPU timing on Colab (`training/colab_m1_weights.ipynb`), the remaining route to live
   vision. Needs a Google sign-in.
2. Shot and possession truth for the second game, `data/golden/pba_gin_smb_q1`. Someone
   has to watch the clip and write the truth lines.

Then, in the order of the v1 spec: M2 commentary (ASR as an evidence channel), M3
hardening, and spec 2 (market layer plus the APS adapter that publishes LSports-shaped
Feed XML). Smaller leftovers: deferred minors in
`docs/superpowers/plans/2026-10-31-sports-watcher-m0-skeleton.notes.md`, the PyAV
licence check (its FFmpeg build may bundle GPL x264), the PBA court keypoint model
(parked until someone labels points), and more replay labels for the camera probe.

## Decisions Stephen made

- Basketball first, PBA as the first league (One Sports on YouTube). YouTube downloads
  are for development only, never a product feed.
- NBA League Pass and Disney+ are DRM-locked and are not inputs. NBA terms bar gambling
  use of NBA statistics.
- The watcher emits a running event stream. Ingest is pluggable (file, url, youtube).
  Commentary audio is a required evidence channel.
- Runtime dependencies are Apache-2.0, MIT or BSD only.
- Work goes research, then design, then code, with the superpowers skills and test-first.
  Prose follows the humanizer rules in `~/.claude/CLAUDE.md`.

## What is not in git

This repository is public. The items below exist only on Stephen's Mac, under the repo
folder unless noted. Back them up before wiping the machine.

| Path | Size | What | How to get it back |
|---|---|---|---|
| `.env` | small | `ROBOFLOW_API_KEY` | Roboflow account settings |
| `models/` | 0.9 GB | ONNX weights, camera probes, three trial court models | `scripts/download_models.py` fetches the `m1-models` release; the trial court models and old probes are local only |
| `data/golden/*.mp4` | 0.7 GB | six golden clips | yt-dlp section commands in the meta files and `data/golden/README.md` |
| `data/clips/`, `data/highlights/` | 2.1 GB | source footage | yt-dlp again |
| `data/cache/` | 1.1 GB | perception caches | `scripts/perceive_memo.py`, slow the first time |
| `data/datasets/`, `data/labels/` | 0.3 GB | training sets, camera labels, PBA court prelabels, kit crops | datasets from Roboflow; the hand labels cannot be regenerated |
| `tmp/` | 1.2 GB | trace scripts, run outputs, `kitset/` (300 hand-labelled crops) | cannot be regenerated |
| `data/handoff/claude-sessions.tgz` | 0.34 GB | both Claude Code transcripts and the memory files | copy made on 2026-11-17 |

The transcripts may contain secrets read during the work, so keep that archive out of
git and off shared drives.

## Things outside this repo

- GT/APS sportsbook sources: `~/Work_These_Out/GlobalNet/repositories` (`aps` is the
  maintained engine; `local-tools/rabbitmq` imitates LSports).
- The GT `sports` Postgres database needs Stephen's VPN. Host and credentials are in
  `gtservices/goblin/src/properties/pooler.xml` and must never be copied here.
- GodsEye, the app this one is modelled on: `~/Work_These_Out/AAFE/repositories/GodsEye/godseye-backend`.
- Global Claude instructions: `~/.claude/CLAUDE.md` (humanizer and superpowers rules).
  The humanizer and superpowers plugins live in `~/Work_These_Out/LLMWritings`.

## Starting on a new Claude account

1. Sign in to the new account in Claude Code on the same Mac and open this folder. The
   project `CLAUDE.md` loads by itself.
2. Claude's memory for this project is stored on disk by folder path, not by account:
   `~/.claude/projects/-Users-eztephen-Work-These-Out-EZFEICO-sports-watcher/memory/`.
   On the same Mac it carries over. On another machine, copy `docs/handoff/memory/*.md`
   into that folder.
3. Tell the new session: "Read docs/HANDOFF.md and the last sections of the M1 notes,
   then continue with the open items."
4. Old chats do not move between accounts. The archive in `data/handoff/` is the record;
   it unpacks into `~/.claude/projects/` if the old sessions are wanted in the resume list.
