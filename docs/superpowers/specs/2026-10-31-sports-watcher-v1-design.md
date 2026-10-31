# Sports Watcher v1: design spec

Date: 2026-10-31. Status: approved in conversation, awaiting written review.
Research behind every choice here: `docs/research/2026-10-31-sports-watcher-research.md`.
M1 addendum: `2026-10-31-sports-watcher-m1-vision-design.md` revises sections 4, 5.2,
5.4 to 5.7, 7, 8 and 10 for the vision milestone.

## 1. Purpose

Sports Watcher watches a live or recorded basketball broadcast and publishes a
timestamped stream of what is happening: tip-off, possession, shot attempt, made or
missed, free throw, rebound, turnover, period boundaries, score changes. It also
answers the "first X" questions over that stream: who won the tip, who scored first,
what the first basket was, who took the first shot, race to N points, and the result of
each possession.

It is a Python service in the shape of GodsEye (FastAPI, detect, track, classify,
latest-frame live reader), with three evidence channels fused per event: vision,
scoreboard OCR, and the commentators' voices.

First league: PBA (FIBA court, 24 s shot clock, 10-minute quarters). NBA follows as
configuration once a lawful feed exists.

## 2. Scope

In scope for this spec (v1):

1. Ingest from whichever feed we are given, through one adapter interface.
2. Perception: players, ball, rim, referees; tracking; team assignment; court
   homography; scoreboard OCR; camera-state classification.
3. Commentary channel: streaming speech-to-text and cue extraction.
4. Basketball game-state tracker and event rules, behind a sport plug-in interface.
5. Event model with provisional, confirmed and retracted states, and a "firsts" read
   model.
6. FastAPI service: start and stop a game job, poll status, stream events over a
   websocket, read firsts.
7. Replay rig and evaluation harness: golden clips with truth files, precision,
   recall and latency per event type.

Out of scope (spec 2, later): pricing or opening markets, settlement messages,
LSports-shaped XML into the APS RabbitMQ queue, new `MarketType` entries, player
jersey-number OCR, pose estimation, multi-camera, soccer plug-in.

## 3. Constraints and decisions

- Language: Python 3.12. Inference engines: PyTorch with MPS or ONNX Runtime with
  CoreML on the Mac; TensorRT FP16 on a Linux GPU box later; the same ONNX graphs.
- Licences: Apache-2.0 or MIT only for anything shipped. Ultralytics YOLO, boxmot,
  T-DEED and PnLCalib are reference material, never dependencies.
- Latency targets, measured from frame receipt: provisional event under 0.5 s,
  confirmed event under 3 s. Events carry the adapter's own "behind live" estimate.
- Events are only derived from main-camera frames. Replays and close-ups freeze the
  state machine.
- One process per game. No shared state between games.
- Commentary confirms and attributes; it never triggers an event on its own except a
  low-confidence provisional score cue that the scoreboard must confirm.
- Development data: PBA full games from the One Sports YouTube channel (free, no
  delay). DRM services (League Pass, Disney+) are not inputs.
- Secrets never live in the repo. `.env` only, with `.env.example` committed.

## 4. Architecture

```
feed (file | HLS/DASH | YouTube | RTSP | RTMP/SRT/UDP push | capture card | screen)
  │  FrameSource adapter: Frame(pts, wall, image) + AudioChunk(pts, pcm)
  ▼
LatestFrameReader (drop policy, counts dropped frames)
  │
  ├─► CameraStateClassifier ──► main | replay | closeup | graphics
  │
  ├─► Perception (main-camera frames only)
  │     Detector (player, ball, rim, referee) ─► Tracker ─► TeamAssigner
  │     CourtMapper (keypoints ─► homography ─► court metres)
  │     ScoreboardReader (score, clock, period, shot clock)
  │
  ├─► Audio: StreamingASR ─► CueExtractor ─► CommentaryCue(t, kind, team, player, text)
  │
  ▼
GameState (possession, ball track, holder, score, clock, period, camera state)
  │
  ▼
EventEngine: SportPlugin.rules(state) ─► provisional Event
             EvidenceFuser (scoreboard, cues, vision) ─► confirmed | retracted
             TTL expiry ─► retracted
  │
  ├─► Sinks: in-process queue, JSONL per game, websocket broadcaster
  └─► FirstsReadModel (fold over confirmed events)
  ▼
FastAPI: POST /games, GET /games/{id}, WS /games/{id}/events, GET /games/{id}/firsts
```

Everything above the sport plug-in is sport-agnostic. `sports/basketball` supplies
the court template, detector class map, cue lexicon, state tracker, rules and firsts.

## 5. Components and interfaces

### 5.1 Ingest (`sports_watcher/ingest/`)

```python
class FrameSource(Protocol):
    latency_class: LatencyClass        # FILE, PUSH, PULL, HLS, CAPTURE
    def open(self) -> None: ...
    def frames(self) -> Iterator[Frame]:        # Frame(pts_s, wall, image_bgr, index)
    def audio(self) -> Iterator[AudioChunk]:    # AudioChunk(pts_s, pcm16_mono_16k)
    def close(self) -> None: ...
```

Adapters, all on PyAV with low-delay options (`nobuffer`, `low_delay`, small probesize,
`live_start_index=-1`):

| Adapter | Input | Latency class | Notes |
|---|---|---|---|
| `FileSource` | path or directory | FILE | Paces to real time when `realtime=True` |
| `UrlSource` | HLS or DASH manifest, RTSP, RTMP, SRT, UDP `mpegts` | PULL, PUSH, HLS | One adapter; PyAV picks the demuxer from the URL |
| `YouTubeSource` | YouTube watch URL | HLS | Resolves the manifest with `yt-dlp -g`, then delegates to `UrlSource`; development only |
| `CaptureSource` | UVC capture card or camera index | CAPTURE | AVFoundation on the Mac, V4L2 on Linux |
| `ScreenSource` | display or window id | CAPTURE | Mac only, quick trials |

`LatestFrameReader` (from GodsEye) sits between the adapter and the pipeline so
inference never falls behind: it hands out the newest frame and counts drops.
Reconnect with exponential backoff on stream loss; a game job records every gap.

### 5.2 Perception (`sports_watcher/perception/`)

| Module | Model | Licence | Output |
|---|---|---|---|
| `detector.py` | RF-DETR nano/small fine-tuned on Roboflow basketball player, ball and court sets; classes `player, ball, rim, referee` | Apache-2.0 | boxes, scores, class |
| `tracker.py` | OC-SORT | MIT | track ids over player boxes; ball tracked separately with a constant-velocity Kalman filter and a WASB-style crop refinement when detection drops |
| `team.py` | SigLIP crop embeddings, UMAP, KMeans k=3 (two teams plus referees), re-fit every 30 s; cluster to home/away by nearest match to the `home_color` / `away_color` set in the game request (operator supplies them; the scoreboard swatch is a later refinement) | Apache-2.0 / BSD | team per track |
| `court.py` | keypoint model on the Roboflow basketball-court set, homography to the league template, temporal smoothing, confidence | Apache-2.0 | `image_xy -> court_xy` (metres), zone lookup |
| `scoreboard.py` | ROI located once per broadcaster (config), digit recogniser (small CNN or PaddleOCR) with majority vote over 5 frames | Apache-2.0 | `ScoreboardState(home, away, clock_s, period, shot_clock_s, confidence)` |
| `camera_state.py` | small image classifier (MobileNet-class) trained on PBA frames labelled main, replay, closeup, graphics; replay also detected by the broadcaster's replay wipe or "REPLAY" bug | Apache-2.0 | state per frame with hysteresis |

All models are loaded once per process and run through `onnxruntime` (CoreML or CUDA
provider) or PyTorch MPS, chosen by config.

### 5.3 Audio (`sports_watcher/audio/`)

- `asr.py`: faster-whisper (MIT) in streaming mode on 16 kHz mono, small or base
  model on the Mac, with word-level timestamps. Language set per league config
  (English and Filipino for PBA).
- `cues.py`: matches transcript segments against the sport lexicon and the game
  roster. Produces `CommentaryCue(t_pts, kind, team, player, text, confidence)`.
  Kinds for basketball: `tip`, `three`, `two`, `and_one`, `free_throw`, `miss`,
  `rebound`, `turnover`, `steal`, `block`, `timeout`, `foul`, `period`. Roster names
  fuzzy-matched (rapidfuzz, MIT); team names and nicknames from league config.
- `sfx.py` (optional, off by default): PANNs tagger from GodsEye for horn and crowd
  spikes, used as a weak cue for period end and scores.

Commentary lags the play by 1 to 3 s. Cues attach to events whose `t_pts` falls in
`[t_cue - 4 s, t_cue + 1 s]`.

### 5.4 Game state (`sports_watcher/state/`)

`GameState` holds: period, clock, shot clock, score, camera state, player tracks with
team and court position, ball track, current holder (track id, team, since frame),
possession (team, since pts), last shot attempt, pending free throws. It is updated
only from main-camera frames; on return from a replay it re-acquires possession before
emitting anything.

### 5.5 Sport plug-in (`sports_watcher/sports/`)

```python
class SportPlugin(Protocol):
    name: str
    def court_template(self, league: LeagueConfig) -> CourtTemplate: ...
    def detector_classes(self) -> dict[int, str]: ...
    def cue_lexicon(self, league: LeagueConfig) -> Lexicon: ...
    def update_state(self, state: GameState, obs: Observation) -> None: ...
    def rules(self) -> list[Rule]:             # each Rule: (state, obs) -> list[Event]
    def confirmers(self) -> list[Confirmer]:   # (event, evidence window) -> confirmed | retracted | pending
    def fold_firsts(self, events: list[Event]) -> Firsts: ...
```

League config (`configs/leagues/pba.yaml`, `nba.yaml`): court length and width,
three-point radius and corner distance, key dimensions, period length and count,
shot clock, scoreboard ROI per broadcaster, roster source, ASR languages.

### 5.6 Basketball rules (`sports_watcher/sports/basketball/rules.py`)

| Event | Trigger (provisional) | Confirm | Retract |
|---|---|---|---|
| `period_start` / `period_end` | scoreboard period change or clock reset to full; horn cue | second OCR read agrees | OCR disagreement |
| `tip_off` | period 1, clock at full, ball rises then falls near centre circle, two players within 2 m of it | first controlled possession within 5 s; cue `tip` | no possession within 10 s |
| `possession_start` | holder changes team: ball centre inside or within 0.5 m of a player box for 5 consecutive main-camera frames | holds for 15 frames | holder flips back within 15 frames |
| `shot_attempt` | ball leaves holder with upward velocity and the projected trajectory passes within 1.5 rim widths of the rim box within 1.5 s; shooter is last holder; zone from shooter's feet via homography (two, three, unknown) | ball reaches rim region, or score delta, or cue `two`/`three`/`miss` | ball never approaches rim and possession continues |
| `shot_made` | score delta equals 2 or 3 within 3 s of an attempt, or ball passes through rim box top to bottom | both signals, or delta plus cue | delta later reverts (review) |
| `shot_missed` | ball leaves rim region with no delta within 3 s | next possession starts | late delta arrives |
| `free_throw_attempt` / `_made` / `_missed` | clock stopped, lane formation (players at key edges), single shooter at the line; delta of 1 | delta or cue `free_throw` | formation breaks before release |
| `rebound` | first controlled possession after a miss, offensive or defensive by team | possession holds 15 frames | holder flips back |
| `turnover` | possession changes with no shot attempt and clock running, or shot clock resets to full without a shot | possession holds 15 frames; cue `turnover`/`steal` | shot attempt detected retroactively |
| `score_change` | scoreboard delta of any size | majority vote over 5 frames | OCR flips back |
| `timeout` | clock stopped and camera state leaves main for over 20 s, or cue `timeout` | clock still stopped at 30 s | play resumes within 10 s |
| `shot_made` (cue-only) | cue `two`, `three` or `and_one` with no vision attempt in the window; `confidence` 0.3 | scoreboard delta within 4 s | no delta by TTL |

When a score delta arrives with no matching attempt (occluded play), the engine emits a
retroactive `shot_attempt` with `confidence` 0.4, `court_xy` null and zone from the
delta (1 free throw, 2 two, 3 three).

### 5.7 Event model (`sports_watcher/events/model.py`)

```python
class Event(BaseModel):
    id: str; game_id: str; seq: int
    type: str                     # sport-specific name, e.g. "shot_attempt"
    status: Literal["provisional", "confirmed", "retracted"]
    supersedes: str | None        # id of the provisional this confirms or retracts
    t_pts: float; t_wall: datetime; behind_live_s: float | None
    period: int | None; game_clock_s: float | None; shot_clock_s: float | None
    team: Literal["home", "away", "unknown"]
    player: PlayerRef | None      # track_id, name, jersey (always null in v1), confidence
    court_xy: tuple[float, float] | None; zone: str | None
    evidence: Evidence            # score_before, score_after, ball_speed, rim_distance, cues[], camera_state, detector_conf
    confidence: float
    source: SourceInfo            # adapter, latency_class
```

Every provisional has a TTL (default 4 s). The engine always emits a terminal
`confirmed` or `retracted` with `supersedes` set. Consumers key on `(game_id, seq)`.

`Firsts` (read model): `first_possession`, `tip_off_winner`, `first_shot_attempt`,
`first_score` (team, kind FT/2/3, player), `first_three`, `race_to` for 10 and 20 per
period, and `possessions[]` with result `score | miss | turnover | foul | end_period`.

### 5.8 API (`sports_watcher/api/`)

Same job pattern as GodsEye:

- `POST /api/v1/games` body `{source: {kind, url|path|device}, league: "pba", home, away, roster?, realtime?}` returns `game_id`; the job runs on a daemon thread.
- `GET /api/v1/games/{id}` status, frame and drop counters, camera state, current score and clock, last 20 events.
- `WS /api/v1/games/{id}/events` live events as JSON, including retractions.
- `GET /api/v1/games/{id}/events?since_seq=` backfill.
- `GET /api/v1/games/{id}/firsts` the read model.
- `POST /api/v1/games/{id}/stop`.
- `GET /api/v1/health`.

Settings through pydantic-settings and `.env`, as in GodsEye.

### 5.9 Replay rig and evaluation (`sports_watcher/replay/`, `sports_watcher/eval/`)

- `replay/rig.py`: `ffmpeg -re` loops a recording into a local mediamtx over RTSP or
  SRT so the pipeline is tested against a pushed live stream with known latency.
- `replay/truth.py`: truth schema, one JSONL per clip: `{t_pts, type, team, player?,
  points?}`; `scripts/label_truth.py` writes it while scrubbing a clip.
- `eval/score.py`: per event type precision and recall at a plus or minus 1 s match
  window, provisional and confirmed latency p50 and p95, retraction rate, firsts
  accuracy. Prints a table and writes JSON.

## 6. Error handling

- Stream loss: reconnect with backoff; emit `stream_gap` events; never crash the job.
- Camera cut: freeze state, no events; log the time out of main camera.
- Homography failure: events still emitted with `court_xy` null and zone `unknown`.
- OCR and vision disagree: event stays provisional until TTL, then retracted with
  reason; the firsts model ignores retracted events.
- ASR failure or no audio: commentary channel disabled, pipeline continues.
- Unknown team for a track: event `team` is `unknown`; firsts skip it.
- Every event carries the evidence it was built from, so a wrong call can be audited
  from the JSONL alone.

## 7. Testing

Test-driven throughout. Three tiers:

1. Unit (CPU, no models, runs in CI): rules on synthetic track sequences built by a
   fixture helper; firsts fold; cue extractor against transcript strings; scoreboard
   parser against synthetic digit images; PTS and wall-clock mapping; TTL and
   retraction behaviour; adapter contract tests against a short bundled MP4.
2. Golden (needs clips, skipped when absent): ten PBA clips of 30 to 120 s each
   covering tip-off, made two, made three, miss and rebound, free throws, turnover,
   timeout, a replay segment and a hard camera cut, each with a truth file. Assert
   per-type recall and precision floors and the firsts answers.
3. Latency (manual or nightly): replay rig over RTSP, report p50 and p95 for
   provisional and confirmed events.

## 8. Repository layout

```
sports-watcher/
  pyproject.toml           # package sports_watcher; extras: dev, mac, cuda
  README.md  CLAUDE.md  .env.example  .gitignore
  configs/watcher.yaml  configs/leagues/{pba,nba}.yaml
  sports_watcher/
    config.py  clock.py
    ingest/      base.py latest_frame.py file.py url.py youtube.py capture.py screen.py
    perception/  detector.py tracker.py team.py court.py scoreboard.py camera_state.py
    audio/       asr.py cues.py sfx.py
    state/       game_state.py
    events/      model.py engine.py fuser.py sinks.py firsts.py
    sports/      base.py basketball/{plugin,rules,court_templates,lexicon}.py
    api/         main.py routes/{games,health}.py schemas.py jobs.py
    replay/      rig.py truth.py
    eval/        score.py
  scripts/       download_models.py make_clip.py label_truth.py run_watch.py
  tests/unit/    tests/golden/
  models/ data/  # gitignored
  docs/research/ docs/superpowers/specs/ docs/superpowers/plans/
```

Dependencies (all Apache-2.0, MIT or BSD): fastapi, uvicorn, pydantic, pydantic-settings,
av, numpy, opencv-python-headless, onnxruntime, torch (MPS), rfdetr, filterpy,
scikit-learn, umap-learn, open-clip or transformers for SigLIP, faster-whisper,
rapidfuzz, paddleocr (optional), websockets, pytest, ruff. Weights are downloaded by
`scripts/download_models.py`, never committed.

## 9. Milestones

- M0 Skeleton: package, config, ingest adapters (file, URL, YouTube), latest-frame
  reader, event model, sinks, API with jobs, scoreboard OCR, `score_change`,
  `period_*` and the firsts that need only the scoreboard (first team to score, race
  to N). Unit tests green. Runs on a PBA recording on the Mac.
- M1 Vision events: detector, tracker, team, court, camera state; possession, shot
  attempt, made or missed, rebound, turnover, free throws; golden clips and
  evaluation script.
- M2 Commentary: ASR, cue extractor, fusion into confidence and player attribution.
- M3 Hardening: replay rig latency report, retraction tuning, Linux GPU build.
- Spec 2: market layer and APS adapter.

## 10. Resolved and open questions

Resolved: Python; basketball first with PBA; Apache and MIT only; standalone event
service now, APS integration in spec 2; recorded PBA games plus the replay rig as
development data; commentary is a required channel; ingest must accept any feed.

Open, not blocking M0:

1. Dataset licences for fine-tuning (Roboflow basketball sets are mixed; confirm
   before any commercial deployment).
2. Where the Linux GPU box lives for real-time runs.
3. Whether Cignal or the PBA will grant a contribution feed for production.
