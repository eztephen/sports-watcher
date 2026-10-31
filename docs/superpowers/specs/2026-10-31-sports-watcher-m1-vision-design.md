# Sports Watcher M1: vision events (addendum to the v1 design)

Date: 2026-10-31. Status: design approved in conversation, awaiting written review.
Extends `2026-10-31-sports-watcher-v1-design.md`. Where the two disagree, this
document wins for M1; section 2 lists every such place.

## 1. Purpose and scope

M1 turns the scoreboard-only watcher from M0 into one that sees the game: players,
ball, rim, referees, teams, court position and camera state, and from those the
basketball events in v1 section 5.6 that need vision: `possession_start`,
`shot_attempt`, `shot_made`, `shot_missed`, `free_throw_*`, `rebound`, `turnover`,
`timeout` and `tip_off`. The firsts read model gains `first_possession`,
`tip_off_winner`, `first_shot_attempt` and `possessions[]`.

Out of scope: commentary audio (M2), the replay rig latency report and Linux GPU build
(M3), jersey numbers, pose.

Development happens on an Apple M1 Pro with 16 GB and no NVIDIA GPU. Inference runs
through ONNX Runtime. Training runs on Google Colab's free tier, with the Mac (MPS) as
the fallback; the same scripts run on both.

## 2. What changes in the v1 spec

| v1 section | Change |
|---|---|
| 4 Architecture | Camera state is decided after perception and OCR, from the perception output and the scoreboard clock, not by a classifier ahead of perception. |
| 5.2 Perception | Modules revised (section 5 below): ball crop refinement, in-repo tracker, team k=2 with referees from the detector, court keypoints at 5 Hz, camera state from a SigLIP linear probe plus features. A `Perceiver` interface and a perception cache are added. |
| 5.4 Game state | Contents fixed in section 4 below. Scoreboard fields update on every frame; vision fields only on main-camera frames. |
| 5.5 Sport plug-in | The protocol as implemented in M0 is the binding one: `update_state(state, obs) -> GameState` and `rules()`, with a confirmer per draft. `court_template`, `detector_classes`, `cue_lexicon` and `fold_firsts` leave the protocol: the court template is the `LeagueConfig`, the class map is the detector manifest, and `fold_firsts` stays a function in `events/firsts.py`. |
| 5.6 Rules | Frame-count thresholds become seconds (the footage is 60 fps, the spec assumed 25 to 30). A clock gate is added. `shot_made` confirms on the scoreboard delta alone, since no commentary exists yet. |
| 5.7 Event model | `Evidence` gains `homography_confidence`, `clock_state` and `track_id`. `possessions[].result` never says `foul` in M1. |
| 7 Testing | The M1 golden set is two PBA clips (one for play events, one for camera state), not ten. A fourth tier, integration with real weights, is added. |
| 8 Layout | Adds `training/`, `data/cache/`, `scripts/perceive.py`, `scripts/download_models.py`. |
| 10 Open questions | Question 1 (dataset licences) is resolved for development: both Roboflow sets are CC BY 4.0. Commercial deployment still needs the attribution check. |

## 3. Pipeline

Per frame in `GameWatcher._process`, file sources still seeing every frame:

```
Frame
  ├─ every ocr_every_n_frames:   ScoreboardReader ─► ScoreboardState        (M0)
  ├─ every perceive_every_n_frames: Perceiver ─► VisionFrame                (new)
  ├─ CameraStateTracker(VisionFrame.camera_hint, hysteresis) ─► camera_state (new)
  ▼
Observation(frame_index, t_pts, t_wall, scoreboard, vision, camera_state)
  ▼
BasketballPlugin.update_state   scoreboard on every frame; vision only when main
  ▼
EventEngine: resolve pending ─► rules ─► provisional; sinks; fold_firsts       (M0)
```

`perceive_every_n_frames` defaults so that perception runs at about 30 Hz: 2 on a 60 fps
file, 1 at 30 fps. Between perception frames the vision state carries over, the ball filter
predicts. This keeps file runs deterministic and halves the model cost on the 60 fps
PBA footage.

### 3.1 Perceiver

```python
class Perceiver(Protocol):
    version: str                              # hashes weights and settings; keys the cache
    def perceive(self, frame: Frame) -> VisionFrame: ...
    def reset(self) -> None: ...              # after a reconnect or seek
```

Three implementations, injected into `GameWatcher` the way `text_reader` is:

- `OnnxPerceiver`: detector, ball crop, tracker, team assigner, court mapper, camera
  probe, all ONNX Runtime.
- `CachedPerceiver`: reads `data/cache/<clip-stem>.<version>.jsonl`, one `VisionFrame`
  per perception frame, matched by `frame_index`. On a miss it runs the wrapped
  perceiver and appends the result. File sources only.
- `ScriptedPerceiver`: a list of `VisionFrame`s for tests.

When `models/` lacks the weights, the watcher starts in scoreboard-only mode (no
perceiver), logs it once, and behaves exactly as M0.

## 4. Interfaces

```python
@dataclass(slots=True)
class Track:
    track_id: int
    box: tuple[float, float, float, float]    # x1, y1, x2, y2 pixels
    cls: Literal["player", "referee"]
    team: Literal["home", "away", "unknown"]
    feet_xy: tuple[float, float] | None       # court metres, None without a homography
    confidence: float

@dataclass(slots=True)
class BallObs:
    xy: tuple[float, float]                   # image pixels, centre
    radius_px: float
    floor_xy: tuple[float, float] | None      # the centre projected onto the floor plane
    source: Literal["detected", "predicted"]
    confidence: float

@dataclass(slots=True)
class Homography:
    matrix: np.ndarray                        # 3x3, image pixels -> court metres
    inliers: int
    reprojection_error_px: float
    confidence: float                         # 0..1 from inliers and error

@dataclass(slots=True)
class VisionFrame:
    frame_index: int
    t_pts: float
    camera_hint: Literal["main", "replay", "closeup", "graphics"]
    camera_confidence: float
    tracks: list[Track]
    ball: BallObs | None
    rim: tuple[float, float, float, float] | None
    homography: Homography | None
```

`Observation` gains `vision: VisionFrame | None`. `camera_state` keeps its M0 place on
the observation and is now filled by the camera-state tracker.

`GameState` gains:

```python
tracks: list[Track]
ball: BallObs | None
holder: Holder | None                 # track_id, team, since_pts
last_shot: ShotRecord | None          # t_pts, team, track_id, zone, outcome pending/made/missed
free_throws_pending: int
clock_running: bool
clock_state: Literal["read", "assumed", "unknown"]
last_clock_change_pts: float | None
```

`Evidence` gains `homography_confidence: float | None`, `clock_state: str | None` and
`track_id: int | None`.

`LeagueConfig` gains the court geometry the rules need, with FIBA values in `pba.yaml`:
`basket_from_baseline_m: 1.575`, `free_throw_line_from_baseline_m: 5.8`,
`key_width_m: 4.9`. `nba.yaml` gets its own values.

## 5. Perception modules (`sports_watcher/perception/`)

| Module | What it does | Dependencies |
|---|---|---|
| `detector.py` | RF-DETR fine-tuned on the Roboflow player set, classes collapsed to `player, ball, rim, referee`. Full-frame pass every perception frame. Default Small at 512; plan task 1 measures Nano and Small at 384, 512 and 640 on this Mac for speed and ball recall and fixes the choice in the model manifest. One rim is kept per camera segment: the most stable box. | onnxruntime |
| `ball.py` | A second detector pass on a 384-pixel crop around the ball filter's prediction at native resolution, every perception frame. Falls back to the full-frame ball when the prediction is lost. A 1080p ball is 15 to 25 pixels; at 512 input it is 4 to 7, too small for the full-frame pass alone. | onnxruntime |
| `tracker.py` | In-repo OC-SORT style tracker: one Kalman filter per player or referee track, IoU cost, Hungarian assignment, observation-centric re-update on re-association, max age 1 s, three hits before a track is published. Ball: a constant-velocity filter in image space, `predicted` when no detection matches, dropped after 0.5 s. The PyPI `ocsort` is a stale 2022 snapshot and boxmot is AGPL, hence in-repo. | filterpy (MIT), scipy (BSD) |
| `team.py` | SigLIP vision-tower embeddings (ONNX export of `google/siglip-base-patch16-224`) of player crops at track birth and once per second per track. KMeans k=2 on the first 200 crops, refit every 30 s; a track's team is the majority of its crop labels. Clusters map to home and away by the distance between each cluster's mean crop colour (Lab) and the operator's `--home-color` / `--away-color`. Referees come from the detector class. | scikit-learn (BSD) |
| `court.py` | RF-DETR Keypoint fine-tuned on the Roboflow court set (33 keypoints) at 5 Hz. Keypoints with confidence above 0.5 go to `cv2.findHomography` with RANSAC against the league template; the result is smoothed over the last second and held between keypoint frames. Confidence from inlier count and reprojection error; below 0.4 the homography is reported as `None`. Provides `image_to_court`, `zone_for(feet_xy, basket)` and `in_key(feet_xy)`. | opencv |
| `camera_state.py` | Two parts. `CameraProbe`: a linear probe over the SigLIP full-frame embedding at 5 Hz, four classes `main, replay, closeup, graphics`, trained on frames labelled from the clips on disk, blended with perception features (homography inliers, player count, median box height). `CameraStateTracker`: holds the last state, requires the new hint to persist for 0.5 s before switching. | onnxruntime, scikit-learn |

Replay frames look different from live play in this feed (low angle, top-down rim
camera, slow motion) and the probe learns those, but few examples exist. Event
correctness does not depend on the probe catching replays: the clock gate in section
6 is the real guard, and `replay` from the probe only adds an earlier freeze.

## 6. Basketball state and rules (`sports_watcher/sports/basketball/`)

### 6.1 Clock gate

`clock_running`, computed in `update_state` from confident scoreboard reads, is true when
the OCR clock value decreased within the last 2 s of pts (the clock shows whole seconds
and OCR runs at a few hertz, so 2 s survives one failed read). A vision rule drafts an event only while `clock_running`, with two
exceptions: `tip_off` (the clock is at full) and the free-throw rules (the clock is
stopped and a lane formation is present). Replays, timeouts, dead balls and the pause
before the tip all fail the gate.

When the scoreboard bug is hidden (no confident read for 2 s) while camera state is
main, the previous `clock_running` is carried for up to 5 s with
`clock_state = "assumed"`; events drafted under it carry that in `evidence.clock_state`
and lose 0.2 confidence. After 5 s the state is `unknown` and the gate is closed.

### 6.2 State updates

On a main-camera frame with vision: tracks, ball and homography are copied in; the
holder is the track whose box, grown by a quarter of its height on every side,
contains the ball centre for 0.17 s (five perception frames); `possession` follows the
holder's team. Off main: the holder is cleared, `possession` is kept, tracks and ball
are left as they were. On return to main a holder of the same team is not a new
possession.

### 6.3 Rules

All times are seconds of pts. Confirmers run on later observations, as in M0. Every
draft keeps the M0 TTL of 4 s and is retracted when it expires; the retract column
names the early case.

| Event | Draft | Confirm | Retract |
|---|---|---|---|
| `possession_start` | holder's team differs from `possession` | that team holds the ball for 0.5 s | the other team holds it first |
| `shot_attempt` | ball leaves the holder with upward image velocity, and a parabola fitted to the next four ball observations passes within 1.5 rim widths of the rim box inside 1.5 s; team and player are the last holder; zone from the shooter's feet through the homography (`two`, `three`, `unknown` below homography confidence 0.4) | ball enters the rim region, or a score delta of 2 or 3 arrives | neither within the TTL and the same team still holds the ball |
| `shot_made` | score delta of 2 or 3 within 3 s of an attempt (drafted with a confirmer that confirms on the next observation: the delta is already a voted read), or the ball centre crosses the rim box top to bottom inside its width (provisional) | the delta, for the vision-only draft | no delta by the TTL |
| `shot_missed` | ball leaves the rim region and 3 s pass without a delta | the next holder appears | a late delta |
| `rebound` | first holder after a miss; `offensive` when the holder's team shot | holder holds 0.5 s | holder flips |
| `turnover` | a confirmed `possession_start` with no `shot_attempt` since the previous one, or the shot clock resets to full without one | possession holds 0.5 s | an attempt is drafted retroactively |
| `timeout` | clock stopped and camera state off main for 20 s | clock still stopped at 30 s | clock runs again within 10 s |
| `tip_off` | period 1 or overtime, clock at full, two players' feet within 2 m of the centre circle and the ball rising from between them | a possession is established within 5 s (`team` stays `unknown`: a terminal copy never changes fields; the winner is read from the `possession_start` that follows) | no possession within 10 s |
| `free_throw_attempt` | clock stopped, at least four players' feet within 1 m of the key's side lines, one player holding the ball within 1 m of the free-throw line; drafted on release | the attempt resolves (below) | formation breaks before release |
| `free_throw_made` / `_missed` | delta of 1 within 3 s of the attempt makes it; otherwise missed at 3 s | made: the delta; missed: no delta by the TTL | missed: a late delta |

A score delta with no attempt inside 3 s emits the retroactive `shot_attempt` of v1
section 5.6 at confidence 0.4, zone from the delta (`ft`, `two`, `three`), followed for
a 2 or 3 by a `shot_made` at the same confidence.

Rules that need court positions (`tip_off`, free throws, zone) stay silent, or report
`unknown`, when the homography is `None`.

## 7. Firsts

`fold_firsts` gains, over confirmed events in `t_pts` order: `first_possession` from the
first `possession_start`; `tip_off_winner` from the first `possession_start` within 5 s
after a confirmed `tip_off`; `first_shot_attempt` from the first `shot_attempt`;
`possessions[]`, one entry per `possession_start` with `result` from what happened
before the next one: `score` (a made shot or free throw by that team), `turnover`,
`miss` (a `shot_missed` or `free_throw_missed` by that team), or `end_period`. A
possession with none of those and no later `possession_start` is left out. `foul`
needs commentary and does not appear in M1.

## 8. Failure behaviour

- Homography `None`: events fire with `court_xy` null and zone `unknown`; tip-off and
  free-throw rules stay silent.
- Track without a team: `team` is `unknown`; no `possession_start` is drafted for it;
  the firsts skip it.
- Ball lost for over 0.5 s: holder kept, no new drafts until the ball is seen again.
- Camera state off main: vision state frozen, scoreboard rules keep running.
- Weights missing: scoreboard-only mode, logged once.
- Every vision event carries `track_id`, `camera_state`, `clock_state`,
  `detector_confidence` and `homography_confidence`, so a wrong call can be checked
  against the cache line for that frame.

## 9. Development loop

- `scripts/perceive.py --source <clip> [--models models/]` builds the cache for a clip
  and prints per-stage timings. A 3-minute clip costs one run of about 15 to 30 minutes
  on this Mac; every later run of rules, engine, firsts and evaluation takes seconds.
- `scripts/run_watch.py` and the golden tests use `CachedPerceiver` when a cache for the
  clip and the current perceiver version exists.
- `scripts/download_models.py` fetches the weights and manifests from a GitHub release
  on this repository into `models/`; copying them in by hand is the fallback.
- `models/<name>/manifest.json` records model family and size, input resolution, class
  map, dataset name and version, training date, licence and the attribution line the
  dataset licence requires.

## 10. Training, datasets, licences

| Model | Data | Script | Output |
|---|---|---|---|
| Detector | Roboflow `roboflow-jvuqo/basketball-player-detection-2` (1,398 images, CC BY 4.0), classes remapped: `player` and every `player-*` action class to `player`; `ball`, `rim`, `referee` kept; `number` dropped | `training/train_detector.py` | `models/detector/model.onnx` + manifest |
| Court keypoints | Roboflow `roboflow-jvuqo/basketball-court-detection-2` v19 (850 images, 1 class, 33 keypoints, CC BY 4.0) | `training/train_court.py` | `models/court/model.onnx` + manifest |
| Camera probe | frames labelled from the three clips on disk, `data/labels/camera_state.jsonl` | `training/train_camera_probe.py` | `models/camera/probe.json` (weights of a linear layer) |
| SigLIP vision tower | `google/siglip-base-patch16-224` (Apache-2.0), no training | `training/export_siglip.py` | `models/siglip/vision.onnx` |

Training scripts run on Colab (free T4) and on the Mac with `--device mps`. They read
`ROBOFLOW_API_KEY` from the environment (`.env` locally). Training dependencies live in
a `train` extra (`rfdetr[train,onnx]`, torch, transformers, roboflow) and are never
imported at runtime. Only the Apache-2.0 RF-DETR sizes (Nano, Small, Medium) and
`RFDETRKeypointPreview` are used; Atto, Femto, Pico, XL and 2XL carry a different
licence and are off limits.

Runtime additions to `pyproject.toml`: `filterpy` (MIT), `scipy` (BSD),
`scikit-learn` (BSD). `onnxruntime` and `opencv-python-headless` are already present.

## 11. Testing

1. Unit, no models. `tests/helpers/vision.py` scripts a play in court metres (players'
   feet, ball path, release, rim arrival) and projects it through a fixed synthetic
   homography into `VisionFrame` sequences, so rules, zone lookup, clock gate, state
   updates and firsts folds run on deterministic fixtures. Tracker, ball filter, team
   assigner (synthetic embeddings), court mapper (synthetic keypoints with noise),
   camera-state tracker and the cache round trip get their own tests.
2. Integration, skipped without weights in `models/`: `OnnxPerceiver` on a dozen frames
   from the clips on disk, asserting players and the rim are found and the homography
   has confidence above 0.4 on a main-camera frame.
3. Golden, skipped without clips or cache:
   - `pba_ros_cvg_vision` (minutes 1:00 to 4:00 of `data/clips/pba_ros_cvg_2700_3100.mp4`,
     cut to `data/golden/`): precision and recall of 0.8 or better for `possession_start`
     and `shot_attempt`, `shot_made` consistent with the score changes, and no vision
     event during the free-throw graphic or the close-ups.
   - `pba_ros_cvg_q1` (the M0 clip): camera-state frame accuracy of 0.9 or better
     against `pba_ros_cvg_q1.camera.jsonl`, segments of `{"start", "end", "state"}`
     labelled by hand. Its `score_change` check from M0 stays.
   - Free throws and `tip_off` have unit coverage only until footage with a clear
     free-throw sequence and a Q1 opening is on disk.
4. Latency stays M3.

Truth files: `TruthEvent` gains an optional `zone`. I draft the vision truth from
contact sheets at four frames per second around each play; Stephen checks it against
the video before it is committed.

## 12. Decisions and open questions

Decided here: training on Colab free tier with the Mac as fallback; the two Roboflow
CC BY 4.0 datasets; ONNX Runtime for all inference; in-repo tracker; k=2 teams with
referees from the detector; camera state from a probe plus features with the clock
gate as the event guard; perception cache; thresholds in seconds; perception at 30 Hz.

Open:

1. Where the weights are hosted. Default: a GitHub release on this repository.
2. Footage for free throws and the tip-off golden checks (a Q1 opening segment of the
   same game, about 100 MB, not yet downloaded).
3. Commercial use of the CC BY 4.0 datasets: attribution is straightforward, but the
   provenance of the images inside the Roboflow sets has not been checked.
4. The Linux GPU box (v1 open question 2) stays open.
