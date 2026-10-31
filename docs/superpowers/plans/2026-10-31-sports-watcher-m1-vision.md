# Sports Watcher M1 Vision Events Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The watcher sees players, ball, rim, teams, court position and camera state, and emits provisional, confirmed and retracted `possession_start`, `shot_attempt`, `shot_made`, `shot_missed`, `free_throw_*`, `rebound`, `turnover`, `timeout` and `tip_off` events, with the firsts read model answering first possession, tip-off winner, first shot attempt and the possession list.

**Architecture:** A `Perceiver` turns each frame into a `VisionFrame` (tracks with teams and court feet positions, ball, rim, homography, camera hint); three implementations share the interface: ONNX models, a per-clip JSONL cache, and a scripted one for tests. A sport-agnostic camera-state tracker fills `Observation.camera_state`. The basketball plug-in's `update_state` derives geometry facts (clock gate, holder, release, rim interaction, lane formation) into `GameState`; rule classes draft events from those facts with confirmers that run on later observations, through the unchanged M0 engine. Training runs on Colab (Mac MPS fallback) and ships ONNX weights into gitignored `models/`.

**Tech Stack:** Python 3.12, onnxruntime (CoreML or CPU provider), filterpy, scipy, scikit-learn, OpenCV headless, NumPy, pydantic v2. Training extra: rfdetr[train,onnx], torch, transformers, roboflow.

**Spec:** `docs/superpowers/specs/2026-10-31-sports-watcher-m1-vision-design.md` (all sections) on top of `docs/superpowers/specs/2026-10-31-sports-watcher-v1-design.md`.

## Global Constraints

- Runtime dependencies Apache-2.0, MIT or BSD only. No Ultralytics, no boxmot, no GPL. Only RF-DETR Nano, Small, Medium and `RFDETRKeypointPreview` (Apache-2.0); never Atto, Femto, Pico, XL or 2XL.
- Events are append-only. Terminal copies keep `t_pts`/`t_wall`; only `emitted_wall` changes. Every draft keeps the M0 TTL of 4 s unless the draft sets `ttl_s`.
- File sources process every frame; perception runs on every `perceive_every_n_frames`-th frame (2 on 60 fps input, 1 on 30 fps). Only live sources drop frames.
- All thresholds in seconds of pts: hold 0.17 s, possession confirm 0.5 s, shot gate 1.5 rim widths within 1.5 s from four ball observations, score-delta window 3 s, clock gate 2 s, assumed clock at most 5 s with confidence minus 0.2, timeout 20 s draft / 30 s confirm / 10 s retract, tip-off 2 m / 5 s confirm / 10 s retract, free throws four players within 1 m of the lane edges and a shooter within 1 m of the line, 3 s to resolve.
- Perception constants: tracker max age 1 s, three hits to publish, ball dropped after 0.5 s; homography keypoint confidence 0.5, reported `None` below confidence 0.4, smoothed over 1 s; team fit after 200 crops, refit every 30 s, one crop per second per track; keypoints and camera probe at 5 Hz; camera hysteresis 0.5 s.
- League geometry: PBA (FIBA) basket 1.575 m from the baseline, free-throw line 5.8 m, key width 4.9 m; NBA basket 1.60 m, line 5.79 m, key 4.88 m.
- Weights in `models/`, footage in `data/`, cache in `data/cache/`, labels in `data/labels/`; all gitignored except `data/golden/*.truth.jsonl`, `*.meta.json`, `*.camera.jsonl` and `README.md`. Secrets (`ROBOFLOW_API_KEY`) in `.env` only.
- Every commit leaves `./.venv/bin/python -m pytest -q` green and `./.venv/bin/ruff check .` clean. Use `./.venv/bin/python` for every command. Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Work on branch `feat/m1-vision` in this checkout (M0 ruling: feature branch, not a worktree).

## Review Focus

1. A replay of a made shot (wide frame, game clock frozen) must draft no `possession_start` and no `shot_attempt`. Pinned in Task 6 (`test_clock_gate_closes_two_seconds_after_the_last_clock_change`) and Task 8 (`test_shot_during_frozen_clock_is_not_drafted`).
2. A chest pass between teammates whose arc never nears the rim must not become a `shot_attempt`, and a lob that does pass the rim gate but ends in a teammate's hands must be retracted early. Pinned in Task 8 (`test_pass_away_from_rim_is_not_a_shot`, `test_lob_caught_by_teammate_retracts_attempt`).
3. The scoreboard bug hidden by an ad overlay during live play: events keep flowing for 5 s with `evidence.clock_state == "assumed"` and confidence lowered by 0.2, then stop. Pinned in Task 6 (`test_hidden_bug_assumes_running_for_five_seconds`) and Task 7 (`test_assumed_clock_lowers_confidence`).
4. Returning from a close-up with the same team still holding the ball must not emit a `possession_start`; a different team must. Pinned in Task 7 (`test_same_team_after_closeup_is_not_a_new_possession`).
5. A score delta 2.5 s after an attempt whose ball path was lost confirms `shot_made` and emits no duplicate retroactive attempt; a delta with no attempt in 3 s emits the retroactive attempt at confidence 0.4. Pinned in Task 8 (`test_delta_after_lost_ball_confirms_made_without_duplicate`, `test_delta_without_attempt_emits_retroactive_attempt`).

External dependency: training needs a Roboflow API key (free account) and a Colab session. Task 3 lands the scripts first so training runs while Tasks 4 to 16 are built; Tasks 17 and 18 need the weights.

---

### Task 1: Branch, dependencies and league geometry

**Files:**
- Modify: `pyproject.toml`, `.gitignore`, `.env.example`
- Modify: `sports_watcher/config.py`, `configs/leagues/pba.yaml`, `configs/leagues/nba.yaml`, `tests/fixtures/configs/leagues/pba.yaml`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `LeagueConfig` gains `basket_from_baseline_m: float`, `free_throw_line_from_baseline_m: float`, `key_width_m: float`. `Settings` gains `models_dir: Path = Path("models")` and `cache_dir: Path = Path("data/cache")`.

- [ ] **Step 1: Create the branch**

```bash
git checkout -b feat/m1-vision
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit/test_config.py (append)
def test_league_geometry_fields():
    pba = load_league("pba")
    assert (pba.basket_from_baseline_m, pba.free_throw_line_from_baseline_m, pba.key_width_m) == (1.575, 5.8, 4.9)
    nba = load_league("nba")
    assert (nba.basket_from_baseline_m, nba.free_throw_line_from_baseline_m, nba.key_width_m) == (1.60, 5.79, 4.88)

def test_settings_model_and_cache_dirs():
    s = Settings(_env_file=None)
    assert s.models_dir == Path("models") and s.cache_dir == Path("data/cache")
```

- [ ] **Step 3: Run them to verify they fail**

Run: `./.venv/bin/python -m pytest tests/unit/test_config.py -q`
Expected: FAIL, `AttributeError` or validation error on the new fields.

- [ ] **Step 4: Add the fields**

Add the three geometry fields to `LeagueConfig` (required, no defaults) and to all three league yaml files; add `models_dir` and `cache_dir` to `Settings`.

- [ ] **Step 5: Add dependencies and ignores**

`pyproject.toml`: runtime `filterpy>=1.4.5`, `scipy>=1.13`, `scikit-learn>=1.5`, `onnxruntime>=1.18`; new extra `train = ["rfdetr[train,onnx]>=1.11", "torch>=2.4", "transformers>=4.44", "roboflow>=1.1"]`. `.gitignore`: add `!data/golden/*.camera.jsonl`. `.env.example`: add `ROBOFLOW_API_KEY=`. Then `./.venv/bin/pip install -e ".[dev,youtube]"`.

- [ ] **Step 6: Run the suite and lint**

Run: `./.venv/bin/python -m pytest tests/unit -q && ./.venv/bin/ruff check .`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore .env.example sports_watcher/config.py configs tests
git commit -m "feat(m1): league geometry, model and cache dirs, runtime deps for vision"
```

---

### Task 2: Vision types and the state, observation and evidence extensions

**Files:**
- Create: `sports_watcher/perception/types.py`
- Modify: `sports_watcher/state/game_state.py`, `sports_watcher/sports/base.py`, `sports_watcher/events/model.py`
- Test: `tests/unit/test_vision_types.py`

**Interfaces:**
- Produces (`perception/types.py`): `Box = tuple[float, float, float, float]` (x1, y1, x2, y2 pixels); `Team = Literal["home", "away", "unknown"]`; `CameraHint = Literal["main", "replay", "closeup", "graphics"]`; dataclasses (slots) `Track(track_id: int, box: Box, cls: Literal["player", "referee"], team: Team, feet_xy: tuple[float, float] | None, confidence: float)`, `BallObs(xy: tuple[float, float], radius_px: float, floor_xy: tuple[float, float] | None, source: Literal["detected", "predicted"], confidence: float)`, `Homography(matrix: np.ndarray, inliers: int, reprojection_error_px: float, confidence: float)`, `VisionFrame(frame_index: int, t_pts: float, camera_hint: CameraHint, camera_confidence: float, tracks: list[Track], ball: BallObs | None, rim: Box | None, homography: Homography | None)`; `empty_vision(frame_index: int, t_pts: float) -> VisionFrame` (hint `main`, confidence 0, no tracks); `vision_to_dict(vf: VisionFrame) -> dict` and `vision_from_dict(d: dict) -> VisionFrame` (matrix as nine floats); `box_center(box) -> tuple[float, float]`, `box_height(box) -> float`, `feet_of(box) -> tuple[float, float]` (bottom centre).
- Produces (`state/game_state.py`): dataclasses `Holder(track_id: int, team: Team, since_pts: float)`, `Release(t_pts: float, track_id: int, team: Team, image_xy: tuple[float, float], feet_xy: tuple[float, float] | None, zone: str)`, `RimInteraction(entered_pts: float | None = None, left_pts: float | None = None, crossed_pts: float | None = None)`, `ShotRecord(t_pts: float, track_id: int, team: Team, zone: str, court_xy: tuple[float, float] | None)`, `LaneFormation(shooter_track_id: int, team: Team, side: Literal["left", "right"], since_pts: float)`. `GameState` gains `tracks: list[Track] = []`, `ball: BallObs | None`, `rim: Box | None`, `homography_confidence: float = 0.0`, `holder: Holder | None`, `holder_candidate: tuple[int, float] | None` (track id, since pts), `last_release: Release | None`, `ball_path: list[tuple[float, float, float]] = []` (t_pts, x, y since release), `rim_interaction: RimInteraction | None`, `last_shot: ShotRecord | None`, `lane_formation: LaneFormation | None`, `free_throws_pending: int = 0`, `possession_since_pts: float | None`, `clock_running: bool = False`, `clock_state: Literal["read", "assumed", "unknown"] = "unknown"`, `last_clock_change_pts: float | None`, `last_scoreboard_pts: float | None`, `camera_left_main_pts: float | None`. Lists use `field(default_factory=list)`.
- Produces: `Observation.vision: VisionFrame | None = None`. `Evidence` gains `homography_confidence: float | None`, `clock_state: str | None`, `track_id: int | None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_vision_types.py
import numpy as np
from sports_watcher.perception.types import (BallObs, Homography, Track, VisionFrame,
    empty_vision, feet_of, vision_from_dict, vision_to_dict)

def _frame():
    return VisionFrame(frame_index=7, t_pts=0.233, camera_hint="main", camera_confidence=0.8,
        tracks=[Track(3, (10.0, 20.0, 50.0, 120.0), "player", "home", (4.0, 7.5), 0.9)],
        ball=BallObs((30.0, 15.0), 6.0, (4.1, 7.4), "detected", 0.7),
        rim=(900.0, 300.0, 940.0, 320.0),
        homography=Homography(np.eye(3), 12, 2.5, 0.85))

def test_round_trip_through_dict():
    vf = _frame()
    back = vision_from_dict(vision_to_dict(vf))
    assert back.tracks == vf.tracks and back.ball == vf.ball and back.rim == vf.rim
    assert back.homography.inliers == 12 and np.allclose(back.homography.matrix, np.eye(3))
    assert back.camera_hint == "main" and back.frame_index == 7

def test_empty_vision_has_no_detections():
    vf = empty_vision(1, 0.0)
    assert vf.tracks == [] and vf.ball is None and vf.homography is None and vf.camera_hint == "main"

def test_feet_of_is_bottom_centre():
    assert feet_of((10.0, 20.0, 50.0, 120.0)) == (30.0, 120.0)

def test_game_state_and_observation_defaults():
    from sports_watcher.sports.base import Observation
    from sports_watcher.state.game_state import GameState
    from datetime import UTC, datetime
    s = GameState()
    assert s.holder is None and s.clock_running is False and s.clock_state == "unknown" and s.tracks == []
    obs = Observation(frame_index=0, t_pts=0.0, t_wall=datetime.now(UTC))
    assert obs.vision is None

def test_evidence_vision_fields_default_none():
    from sports_watcher.events.model import Evidence
    e = Evidence()
    assert e.homography_confidence is None and e.clock_state is None and e.track_id is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `./.venv/bin/python -m pytest tests/unit/test_vision_types.py -q`
Expected: FAIL with `ModuleNotFoundError: sports_watcher.perception.types`.

- [ ] **Step 3: Implement the types and the extensions**

`vision_to_dict` writes dataclasses as plain dicts, tuples as lists, the matrix as a flat list of nine floats; `vision_from_dict` reverses it and restores tuples. `GameState` keeps `dataclasses.replace` working (all new fields have defaults).

- [ ] **Step 4: Run the whole unit suite**

Run: `./.venv/bin/python -m pytest tests/unit -q && ./.venv/bin/ruff check .`
Expected: all pass (M0 tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add sports_watcher tests/unit/test_vision_types.py
git commit -m "feat(m1): vision types; state, observation and evidence fields for vision"
```

---

### Task 3: Model manifests and the training scripts

**Files:**
- Create: `sports_watcher/perception/models.py`
- Create: `training/__init__.py`, `training/README.md`, `training/datasets.py`, `training/manifest.py`, `training/train_detector.py`, `training/train_court.py`, `training/export_siglip.py`
- Test: `tests/unit/test_models_manifest.py`, `tests/unit/test_training_datasets.py`

**Interfaces:**
- Produces (`perception/models.py`): `ModelManifest(BaseModel)`: `name: str`, `family: str` (`rfdetr`, `rfdetr-keypoint`, `siglip`, `linear-probe`), `size: str | None`, `resolution: int | None`, `classes: dict[int, str] = {}`, `keypoints: list[str] = []`, `dataset: str | None`, `dataset_version: str | None`, `trained_on: str | None`, `licence: str`, `attribution: str | None`, `sha256: str`. `load_manifest(path: Path) -> ModelManifest`. `ModelSet` dataclass: `detector: Path`, `detector_manifest: ModelManifest`, `court: Path | None`, `court_manifest: ModelManifest | None`, `siglip: Path | None`, `camera_probe: Path | None`; `ModelSet.load(models_dir: Path) -> ModelSet | None` (layout `models/detector/model.onnx` + `manifest.json`, `models/court/model.onnx` + `manifest.json`, `models/siglip/vision.onnx`, `models/camera/probe.json`; `None` when the detector is missing, optional parts `None` when absent); `ModelSet.version -> str` (first 12 hex of sha256 over the present manifests' `sha256` values and file names, in a fixed order).
- Produces (`training/datasets.py`): `PLAYER_CLASS_MAP: dict[str, str | None] = {"ball": "ball", "player": "player", "referee": "referee", "rim": "rim", "number": None, "player-dribble": "player", "player-fall": "player", "player-jump-shot": "player", "player-layup": "player", "player-screen": "player", "player-shot-block": "player"}`; `TARGET_CLASSES = ["ball", "player", "referee", "rim"]` (ids 1 to 4 in this order); `remap_coco(src: Path, dst: Path, class_map: dict[str, str | None], target_classes: list[str]) -> dict[str, int]` (returns annotation counts per target class; drops annotations of `None` classes; rewrites `categories` to the targets with ids 1..n; keeps images); `download_roboflow(workspace: str, project: str, version: int, fmt: str, dest: Path) -> Path` (roboflow SDK, key from `ROBOFLOW_API_KEY`); `remap_split_dirs(dataset_dir: Path, ...)` applies `remap_coco` to `train/`, `valid/`, `test/` `_annotations.coco.json`.
- Produces (`training/manifest.py`): `sha256_of(path: Path) -> str`; `write_manifest(model_path: Path, **fields) -> Path` writes `manifest.json` beside the model with `sha256` filled.
- Produces: `training/train_detector.py --dataset-dir --model {nano,small,medium} --resolution --epochs --batch --grad-accum --device {cuda,mps,cpu} --out models/detector`; `training/train_court.py --dataset-dir --resolution 576 --epochs --device --out models/court`; `training/export_siglip.py --out models/siglip/vision.onnx`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_models_manifest.py
import json
from pathlib import Path
from sports_watcher.perception.models import ModelManifest, ModelSet, load_manifest

def _write(dir: Path, name: str, sha: str, **extra):
    d = dir / name; d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"x")
    (d / "manifest.json").write_text(json.dumps({"name": name, "family": "rfdetr", "licence": "Apache-2.0", "sha256": sha, **extra}))

def test_model_set_requires_detector(tmp_path):
    assert ModelSet.load(tmp_path) is None

def test_model_set_loads_optional_parts(tmp_path):
    _write(tmp_path, "detector", "a" * 64, classes={"1": "ball", "2": "player", "3": "referee", "4": "rim"}, resolution=512)
    ms = ModelSet.load(tmp_path)
    assert ms is not None and ms.court is None and ms.siglip is None and ms.camera_probe is None
    assert ms.detector_manifest.classes[2] == "player"
    _write(tmp_path, "court", "b" * 64, family="rfdetr-keypoint", keypoints=[f"k{i}" for i in range(33)])
    ms2 = ModelSet.load(tmp_path)
    assert ms2.court_manifest.keypoints[32] == "k32" and ms2.version != ms.version and len(ms2.version) == 12

def test_load_manifest_validates(tmp_path):
    p = tmp_path / "manifest.json"; p.write_text(json.dumps({"name": "x", "family": "siglip", "licence": "Apache-2.0", "sha256": "c" * 64}))
    assert load_manifest(p).family == "siglip"
```

```python
# tests/unit/test_training_datasets.py
import json
from training.datasets import PLAYER_CLASS_MAP, TARGET_CLASSES, remap_coco

def test_remap_merges_player_actions_and_drops_numbers(tmp_path):
    src = tmp_path / "a.json"
    src.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "f.jpg", "width": 10, "height": 10}],
        "categories": [{"id": 0, "name": "basketball", "supercategory": "none"}, {"id": 1, "name": "ball"}, {"id": 2, "name": "number"},
                       {"id": 3, "name": "player-layup"}, {"id": 4, "name": "rim"}, {"id": 5, "name": "referee"}, {"id": 6, "name": "player"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0},
                        {"id": 2, "image_id": 1, "category_id": 2, "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0},
                        {"id": 3, "image_id": 1, "category_id": 3, "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0},
                        {"id": 4, "image_id": 1, "category_id": 6, "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0}]}))
    dst = tmp_path / "b.json"
    counts = remap_coco(src, dst, PLAYER_CLASS_MAP, TARGET_CLASSES)
    out = json.loads(dst.read_text())
    assert [c["name"] for c in out["categories"]] == ["ball", "player", "referee", "rim"]
    assert [c["id"] for c in out["categories"]] == [1, 2, 3, 4]
    assert counts == {"ball": 1, "player": 2, "referee": 0, "rim": 0}
    assert sorted(a["category_id"] for a in out["annotations"]) == [1, 2, 2]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `./.venv/bin/python -m pytest tests/unit/test_models_manifest.py tests/unit/test_training_datasets.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `perception/models.py`, `training/datasets.py`, `training/manifest.py`**

`training/` is a plain package importable from the repo root (pytest `rootdir`); it must not import torch or rfdetr at module import time, only inside `main()` functions. Add `training` to ruff's checked paths (it is by default) and to `[tool.setuptools.packages.find] include` only if packaging it is wanted (it is not: leave it out).

- [ ] **Step 4: Write the training scripts**

`train_detector.py`: map `--model` to `RFDETRNano`, `RFDETRSmall`, `RFDETRMedium` (import inside `main`); `model = cls(resolution=args.resolution)`; `model.train(dataset_dir=str(args.dataset_dir), epochs=args.epochs, batch_size=args.batch, grad_accum_steps=args.grad_accum, device=args.device, output_dir=str(args.out / "run"))`; then `model.export(output_dir=str(args.out), format="onnx", shape=(args.resolution, args.resolution), batch_size=1, opset_version=17)`, rename the exported file to `model.onnx`, and `write_manifest(args.out / "model.onnx", name="detector", family="rfdetr", size=args.model, resolution=args.resolution, classes={i + 1: n for i, n in enumerate(TARGET_CLASSES)}, dataset="roboflow-jvuqo/basketball-player-detection-2", dataset_version=str(args.dataset_version), trained_on=date.today().isoformat(), licence="Apache-2.0 (model); CC BY 4.0 (data)", attribution="Roboflow Universe basketball-player-detection-2, CC BY 4.0")`. The resolution must be divisible by 32. Check the exported ONNX output names (`dets`, `labels`) and record them in the manifest under `name`-level fields `output_boxes` and `output_logits` if they differ; Task 15 reads those two optional fields with those defaults.

`train_court.py`: `RFDETRKeypointPreview(num_classes=1, num_keypoints_per_class=[33], resolution=args.resolution)` with `KeypointTrainConfig(dataset_file="roboflow", dataset_dir=...)` as the rfdetr 1.11 API requires; export ONNX the same way; manifest `family="rfdetr-keypoint"`, `keypoints` = the 33 names from the dataset's COCO `categories[0]["keypoints"]` list in order.

`export_siglip.py`: `SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")`, `torch.onnx.export` with input `pixel_values` `[1, 3, 224, 224]`, output `pooler_output`, opset 17; manifest `family="siglip"`, `licence="Apache-2.0"`.

`training/README.md`: the Colab recipe (new notebook, GPU runtime, `!pip install "rfdetr[train,onnx]" roboflow`, upload or `git clone` this repo, `%env ROBOFLOW_API_KEY=...`, `!python training/train_detector.py --dataset-dir ... --model small --resolution 512 --epochs 30 --batch 8 --grad-accum 2 --device cuda --out models/detector`, same for court, zip `models/` and download), the Mac variant (`--device mps`, `--batch 2 --grad-accum 8`), dataset download lines (`download_roboflow("roboflow-jvuqo", "basketball-player-detection-2", <version>, "coco", Path("data/datasets/players"))` then `remap_split_dirs`), and the licence note (CC BY 4.0 attribution text that the manifest carries).

- [ ] **Step 5: Run tests and lint**

Run: `./.venv/bin/python -m pytest tests/unit -q && ./.venv/bin/ruff check .`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add sports_watcher/perception/models.py training tests/unit/test_models_manifest.py tests/unit/test_training_datasets.py
git commit -m "feat(m1): model manifests and Colab training scripts for detector, court keypoints and SigLIP export"
```

After this commit: tell Stephen the Colab recipe is ready and needs his Roboflow key; training runs while the next tasks are built.

---

### Task 4: Perceiver interface, scripted perceiver and the cache

**Files:**
- Create: `sports_watcher/perception/perceiver.py`, `sports_watcher/perception/cache.py`
- Test: `tests/unit/test_perceiver_cache.py`

**Interfaces:**
- Produces (`perceiver.py`): `class Perceiver(Protocol): version: str; def perceive(self, frame: Frame) -> VisionFrame; def reset(self) -> None`. `ScriptedPerceiver(frames: Sequence[VisionFrame], version: str = "scripted")`: `perceive` returns the scripted frame whose `frame_index` equals `frame.index`, else `empty_vision(frame.index, frame.pts_s)`; `calls: list[int]` records indices.
- Produces (`cache.py`): `cache_path(cache_dir: Path, clip: Path, version: str) -> Path` = `cache_dir / f"{clip.stem}.{version}.jsonl"`; `find_cache(cache_dir: Path, clip: Path) -> Path | None` (newest matching `f"{clip.stem}.*.jsonl"`); `class CacheMiss(KeyError)`; `CachedPerceiver(inner: Perceiver | None, path: Path)`: header line `{"version": ..., "clip": ...}` written on first append; `version` is `inner.version` or the header's; `perceive` returns the cached frame for `frame.index`, else runs `inner` and appends (flush per line), or raises `CacheMiss` when `inner is None`; `hits: int`, `misses: int`; `reset()` forwards to `inner`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_perceiver_cache.py
from datetime import UTC, datetime
import numpy as np
from sports_watcher.ingest.base import Frame
from sports_watcher.perception.cache import CachedPerceiver, CacheMiss, cache_path, find_cache
from sports_watcher.perception.perceiver import ScriptedPerceiver
from sports_watcher.perception.types import Track, VisionFrame, empty_vision

def _frame(i):
    return Frame(index=i, pts_s=i / 30, wall=datetime(2026, 1, 1, tzinfo=UTC), image=np.zeros((4, 4, 3), np.uint8))

def _vf(i):
    return VisionFrame(i, i / 30, "main", 0.9, [Track(1, (0.0, 0.0, 10.0, 20.0), "player", "home", None, 0.9)], None, None, None)

def test_scripted_perceiver_returns_frame_by_index_or_empty():
    p = ScriptedPerceiver([_vf(0), _vf(2)])
    assert p.perceive(_frame(2)).tracks[0].track_id == 1
    assert p.perceive(_frame(1)) == empty_vision(1, 1 / 30)
    assert p.calls == [2, 1]

def test_cache_path_and_find(tmp_path):
    clip = tmp_path / "clip.mp4"
    p = cache_path(tmp_path, clip, "abc123")
    assert p.name == "clip.abc123.jsonl"
    assert find_cache(tmp_path, clip) is None
    p.write_text("{}\n")
    assert find_cache(tmp_path, clip) == p

def test_cached_perceiver_writes_through_then_hits(tmp_path):
    inner = ScriptedPerceiver([_vf(0), _vf(1)], version="v1")
    path = cache_path(tmp_path, tmp_path / "c.mp4", inner.version)
    c = CachedPerceiver(inner, path)
    assert c.perceive(_frame(0)).tracks[0].team == "home" and c.misses == 1
    assert c.perceive(_frame(0)).tracks[0].team == "home" and c.hits == 1 and inner.calls == [0]
    reopened = CachedPerceiver(None, path)
    assert reopened.version == "v1" and reopened.perceive(_frame(0)) == _vf(0)
    import pytest
    with pytest.raises(CacheMiss):
        reopened.perceive(_frame(5))
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/bin/python -m pytest tests/unit/test_perceiver_cache.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement both modules**

The cache reads the whole file into a dict at open (a 3-minute clip is about 5,400 lines). Lines are `vision_to_dict` JSON.

- [ ] **Step 4: Run tests and lint, then commit**

```bash
git add sports_watcher/perception/perceiver.py sports_watcher/perception/cache.py tests/unit/test_perceiver_cache.py
git commit -m "feat(m1): Perceiver interface, scripted perceiver and per-clip perception cache"
```

---

### Task 5: Basketball court geometry

**Files:**
- Create: `sports_watcher/sports/basketball/geometry.py`
- Test: `tests/unit/test_basketball_geometry.py`

**Interfaces:**
- Produces: court frame: x along the length from the left baseline (0 to `court_length_m`), y along the width from the near sideline (0 to `court_width_m`); `Side = Literal["left", "right"]`. `basket_xy(league, side) -> tuple[float, float]` (left: `(basket_from_baseline_m, width/2)`, right: `(length - basket_from_baseline_m, width/2)`); `free_throw_xy(league, side)`; `centre_xy(league)`; `nearest_side(court_xy, league) -> Side`; `zone_for(court_xy, league, side) -> Literal["two", "three"]`: `three` when the distance to the basket is at least `three_point_radius_m`, except inside the corner strip (|y - width/2| > `three_point_corner_m` is impossible, so: a point is in the corner strip when its x is between the baseline and the basket x plus 1.575 m... see step 3) ; `in_lane_edges(court_xy, league, side, margin_m=1.0) -> bool` (within `margin_m` of either long edge of the key, between the baseline and the free-throw line); `near(a, b, radius_m) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_basketball_geometry.py
from sports_watcher.sports.basketball.geometry import (basket_xy, centre_xy, free_throw_xy,
    in_lane_edges, nearest_side, zone_for)
from tests.helpers.leagues import fixture_league

L = fixture_league("pba")

def test_fixed_points():
    assert basket_xy(L, "left") == (1.575, 7.5) and basket_xy(L, "right") == (26.425, 7.5)
    assert free_throw_xy(L, "left") == (5.8, 7.5) and centre_xy(L) == (14.0, 7.5)
    assert nearest_side((3.0, 2.0), L) == "left" and nearest_side((20.0, 2.0), L) == "right"

def test_zones():
    assert zone_for((4.0, 7.5), L, "left") == "two"           # 2.4 m from the basket
    assert zone_for((9.0, 7.5), L, "left") == "three"         # 7.4 m straight on
    assert zone_for((8.0, 7.5), L, "left") == "two"           # 6.4 m, inside the 6.75 arc
    assert zone_for((1.575, 0.5), L, "left") == "three"       # corner: 7.0 m sideways, past 6.60
    assert zone_for((1.575, 1.5), L, "left") == "two"         # corner: 6.0 m sideways

def test_lane_edges():
    assert in_lane_edges((3.0, 7.5 + 2.45), L, "left")          # on the edge
    assert in_lane_edges((3.0, 7.5 + 2.45 + 0.9), L, "left")    # 0.9 m outside
    assert not in_lane_edges((3.0, 7.5), L, "left")             # middle of the key
    assert not in_lane_edges((7.0, 7.5 + 2.45), L, "left")      # past the free-throw line
```

- [ ] **Step 2: Run to verify failure, implement, run, commit**

`zone_for`: compute `d = distance to basket_xy(side)`. The corner rule: when the point's x lies between the baseline and the basket's x plus the distance where the arc meets the corner line (FIBA: the straight corner segment runs from the baseline to 2.99 m along the sideline; use `x` within `basket_from_baseline_m + 1.425` of the baseline), the zone is `three` when `|y - width/2| >= three_point_corner_m`, else `two`; otherwise `three` when `d >= three_point_radius_m`. Commit: `feat(m1): basketball court geometry (baskets, zones, lane edges)`.

---

### Task 6: Synthetic play builder and the vision state update

**Files:**
- Create: `tests/helpers/vision.py`, `sports_watcher/sports/basketball/vision_state.py`
- Modify: `sports_watcher/sports/basketball/plugin.py`
- Test: `tests/unit/test_basketball_vision_state.py`

**Interfaces:**
- Produces (`tests/helpers/vision.py`): `Scene(league: LeagueConfig)` with a fixed side-camera homography built by `cv2.getPerspectiveTransform` from court corners `(0,0), (L,0), (L,W), (0,W)` to pixels `(100,1000), (1820,1000), (1500,350), (420,350)`; `project(court_xy, height_m=0.0) -> tuple[float, float]` (vertical offset `height_m * ppm(court_xy)` where `ppm` is the pixel length of a 1 m step along x at that point); `player(track_id, team, feet_xy, cls="player") -> Track` (box 0.6 m wide, 1.9 m tall in pixels, `feet_xy` set); `ball(floor_xy, height_m, source="detected") -> BallObs` (radius 0.12 m in pixels, `floor_xy` set); `rim(side) -> Box` (0.45 m wide at 3.05 m above `basket_xy`); `homography(confidence=0.9) -> Homography` (image to court, inliers 20, error 2.0). `PlayBuilder(scene, fps=30.0, start_pts=0.0)`: `frame(players: dict[int, tuple[Team, tuple[float, float]]], ball: tuple[tuple[float, float], float] | None, *, rim_side="right", homography=True, camera_hint="main") -> VisionFrame` appends one frame and advances `t` by `1/fps`; `hold(seconds, players, holder_id)` puts the ball at the holder's feet at 1.0 m height for that long; `shot(shooter_id, players, rim_side, flight_s=1.0, outcome="make" | "miss")`: a parabola in court space from the shooter's position at 2.0 m to the rim centre (`make`: through the rim, continuing 0.5 m below; `miss`: ends 0.3 m beside the rim at rim height then falls 1.5 m); `lob(shooter_id, catcher_id, players, rim_side, flight_s=0.8)`: the first 60 % of a rim-bound arc, then the ball at the catcher's hands (an alley-oop the rim never sees); `pass_(from_id, to_id, players, flight_s=0.5)`: straight at 1.5 m; `frames: list[VisionFrame]`. `observations(frames, *, clock_start_s=600.0, running=True, resume_at: float | None = None, ocr_every_s=0.25, period=1, score=(0, 0), deltas: dict[float, tuple[int, int]] | None = None, hidden: list[tuple[float, float]] | None = None) -> list[Observation]`: a confident `ScoreboardState` every `ocr_every_s` (clock decreasing when `running`, frozen otherwise; `resume_at` makes a frozen clock start running at that pts; `deltas` sets the score from that pts on; `hidden` intervals yield no scoreboard), `camera_state` from each frame's `camera_hint`. `drive(observations, league=None) -> list[Event]`: a fresh `BasketballPlugin`, `EventEngine` with a `MemorySink`, `step` each observation, `finish()`, return all events. `confirmed(events, type) -> list[Event]`, `retracted(events, type)`, `provisional(events, type)`.
- Produces (`vision_state.py`): constants `HOLD_S = 0.17`, `BOX_GROW = 0.25`, `CLOCK_RUNNING_S = 2.0`, `ASSUMED_MAX_S = 5.0`, `RIM_REGION_WIDTHS = 2.0`, `SHOT_GATE_WIDTHS = 1.5`, `SHOT_FIT_OBS = 4`, `SHOT_WINDOW_S = 1.5`, `BALL_PATH_MAX_S = 1.5`, `LANE_MIN_PLAYERS = 4`, `LANE_MARGIN_M = 1.0`, `FT_LINE_RADIUS_M = 1.0`, `MIN_HOMOGRAPHY_CONF = 0.4`. Functions, all mutating the new state in place: `update_clock_gate(prev: GameState, new: GameState, obs: Observation) -> None` (sets `last_scoreboard_pts`, `last_clock_change_pts`, `clock_running`, `clock_state` per spec 6.1); `apply_vision(prev, new, obs, league) -> None` (copies tracks, ball, rim, homography confidence; holder candidate and holder; `possession` and `possession_since_pts`; release detection; ball path; rim interaction; `last_shot` when `shot_path_passes_rim`; lane formation); `freeze_vision(new, obs) -> None` (off main: clears holder and candidate, sets `camera_left_main_pts` once). Pure helpers: `holder_candidate(tracks, ball) -> int | None`; `grown(box, factor) -> Box`; `shot_path_passes_rim(path: list[tuple[float, float, float]], rim: Box) -> bool` (fit x linear and y quadratic in t by least squares over the first `SHOT_FIT_OBS` points, evaluate at 60 steps to `SHOT_WINDOW_S`, pass when the minimum distance to the rim centre is at most `SHOT_GATE_WIDTHS` rim widths); `rim_region(rim) -> Box` (the rim box grown to `RIM_REGION_WIDTHS` widths square around its centre); `crossed_rim(prev_xy, xy, rim) -> bool` (y goes from above the rim top to below the rim bottom with x inside the rim span); `lane_formation(tracks, holder, league) -> LaneFormation | None`.
- Modify `BasketballPlugin.update_state`: after the M0 scoreboard logic, call `update_clock_gate`, then `apply_vision` when `obs.vision is not None and obs.camera_state == "main"`, else `freeze_vision` when `obs.camera_state != "main"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_basketball_vision_state.py
from sports_watcher.sports.basketball.plugin import BasketballPlugin
from sports_watcher.state.game_state import GameState
from tests.helpers.leagues import fixture_league
from tests.helpers.vision import PlayBuilder, Scene, observations

L = fixture_league("pba")
HOME_A, AWAY_B = (10.0, 7.0), (12.0, 8.0)

def _states(obs):
    plugin, state, out = BasketballPlugin(L), GameState(), []
    for o in obs:
        state = plugin.update_state(state, o)
        out.append(state)
    return out

def test_holder_needs_hold_time():
    b = PlayBuilder(Scene(L))
    players = {1: ("home", HOME_A), 2: ("away", AWAY_B)}
    b.hold(0.10, players, holder_id=1)          # 3 frames
    states = _states(observations(b.frames))
    assert states[-1].holder is None and states[-1].holder_candidate == (1, 0.0)
    b.hold(0.10, players, holder_id=1)          # 6 frames in total, 0.17 s reached
    states = _states(observations(b.frames))
    assert states[-1].holder is not None and states[-1].holder.track_id == 1
    assert states[-1].possession == "home" and states[-1].possession_since_pts == 0.0

def test_clock_gate_closes_two_seconds_after_the_last_clock_change():
    b = PlayBuilder(Scene(L)); players = {1: ("home", HOME_A), 2: ("away", AWAY_B)}
    b.hold(3.0, players, holder_id=1)
    states = _states(observations(b.frames, running=False))
    assert states[-1].clock_running is False and states[-1].clock_state == "read"
    states = _states(observations(b.frames, running=True))
    assert states[-1].clock_running is True

def test_hidden_bug_assumes_running_for_five_seconds():
    b = PlayBuilder(Scene(L)); players = {1: ("home", HOME_A), 2: ("away", AWAY_B)}
    b.hold(9.0, players, holder_id=1)
    obs = observations(b.frames, running=True, hidden=[(1.0, 9.0)])
    states = _states(obs)
    at = lambda t: states[int(t * 30)]
    assert at(2.5).clock_running and at(2.5).clock_state == "read"      # within 2 s of the last change
    assert at(5.0).clock_running and at(5.0).clock_state == "assumed"
    assert at(7.5).clock_running and at(7.5).clock_state == "assumed"   # 2 s read + 5 s assumed
    assert not at(8.5).clock_running and at(8.5).clock_state == "unknown"

def test_closeup_clears_holder_but_keeps_possession():
    b = PlayBuilder(Scene(L)); players = {1: ("home", HOME_A), 2: ("away", AWAY_B)}
    b.hold(0.5, players, holder_id=1)
    b.frame(players, None, camera_hint="closeup")
    states = _states(observations(b.frames))
    assert states[-1].holder is None and states[-1].possession == "home"
    assert states[-1].camera_left_main_pts == states[-1].t_pts

def test_release_and_rim_interaction_recorded_for_a_make():
    b = PlayBuilder(Scene(L)); players = {1: ("home", (22.0, 7.5)), 2: ("away", AWAY_B)}
    b.hold(0.3, players, holder_id=1)
    b.shot(1, players, "right", outcome="make")
    states = _states(observations(b.frames))
    rel = next(s.last_release for s in states if s.last_release is not None)
    assert rel.track_id == 1 and rel.team == "home" and rel.zone == "two"
    assert states[-1].last_shot is not None and states[-1].last_shot.zone == "two"
    ri = states[-1].rim_interaction
    assert ri.entered_pts is not None and ri.crossed_pts is not None

def test_pass_records_release_but_no_shot():
    b = PlayBuilder(Scene(L)); players = {1: ("home", HOME_A), 3: ("home", (16.0, 5.0)), 2: ("away", AWAY_B)}
    b.hold(0.3, players, holder_id=1)
    b.pass_(1, 3, players)
    b.hold(0.3, players, holder_id=3)
    states = _states(observations(b.frames))
    assert any(s.last_release is not None for s in states)
    assert states[-1].last_shot is None and states[-1].holder.track_id == 3

def test_ball_lost_keeps_holder():
    b = PlayBuilder(Scene(L)); players = {1: ("home", HOME_A), 2: ("away", AWAY_B)}
    b.hold(0.5, players, holder_id=1)
    for _ in range(20): b.frame(players, None)          # ball unseen for 0.67 s
    states = _states(observations(b.frames))
    assert states[-1].holder is not None and states[-1].holder.track_id == 1

def test_lane_formation_detected_when_clock_stopped():
    b = PlayBuilder(Scene(L))
    lane = {1: ("home", (5.8, 7.5)), 2: ("away", (3.0, 7.5 + 2.45)), 3: ("home", (3.0, 7.5 - 2.45)),
            4: ("away", (4.5, 7.5 + 2.6)), 5: ("home", (4.5, 7.5 - 2.6)), 6: ("away", (10.0, 7.5))}
    b.hold(0.5, lane, holder_id=1)
    states = _states(observations(b.frames, running=False))
    lf = states[-1].lane_formation
    assert lf is not None and lf.shooter_track_id == 1 and lf.side == "left" and lf.team == "home"
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/bin/python -m pytest tests/unit/test_basketball_vision_state.py -q`
Expected: FAIL with `ModuleNotFoundError: tests.helpers.vision`.

- [ ] **Step 3: Implement the builder, `vision_state.py` and the plug-in change**

Release detection: `prev.holder is not None`, `new` has no holder candidate matching it, the ball is `detected`, and its image y is smaller than on the previous frame; `Release.zone` comes from `zone_for(feet_xy, league, nearest_side(feet_xy))` when `homography_confidence >= MIN_HOMOGRAPHY_CONF` else `"unknown"`. The ball path collects `(t, x, y)` of detected ball observations after the release for `BALL_PATH_MAX_S`; when it reaches `SHOT_FIT_OBS` points, `shot_path_passes_rim` decides `last_shot` once per release. `possession` follows the holder's team; `possession_since_pts` is the holder's `since_pts` when the team changes.

- [ ] **Step 4: Run the unit suite and lint, then commit**

```bash
git add tests/helpers/vision.py sports_watcher/sports/basketball tests/unit/test_basketball_vision_state.py
git commit -m "feat(m1): synthetic play builder; clock gate, holder, release and rim facts in basketball state"
```

---

### Task 7: Confirmers and the possession rules

**Files:**
- Create: `sports_watcher/sports/basketball/confirmers.py`, `sports_watcher/sports/basketball/rules_possession.py`
- Modify: `sports_watcher/sports/basketball/plugin.py` (rule list)
- Test: `tests/unit/test_basketball_possession_rules.py`

**Interfaces:**
- Produces (`confirmers.py`): `always_confirm() -> Confirmer`; `team_holds(team: Team, hold_s: float = 0.5) -> Confirmer` (confirmed when `state.possession == team` and `obs.t_pts - state.possession_since_pts >= hold_s`; retracted when `state.possession` is the other team; pending otherwise); `team_holds_unless_attempt(team: Team, lost_team: Team, window: tuple[float, float]) -> Confirmer` (as `team_holds`, but retracted when `state.last_shot` by `lost_team` has `t_pts` inside `window`); `possession_changes_within(from_team: Team, since_pts: float, window_s: float) -> Confirmer`; `rim_or_delta(team: Team, since_pts: float, score_before: tuple[int, int]) -> Confirmer`; `delta_within(team: Team, points: tuple[int, ...], score_before: tuple[int, int]) -> Confirmer`; `holder_or_late_delta(team: Team, since_pts: float, score_before: tuple[int, int]) -> Confirmer`; `clock_stopped_until(deadline_pts: float) -> Confirmer`; `possession_within(since_pts: float, window_s: float = 5.0) -> Confirmer`; `ft_resolves(team: Team, since_pts: float, score_before: tuple[int, int]) -> Confirmer`. `team_delta(team, score_before, score_after) -> int`. `base_confidence(value: float, state: GameState) -> float` = `value - 0.2` when `state.clock_state == "assumed"`, else `value`, floored at 0.05. `vision_evidence(state: GameState, obs: Observation, **extra) -> Evidence` fills `camera_state`, `clock_state`, `homography_confidence`, `score_before`, `score_after` and any extra.
- Produces (`rules_possession.py`): `PossessionStartRule` (confidence 0.7), `ReboundRule` (0.6, `evidence.note` `offensive` or `defensive`), `TurnoverRule(league)` (0.6 on a possession change, 0.5 on a shot-clock reset with the holder unknown, `ttl_s=5.0` for the latter). Each is `(prev, new, obs) -> list[EventDraft]` and keeps per-game memory where the spec needs "once per X".
- Modify: `BasketballPlugin.__init__` rule list becomes `[ScoreChangeRule(), PeriodStartRule(league), PeriodEndRule(), PossessionStartRule(), ReboundRule(), TurnoverRule(league)]` (shot and clock rules are appended in Tasks 8 and 9).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_basketball_possession_rules.py
from tests.helpers.leagues import fixture_league
from tests.helpers.vision import PlayBuilder, Scene, confirmed, drive, observations, provisional, retracted

L = fixture_league("pba")
P = {1: ("home", (10.0, 7.0)), 2: ("away", (12.0, 8.0)), 3: ("home", (16.0, 5.0))}

def test_first_possession_is_drafted_then_confirmed_after_half_a_second():
    b = PlayBuilder(Scene(L)); b.hold(1.0, P, holder_id=1)
    ev = drive(observations(b.frames))
    prov = provisional(ev, "possession_start")
    assert len(prov) == 1 and prov[0].team == "home" and prov[0].t_pts == 0.0 and prov[0].player.track_id == 1
    assert prov[0].confidence == 0.7 and prov[0].evidence.clock_state == "read"
    conf = confirmed(ev, "possession_start")
    assert len(conf) == 1 and conf[0].supersedes == prov[0].id and conf[0].t_pts == 0.0

def test_pass_to_teammate_is_not_a_new_possession():
    b = PlayBuilder(Scene(L)); b.hold(0.6, P, holder_id=1); b.pass_(1, 3, P); b.hold(0.6, P, holder_id=3)
    ev = drive(observations(b.frames))
    assert len(provisional(ev, "possession_start")) == 1

def test_steal_flipping_back_within_half_a_second_is_retracted():
    b = PlayBuilder(Scene(L)); b.hold(0.6, P, holder_id=1); b.hold(0.3, P, holder_id=2); b.hold(0.6, P, holder_id=1)
    ev = drive(observations(b.frames))
    away = [e for e in provisional(ev, "possession_start") if e.team == "away"]
    assert len(away) == 1 and len(retracted(ev, "possession_start")) == 1
    assert retracted(ev, "possession_start")[0].supersedes == away[0].id

def test_same_team_after_closeup_is_not_a_new_possession():
    b = PlayBuilder(Scene(L)); b.hold(0.6, P, holder_id=1)
    for _ in range(30): b.frame(P, None, camera_hint="closeup")
    b.hold(0.6, P, holder_id=3)
    ev = drive(observations(b.frames))
    assert len(provisional(ev, "possession_start")) == 1
    b.hold(0.6, P, holder_id=2)
    ev = drive(observations(b.frames))
    assert [e.team for e in provisional(ev, "possession_start")] == ["home", "away"]

def test_no_possession_while_clock_is_stopped():
    b = PlayBuilder(Scene(L)); b.hold(1.0, P, holder_id=1)
    assert provisional(drive(observations(b.frames, running=False)), "possession_start") == []

def test_assumed_clock_lowers_confidence():
    b = PlayBuilder(Scene(L)); b.hold(4.0, P, holder_id=1); b.hold(1.0, P, holder_id=2)
    ev = drive(observations(b.frames, hidden=[(0.5, 5.0)]))
    away = [e for e in provisional(ev, "possession_start") if e.team == "away"][0]
    assert away.evidence.clock_state == "assumed" and away.confidence == 0.5

def test_rebound_after_miss_is_defensive_for_the_other_team():
    S = {1: ("home", (22.0, 7.5)), 2: ("away", (25.0, 6.0))}
    b = PlayBuilder(Scene(L)); b.hold(0.6, S, holder_id=1); b.shot(1, S, "right", outcome="miss"); b.hold(1.0, S, holder_id=2)
    ev = drive(observations(b.frames))
    reb = confirmed(ev, "rebound")
    assert len(reb) == 1 and reb[0].team == "away" and reb[0].evidence.note == "defensive"

def test_turnover_when_possession_changes_without_an_attempt():
    b = PlayBuilder(Scene(L)); b.hold(0.6, P, holder_id=1); b.hold(1.0, P, holder_id=2)
    ev = drive(observations(b.frames))
    to = confirmed(ev, "turnover")
    assert len(to) == 1 and to[0].team == "home" and abs(to[0].t_pts - 0.77) < 0.05   # away holder since 0.6 + 0.17

def test_unknown_team_holder_drafts_nothing():
    U = {1: ("unknown", (10.0, 7.0)), 2: ("away", (12.0, 8.0))}
    b = PlayBuilder(Scene(L)); b.hold(1.0, U, holder_id=1)
    assert provisional(drive(observations(b.frames)), "possession_start") == []

def test_no_turnover_when_the_change_follows_a_shot():
    S = {1: ("home", (22.0, 7.5)), 2: ("away", (25.0, 6.0))}
    b = PlayBuilder(Scene(L)); b.hold(0.6, S, holder_id=1); b.shot(1, S, "right", outcome="miss"); b.hold(1.0, S, holder_id=2)
    assert provisional(drive(observations(b.frames)), "turnover") == []
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/bin/python -m pytest tests/unit/test_basketball_possession_rules.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement confirmers and the three rules, register them**

`PossessionStartRule` fires when `new.possession != prev.possession`, `new.possession in ("home", "away")`, `new.holder is not None` and `new.clock_running`; `t_pts` is `new.possession_since_pts`; `player` is `PlayerRef(track_id=holder.track_id, confidence=track.confidence)`; `court_xy` is the holder track's `feet_xy`. `ReboundRule` remembers the score when `last_shot` changed and fires once per shot when a holder appears after the ball left the rim region (or 1.5 s after the attempt with no entry) with the score unchanged. `TurnoverRule` fires on a possession change when no `last_shot` has `t_pts >= prev.possession_since_pts`, with `team = prev.possession`, `t_pts = new.possession_since_pts`, confirmer `team_holds_unless_attempt(new.possession, prev.possession, (prev.possession_since_pts, new.possession_since_pts))`; and on a shot-clock reset (`prev.shot_clock_s <= league.shot_clock_s - 2` and `new.shot_clock_s == league.shot_clock_s`) with `new.holder is None`, possession unchanged and no attempt in the possession, confirmer `possession_changes_within(new.possession, obs.t_pts, 5.0)`.

- [ ] **Step 4: Run the unit suite and lint, then commit**

```bash
git add sports_watcher/sports/basketball tests/unit/test_basketball_possession_rules.py
git commit -m "feat(m1): possession_start, rebound and turnover rules with vision confirmers"
```

---

### Task 8: Shot rules

**Files:**
- Create: `sports_watcher/sports/basketball/rules_shots.py`
- Modify: `sports_watcher/sports/basketball/plugin.py` (rule list)
- Test: `tests/unit/test_basketball_shot_rules.py`

**Interfaces:**
- Produces: `ShotAttemptRule` (confidence 0.7, confirmer `rim_or_delta`); `ShotOutcomeRule` with per-game memory of open attempts `(t_pts, team, track_id, zone, court_xy, score_at, made_drafted: bool)`: drafts `shot_made` on a 2 or 3 delta within 3 s of an open attempt by that team (confidence 0.9, `always_confirm`), `shot_made` on a rim crossing with no delta yet (0.6, `delta_within(team, (2, 3), score_at)`), `shot_missed` when 3 s pass after the ball left the rim region, or after `t_pts + 1.5` when it never entered, with no delta (0.6, `holder_or_late_delta`), and on a 1, 2 or 3 delta with no open attempt by that team inside 3 s and no open free throw, the retroactive `shot_attempt` (0.4, `zone` `ft`/`two`/`three`, `court_xy` `None`, `evidence.note = "retroactive"`, `always_confirm`) followed for 2 and 3 by a `shot_made` at 0.4. Both rules are registered after the possession rules.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_basketball_shot_rules.py
from tests.helpers.leagues import fixture_league
from tests.helpers.vision import PlayBuilder, Scene, confirmed, drive, observations, provisional, retracted

L = fixture_league("pba")
S = {1: ("home", (22.0, 7.5)), 2: ("away", (25.0, 6.0)), 3: ("home", (18.0, 3.0))}

def _make_two(delta_at=None):
    b = PlayBuilder(Scene(L)); b.hold(0.6, S, holder_id=1); b.shot(1, S, "right", outcome="make")
    b.hold(1.5, S, holder_id=2)
    t_delta = delta_at if delta_at is not None else 0.6 + 1.0 + 0.2
    return b, observations(b.frames, deltas={t_delta: (2, 0)})

def test_attempt_drafted_at_release_with_zone_and_confirmed_by_rim():
    b, obs = _make_two()
    ev = drive(obs)
    att = provisional(ev, "shot_attempt")
    assert len(att) == 1 and att[0].team == "home" and att[0].zone == "two" and att[0].player.track_id == 1
    assert abs(att[0].t_pts - 0.6) < 0.05 and att[0].court_xy is not None
    assert len(confirmed(ev, "shot_attempt")) == 1

def test_delta_after_make_emits_confirmed_shot_made():
    _, obs = _make_two()
    ev = drive(obs)
    made = confirmed(ev, "shot_made")
    assert len(made) == 1 and made[0].team == "home" and made[0].evidence.points == 2 and made[0].confidence == 0.9
    assert retracted(ev, "shot_missed") == [] and provisional(ev, "shot_missed") == []

def test_rim_crossing_without_delta_is_provisional_then_retracted_by_ttl():
    b = PlayBuilder(Scene(L)); b.hold(0.6, S, holder_id=1); b.shot(1, S, "right", outcome="make"); b.hold(5.0, S, holder_id=2)
    ev = drive(observations(b.frames))
    made = provisional(ev, "shot_made")
    assert len(made) == 1 and made[0].confidence == 0.6
    assert retracted(ev, "shot_made")[0].reason == "ttl"

def test_three_point_zone_from_shooter_feet():
    T = {1: ("home", (18.0, 7.5)), 2: ("away", (25.0, 6.0))}
    b = PlayBuilder(Scene(L)); b.hold(0.6, T, holder_id=1); b.shot(1, T, "right", outcome="make"); b.hold(0.5, T, holder_id=2)
    ev = drive(observations(b.frames, deltas={1.8: (3, 0)}))
    assert provisional(ev, "shot_attempt")[0].zone == "three"
    assert confirmed(ev, "shot_made")[0].evidence.points == 3

def test_zone_unknown_without_homography():
    b = PlayBuilder(Scene(L)); b.hold(0.6, S, holder_id=1); b.shot(1, S, "right", outcome="make")
    for vf in b.frames: vf.homography = None
    ev = drive(observations(b.frames))
    assert provisional(ev, "shot_attempt")[0].zone == "unknown" and provisional(ev, "shot_attempt")[0].court_xy is None

def test_miss_then_rebound_emits_shot_missed_confirmed_by_next_holder():
    b = PlayBuilder(Scene(L)); b.hold(0.6, S, holder_id=1); b.shot(1, S, "right", outcome="miss"); b.hold(4.0, S, holder_id=2)
    ev = drive(observations(b.frames))
    missed = confirmed(ev, "shot_missed")
    assert len(missed) == 1 and missed[0].team == "home"

def test_shot_during_frozen_clock_is_not_drafted():
    b = PlayBuilder(Scene(L)); b.hold(0.6, S, holder_id=1); b.shot(1, S, "right", outcome="make"); b.hold(1.0, S, holder_id=2)
    ev = drive(observations(b.frames, running=False))
    assert provisional(ev, "shot_attempt") == [] and provisional(ev, "shot_made") == []

def test_pass_away_from_rim_is_not_a_shot():
    b = PlayBuilder(Scene(L)); b.hold(0.6, S, holder_id=1); b.pass_(1, 3, S); b.hold(1.0, S, holder_id=3)
    assert provisional(drive(observations(b.frames)), "shot_attempt") == []

def test_lob_caught_by_teammate_retracts_attempt():
    U = {1: ("home", (22.0, 7.5)), 3: ("home", (25.5, 7.0)), 2: ("away", (12.0, 6.0))}
    b = PlayBuilder(Scene(L)); b.hold(0.6, U, holder_id=1); b.lob(1, 3, U, "right"); b.hold(1.0, U, holder_id=3)
    ev = drive(observations(b.frames))
    assert len(provisional(ev, "shot_attempt")) == 1
    assert retracted(ev, "shot_attempt")[0].reason in ("same_team_holds", "ttl")

def test_delta_after_lost_ball_confirms_made_without_duplicate():
    b = PlayBuilder(Scene(L)); b.hold(0.6, S, holder_id=1)
    flight = b.shot(1, S, "right", outcome="make")
    for vf in flight[6:]: vf.ball = None                      # ball lost after the fit window
    b.hold(3.0, S, holder_id=2)
    ev = drive(observations(b.frames, deltas={0.6 + 2.5: (2, 0)}))
    assert len(provisional(ev, "shot_attempt")) == 1 and len(confirmed(ev, "shot_made")) == 1
    assert all(e.evidence.note != "retroactive" for e in ev)

def test_delta_without_attempt_emits_retroactive_attempt():
    b = PlayBuilder(Scene(L)); b.hold(3.0, S, holder_id=2)
    ev = drive(observations(b.frames, deltas={2.0: (0, 3)}))
    att = confirmed(ev, "shot_attempt")
    assert len(att) == 1 and att[0].confidence == 0.4 and att[0].zone == "three" and att[0].team == "away"
    assert att[0].court_xy is None and att[0].evidence.note == "retroactive"
    assert confirmed(ev, "shot_made")[0].confidence == 0.4
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/bin/python -m pytest tests/unit/test_basketball_shot_rules.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the two rules, register them, add the retract reason**

`rim_or_delta` returns `"retracted"` when the shooting team re-establishes possession for 0.5 s with no rim entry; the engine maps the verdict to reason `scoreboard_disagrees` today, so extend `Verdict` in `sports/base.py` to accept `"retracted:<reason>"` strings and make `EventEngine._terminal` split the reason off (`RETRACT_REASONS` keeps the plain `retracted` mapping). Use reason `same_team_holds` here. `ShotAttemptRule` skips when `new.lane_formation is not None` (free throws belong to Task 9).

- [ ] **Step 4: Run the unit suite and lint, then commit**

```bash
git add sports_watcher tests/unit/test_basketball_shot_rules.py
git commit -m "feat(m1): shot_attempt, shot_made and shot_missed rules with retroactive attempts"
```

---

### Task 9: Timeout, tip-off and free-throw rules

**Files:**
- Create: `sports_watcher/sports/basketball/rules_clock.py`
- Modify: `sports_watcher/sports/basketball/rules_shots.py` (add `FreeThrowRule`), `sports_watcher/sports/basketball/plugin.py`
- Test: `tests/unit/test_basketball_clock_rules.py`

**Interfaces:**
- Produces (`rules_clock.py`): `TimeoutRule` (team `unknown`, `t_pts = camera_left_main_pts`, confidence 0.7, `ttl_s=12.0`, confirmer `clock_stopped_until(camera_left_main_pts + 30.0)`, once per departure from main); `TipOffRule(league)` (once per period for period 1 and overtime periods; needs clock at full within 0.05 s, camera main, homography confidence at least 0.4, two tracks' feet within 2 m of `centre_xy`, a detected ball whose image y decreased on three consecutive perception frames; team `unknown`, confidence 0.6, `ttl_s=10.0`, confirmer `possession_within(t_pts, 5.0)`).
- Produces (`rules_shots.py`): `FreeThrowRule` drafts `free_throw_attempt` when `new.last_release` is new, its `track_id` is `new.lane_formation.shooter_track_id` and `not new.clock_running` (team of the shooter, `zone="ft"`, `court_xy = free_throw_xy`, confidence 0.6, confirmer `ft_resolves`); then `free_throw_made` on a +1 delta for that team within 3 s (0.9, `always_confirm`) or `free_throw_missed` 3 s after the attempt with no delta (0.6, `holder_or_late_delta`). `ShotOutcomeRule` ignores +1 deltas while a free throw is open (shared `open_free_throw` flag on the plug-in state: `GameState.free_throws_pending`).
- Rule order in the plug-in: scoreboard rules, possession rules, `ShotAttemptRule`, `ShotOutcomeRule`, `FreeThrowRule`, `TimeoutRule`, `TipOffRule`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_basketball_clock_rules.py
from tests.helpers.leagues import fixture_league
from tests.helpers.vision import PlayBuilder, Scene, confirmed, drive, observations, provisional, retracted

L = fixture_league("pba")
P = {1: ("home", (10.0, 7.0)), 2: ("away", (12.0, 8.0))}
LANE = {1: ("home", (5.8, 7.5)), 2: ("away", (3.0, 9.95)), 3: ("home", (3.0, 5.05)),
        4: ("away", (4.5, 10.1)), 5: ("home", (4.5, 4.9)), 6: ("away", (10.0, 7.5))}

def test_timeout_drafted_at_twenty_seconds_and_confirmed_at_thirty():
    b = PlayBuilder(Scene(L)); b.hold(0.5, P, holder_id=1)
    for _ in range(31 * 30): b.frame(P, None, camera_hint="closeup")
    ev = drive(observations(b.frames, running=False))
    to = provisional(ev, "timeout")
    assert len(to) == 1 and abs(to[0].t_pts - 0.5) < 0.05 and to[0].team == "unknown"
    assert len(confirmed(ev, "timeout")) == 1

def test_timeout_retracted_when_play_resumes():
    b = PlayBuilder(Scene(L)); b.hold(0.5, P, holder_id=1)
    for _ in range(22 * 30): b.frame(P, None, camera_hint="closeup")
    b.hold(3.0, P, holder_id=1)
    ev = drive(observations(b.frames, running=False, resume_at=22.5))   # drafted at 20.5, clock runs before 30.5
    assert len(retracted(ev, "timeout")) == 1 and confirmed(ev, "timeout") == []

def test_no_timeout_while_clock_runs_during_a_closeup():
    b = PlayBuilder(Scene(L))
    for _ in range(25 * 30): b.frame(P, None, camera_hint="closeup")
    assert provisional(drive(observations(b.frames, running=True)), "timeout") == []

def test_tip_off_confirmed_by_first_possession():
    C = {1: ("home", (13.5, 7.5)), 2: ("away", (14.5, 7.5)), 3: ("home", (16.0, 5.0))}
    b = PlayBuilder(Scene(L))
    for i in range(6): b.frame(C, ((14.0, 7.5), 2.0 + 0.3 * i))      # ball rising at centre
    b.hold(1.0, C, holder_id=3)
    ev = drive(observations(b.frames, running=False, resume_at=6 / 30, clock_start_s=600.0, period=1))
    tip = provisional(ev, "tip_off")
    assert len(tip) == 1 and tip[0].team == "unknown" and abs(tip[0].t_pts - 1 / 30) < 0.05
    assert len(confirmed(ev, "tip_off")) == 1

def test_tip_off_retracted_without_possession():
    C = {1: ("home", (13.5, 7.5)), 2: ("away", (14.5, 7.5))}
    b = PlayBuilder(Scene(L))
    for i in range(6): b.frame(C, ((14.0, 7.5), 2.0 + 0.3 * i))
    for _ in range(11 * 30): b.frame(C, None)
    ev = drive(observations(b.frames, running=False, clock_start_s=600.0, period=1))
    assert len(retracted(ev, "tip_off")) == 1

def test_no_tip_off_mid_period():
    C = {1: ("home", (13.5, 7.5)), 2: ("away", (14.5, 7.5))}
    b = PlayBuilder(Scene(L))
    for i in range(6): b.frame(C, ((14.0, 7.5), 2.0 + 0.3 * i))
    assert provisional(drive(observations(b.frames, clock_start_s=431.0)), "tip_off") == []

def test_free_throw_made():
    b = PlayBuilder(Scene(L)); b.hold(0.6, LANE, holder_id=1); b.shot(1, LANE, "left", outcome="make"); b.hold(1.0, LANE, holder_id=1)
    ev = drive(observations(b.frames, running=False, deltas={0.6 + 1.2: (1, 0)}))
    att = confirmed(ev, "free_throw_attempt")
    assert len(att) == 1 and att[0].team == "home" and att[0].zone == "ft" and abs(att[0].t_pts - 0.6) < 0.05
    assert len(confirmed(ev, "free_throw_made")) == 1 and provisional(ev, "shot_attempt") == []

def test_free_throw_missed():
    b = PlayBuilder(Scene(L)); b.hold(0.6, LANE, holder_id=1); b.shot(1, LANE, "left", outcome="miss"); b.hold(4.0, LANE, holder_id=2)
    ev = drive(observations(b.frames, running=False))
    assert len(confirmed(ev, "free_throw_missed")) == 1 and confirmed(ev, "free_throw_made") == []
```

- [ ] **Step 2: Run to verify failure, implement, run the suite and lint, commit**

Run: `./.venv/bin/python -m pytest tests/unit/test_basketball_clock_rules.py -q` (expected `ModuleNotFoundError`), implement, then the full unit suite and ruff. Commit: `feat(m1): timeout, tip_off and free-throw rules`.

---

### Task 10: Firsts folds, truth zones, camera truth and evaluation

**Files:**
- Modify: `sports_watcher/events/firsts.py`, `sports_watcher/replay/truth.py`, `sports_watcher/eval/score.py`
- Test: `tests/unit/test_firsts.py` (append), `tests/unit/test_replay.py` (append), `tests/unit/test_eval.py` (append)

**Interfaces:**
- Produces (`firsts.py`): `fold_firsts` fills `first_possession` (team of the first confirmed `possession_start` by `t_pts`), `tip_off_winner` (team of the first confirmed `possession_start` whose `t_pts` is within 5 s after a confirmed `tip_off`), `first_shot_attempt` (`FirstRef` of the first confirmed `shot_attempt` by `t_pts`), `possessions` (one `PossessionResult(period, team, result, t_pts)` per confirmed `possession_start` ordered by `t_pts`; result `score` when a confirmed `shot_made` or `free_throw_made` by that team has `t_pts` in `[start, next_start)`, else `turnover` when a confirmed `turnover` by that team does, else `miss` when a confirmed `shot_missed` or `free_throw_missed` by that team does, else `end_period` when a confirmed `period_end` does; a possession with none of those and no later `possession_start` is left out). `TIP_OFF_WINDOW_S = 5.0`.
- Produces (`truth.py`): `TruthEvent.zone: str | None = None`; `CameraSegment(BaseModel)`: `start: float`, `end: float`, `state: str`; `load_camera_truth(path) -> list[CameraSegment]`.
- Produces (`score.py`): `_compatible` also requires equal `zone` when both the truth and the event carry one; `camera_state_accuracy(log: list[tuple[float, str]], segments: list[CameraSegment]) -> float | None` (fraction of log entries inside any segment whose state matches it; `None` when no entry falls inside a segment).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_firsts.py (append)
from sports_watcher.events.model import EventStatus
from tests.helpers.events import make_event

def _c(seq, type, team, t, **kw):
    return make_event(seq=seq, type=type, team=team, t_pts=t, status=EventStatus.confirmed, **kw)

def test_vision_firsts_and_possessions():
    ev = [
        _c(1, "tip_off", "unknown", 1.0), _c(2, "possession_start", "away", 2.0),
        _c(3, "shot_attempt", "away", 8.0), _c(4, "shot_missed", "away", 9.0),
        _c(5, "possession_start", "home", 10.0), _c(6, "turnover", "home", 15.0),
        _c(7, "possession_start", "away", 15.0), _c(8, "shot_made", "away", 20.0),
        _c(9, "possession_start", "home", 21.0), _c(10, "period_end", "unknown", 30.0),
        _c(11, "possession_start", "away", 40.0, period=2),
    ]
    f = fold_firsts(ev, load_league("pba"))
    assert f.first_possession == "away" and f.tip_off_winner == "away"
    assert f.first_shot_attempt.team == "away" and f.first_shot_attempt.t_pts == 8.0
    assert [(p.team, p.result) for p in f.possessions] == [("away", "miss"), ("home", "turnover"), ("away", "score"), ("home", "end_period")]

def test_tip_off_winner_needs_possession_within_five_seconds():
    ev = [_c(1, "tip_off", "unknown", 1.0), _c(2, "possession_start", "home", 7.0)]
    f = fold_firsts(ev, load_league("pba"))
    assert f.tip_off_winner is None and f.first_possession == "home"
```

```python
# tests/unit/test_replay.py (append)
def test_truth_zone_and_camera_segments(tmp_path):
    from sports_watcher.replay.truth import TruthEvent, load_camera_truth
    assert TruthEvent(t_pts=1.0, type="shot_attempt", team="home", zone="three").zone == "three"
    p = tmp_path / "c.camera.jsonl"
    p.write_text('{"start": 0.0, "end": 3.0, "state": "main"}\n{"start": 3.0, "end": 5.0, "state": "closeup"}\n')
    segs = load_camera_truth(p)
    assert [s.state for s in segs] == ["main", "closeup"] and segs[1].end == 5.0
```

```python
# tests/unit/test_eval.py (append)
def test_zone_mismatch_does_not_match():
    from sports_watcher.eval.score import match_events
    from sports_watcher.replay.truth import TruthEvent
    e = make_event(type="shot_attempt", status=EventStatus.confirmed, t_pts=10.0, zone="two")
    assert match_events([e], [TruthEvent(t_pts=10.2, type="shot_attempt", team="home", zone="three")])[0][1] is None
    assert match_events([e], [TruthEvent(t_pts=10.2, type="shot_attempt", team="home")])[0][1] is e

def test_camera_state_accuracy():
    from sports_watcher.eval.score import camera_state_accuracy
    from sports_watcher.replay.truth import CameraSegment
    segs = [CameraSegment(start=0.0, end=2.0, state="main"), CameraSegment(start=2.0, end=4.0, state="closeup")]
    log = [(0.5, "main"), (1.5, "main"), (2.5, "main"), (3.5, "closeup"), (9.0, "graphics")]
    assert camera_state_accuracy(log, segs) == 0.75
    assert camera_state_accuracy([(9.0, "main")], segs) is None
```

- [ ] **Step 2: Run to verify failure, implement, run the suite and lint, commit**

Commit: `feat(m1): firsts for possession, tip-off and first attempt; truth zones and camera-state accuracy`.

---

### Task 11: Watcher integration, camera-state tracker and the CLI

**Files:**
- Create: `sports_watcher/perception/camera_state.py` (tracker part only)
- Modify: `sports_watcher/watcher.py`, `sports_watcher/perception/perceiver.py` (`build_perceiver`), `scripts/run_watch.py`, `sports_watcher/api/schemas.py` or wherever `WatcherStatus` is exposed (add `camera_state`)
- Test: `tests/unit/test_watcher_vision.py`, `tests/unit/test_camera_state_tracker.py`

**Interfaces:**
- Produces (`camera_state.py`): `CameraStateTracker(hold_s: float = 0.5, min_confidence: float = 0.3)`: `update(hint: CameraHint, confidence: float, t_pts: float) -> str` (the first hint becomes the state; a different hint must persist for `hold_s` before the state changes; hints under `min_confidence` are ignored); `state: str` property (`main` before any update); `reset()`.
- Produces (`watcher.py`): `GameWatcher(..., perceiver: Perceiver | None = None, perceive_every_n_frames: int | None = None, log_camera_states: bool = False)`; `perceive_every_n_frames=None` means derived from the first two frames' pts delta as `max(1, round(1 / (dt * 30)))`; perception runs on every n-th processed frame, `camera_state` comes from the tracker (always `main` without a perceiver), `Observation.vision` is the `VisionFrame` on perception frames and `None` otherwise; `WatcherStatus.camera_state: str`; `camera_log: list[tuple[float, str]]` filled per perception frame when `log_camera_states`; `perceiver.reset()` and tracker reset before each `_run_once` consumption; `vision_enabled: bool` property.
- Produces (`perceiver.py`): `build_perceiver(settings: Settings, league: LeagueConfig, clip: Path | None, home_color: str | None, away_color: str | None, use_cache: bool = True) -> Perceiver | None`: `ModelSet.load(settings.models_dir)`; with models, an `OnnxPerceiver` (imported lazily from `sports_watcher.perception.onnx_perceiver`, Task 17) wrapped in `CachedPerceiver` when `clip` and `use_cache`; without models but with `find_cache(settings.cache_dir, clip)`, `CachedPerceiver(None, path)`; otherwise `None` (logged once: "no vision models in <dir>, scoreboard-only").
- Produces (`run_watch.py`): flags `--models DIR`, `--no-vision`, `--no-cache`, `--home-color`, `--away-color`, `--perceive-every N`, `--camera-log PATH` (writes `{"t_pts", "state"}` lines).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_camera_state_tracker.py
from sports_watcher.perception.camera_state import CameraStateTracker

def test_first_hint_sets_state_and_changes_need_half_a_second():
    t = CameraStateTracker()
    assert t.state == "main"
    assert t.update("closeup", 0.9, 0.0) == "closeup"
    assert t.update("main", 0.9, 0.1) == "closeup"
    assert t.update("main", 0.9, 0.4) == "closeup"
    assert t.update("main", 0.9, 0.6) == "main"

def test_flicker_does_not_switch():
    t = CameraStateTracker(); t.update("main", 0.9, 0.0)
    for i in range(10):
        assert t.update("graphics" if i % 2 else "main", 0.9, 0.1 * i) == "main"

def test_low_confidence_hints_are_ignored():
    t = CameraStateTracker(); t.update("main", 0.9, 0.0)
    assert t.update("graphics", 0.1, 1.0) == "main" and t.update("graphics", 0.1, 2.0) == "main"
```

```python
# tests/unit/test_watcher_vision.py
from sports_watcher.perception.perceiver import ScriptedPerceiver
from sports_watcher.perception.types import VisionFrame, empty_vision
from sports_watcher.watcher import GameWatcher
from sports_watcher.ingest.base import SourceSpec
from sports_watcher.ingest.factory import register_source
from tests.helpers.leagues import fixture_league
from tests.helpers.sources import StubSource
from tests.helpers.text_readers import SeqTextReader

def _watcher(perceiver, n=60, fps=60, **kw):
    register_source("stub", lambda spec: StubSource(n=n, fps=fps))      # the pattern tests/unit/test_watcher.py uses
    return GameWatcher(SourceSpec(kind="stub"), fixture_league(), "one_sports", "g", "H", "A",
                       text_reader=SeqTextReader([[]]), perceiver=perceiver, **kw)

def test_perception_runs_on_every_second_frame_at_sixty_fps():
    p = ScriptedPerceiver([])
    w = _watcher(p); w.run()
    assert p.calls == list(range(0, 60, 2))

def test_perception_runs_on_every_frame_at_thirty_fps():
    p = ScriptedPerceiver([])
    w = _watcher(p, n=30, fps=30); w.run()
    assert p.calls == list(range(30))

def test_camera_state_in_status_and_log():
    frames = [VisionFrame(i, i / 60, "graphics" if i >= 20 else "main", 0.9, [], None, None, None) for i in range(0, 60, 2)]
    w = _watcher(ScriptedPerceiver(frames), log_camera_states=True); w.run()
    assert w.status().camera_state == "graphics"
    assert w.camera_log[0] == (0.0, "main") and w.camera_log[-1][1] == "graphics"
    assert any(s == "main" for t, s in w.camera_log if 20 / 60 <= t < 0.8)   # hysteresis holds main for 0.5 s

def test_without_perceiver_behaviour_is_m0():
    w = _watcher(None); w.run()
    assert w.status().camera_state == "main" and w.vision_enabled is False
```

```python
# tests/unit/test_perceiver_cache.py (append)
def test_build_perceiver_without_models_or_cache_is_none(tmp_path):
    from sports_watcher.config import Settings
    from sports_watcher.perception.perceiver import build_perceiver
    s = Settings(_env_file=None, models_dir=tmp_path / "m", cache_dir=tmp_path / "c")
    assert build_perceiver(s, fixture_league(), tmp_path / "clip.mp4", None, None) is None

def test_build_perceiver_uses_cache_without_models(tmp_path):
    ...  # write a cache file via CachedPerceiver(ScriptedPerceiver([...], "v1"), path) then
    p = build_perceiver(s, fixture_league(), clip, None, None)
    assert isinstance(p, CachedPerceiver) and p.version == "v1"
```

- [ ] **Step 2: Run to verify failure, implement, run the suite and lint, commit**

Commit: `feat(m1): perceiver and camera state in the watcher, status camera_state, run_watch vision flags`.

---

### Task 12: Player tracker and ball filter

**Files:**
- Create: `sports_watcher/perception/tracker.py`
- Test: `tests/unit/test_tracker.py`

**Interfaces:**
- Produces: `Detection(box: Box, cls: str, confidence: float)` dataclass; `TrackState(track_id: int, box: Box, cls: str, confidence: float, hits: int, time_since_update_s: float)`; `Tracker(max_age_s: float = 1.0, min_hits: int = 3, iou_threshold: float = 0.3, velocity_weight: float = 0.2)`: `update(dets: list[Detection], t_pts: float) -> list[TrackState]` (published tracks only: `hits >= min_hits`, or any track updated this frame that was published before; ids start at 1 and are never reused), `reset()`. Per-track `filterpy.kalman.KalmanFilter(dim_x=7, dim_z=4)` on `(cx, cy, area, aspect, vcx, vcy, varea)`; cost = `1 - IoU` plus `velocity_weight` times the angle cost between the track's observation velocity and the direction to the candidate (OC-SORT's OCM); `scipy.optimize.linear_sum_assignment`; on re-association after missed frames, re-run the filter along the straight line between the last and the new observation (ORU). `iou(a: Box, b: Box) -> float`. `BallFilter(drop_after_s: float = 0.5, gate_px: float = 60.0)`: `predict(t_pts) -> tuple[float, float] | None`; `update(xy, radius_px, confidence, t_pts) -> bool` (rejects a detection farther than `max(gate_px, 3 * radius_px)` from the prediction once initialised); `current(t_pts) -> BallObs | None` (`detected` when updated at `t_pts`, `predicted` otherwise, `None` after `drop_after_s` without an update); `reset()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_tracker.py
from sports_watcher.perception.tracker import BallFilter, Detection, Tracker, iou

def _box(x, y, w=40, h=100): return (x, y, x + w, y + h)

def test_iou():
    assert iou(_box(0, 0), _box(0, 0)) == 1.0 and iou(_box(0, 0), _box(100, 100)) == 0.0

def test_constant_motion_keeps_one_id_and_publishes_after_three_hits():
    tr = Tracker()
    out = [tr.update([Detection(_box(10 * i, 50), "player", 0.9)], i / 30) for i in range(30)]
    assert out[0] == [] and out[1] == [] and len(out[2]) == 1
    assert {t.track_id for frame in out[2:] for t in frame} == {1}

def test_ten_frame_occlusion_reassociates_same_id():
    tr = Tracker()
    for i in range(10): tr.update([Detection(_box(10 * i, 50), "player", 0.9)], i / 30)
    for i in range(10, 20): tr.update([], i / 30)
    out = tr.update([Detection(_box(200, 50), "player", 0.9)], 20 / 30)
    assert [t.track_id for t in out] == [1]

def test_track_dropped_after_one_second():
    tr = Tracker()
    for i in range(5): tr.update([Detection(_box(0, 0), "player", 0.9)], i / 30)
    for i in range(5, 40): tr.update([], i / 30)
    out = tr.update([Detection(_box(0, 0), "player", 0.9)], 40 / 30)
    assert out == [] or out[0].track_id == 2

def test_crossing_players_keep_their_ids():
    tr = Tracker()
    ids = []
    for i in range(40):
        a = Detection(_box(10 * i, 100), "player", 0.9)          # left to right
        b = Detection(_box(400 - 10 * i, 100), "player", 0.9)    # right to left
        out = tr.update([a, b], i / 30)
        if out: ids.append(tuple(sorted((t.track_id, round(t.box[0])) for t in out)))
    first, last = ids[0], ids[-1]
    assert dict(first)[1] < dict(first)[2] and dict(last)[1] > dict(last)[2]

def test_ball_filter_predicts_and_drops():
    bf = BallFilter()
    for i in range(10): bf.update((10.0 * i, 100.0 - 2.0 * i), 6.0, 0.8, i / 30)
    assert bf.current(9 / 30).source == "detected"
    pred = bf.predict(12 / 30)
    assert abs(pred[0] - 120.0) < 3 and abs(pred[1] - 76.0) < 3
    assert bf.current(12 / 30).source == "predicted"
    assert bf.current(9 / 30 + 0.6) is None

def test_ball_filter_rejects_far_detection():
    bf = BallFilter()
    for i in range(5): bf.update((10.0 * i, 100.0), 6.0, 0.8, i / 30)
    assert bf.update((500.0, 500.0), 6.0, 0.9, 5 / 30) is False
    assert bf.update((50.0, 100.0), 6.0, 0.9, 5 / 30) is True
```

- [ ] **Step 2: Run to verify failure, implement, run the suite and lint, commit**

Commit: `feat(m1): OC-SORT style player tracker and ball Kalman filter`.

---

### Task 13: Court mapper

**Files:**
- Create: `sports_watcher/perception/court.py`
- Test: `tests/unit/test_court.py`

**Interfaces:**
- Produces: `load_template(path: Path, league: LeagueConfig) -> tuple[list[str], np.ndarray]` reads `keypoints: [{name, x_frac, y_frac}, ...]` and returns names and `(N, 2)` metres (`x_frac * court_length_m`, `y_frac * court_width_m`); `order_template(names: list[str], xy: np.ndarray, manifest_keypoints: list[str]) -> np.ndarray` reorders rows to the model's keypoint order (raises `KeyError` naming any missing keypoint). `CourtMapper(template_xy: np.ndarray, min_point_conf: float = 0.5, min_inliers: int = 6, ransac_px: float = 8.0, smooth_s: float = 1.0, min_confidence: float = 0.4, hold_s: float = 2.0)`: `update(points_xy: np.ndarray, conf: np.ndarray, t_pts: float) -> Homography | None` (`cv2.findHomography(image_pts, court_pts, cv2.RANSAC, ransac_px)` over points with `conf >= min_point_conf`; confidence `min(1, inliers / 12) * max(0, 1 - err / ransac_px)` with `err` the mean inlier reprojection error; `None` under `min_inliers` or `min_confidence`; the returned matrix is the confidence-weighted mean of the accepted matrices of the last `smooth_s`, each normalised to `H[2, 2] == 1`), `current(t_pts) -> Homography | None` (last accepted result, `None` after `hold_s` without one), `reset()`. Functions `image_to_court(H: np.ndarray, xy) -> tuple[float, float]` and `court_to_image(H, xy)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_court.py
import cv2, numpy as np, yaml
from sports_watcher.perception.court import CourtMapper, court_to_image, image_to_court, load_template, order_template
from tests.helpers.leagues import fixture_league

L = fixture_league("pba")
GRID = np.array([(x, y) for x in (0, 5.8, 14, 22.2, 28) for y in (0, 7.5, 15)], float)   # 15 court points
H_TRUE = cv2.getPerspectiveTransform(np.float32([(0, 0), (28, 0), (28, 15), (0, 15)]),
                                     np.float32([(100, 1000), (1820, 1000), (1500, 350), (420, 350)]))
def _img(pts, noise=0.0, rng=np.random.default_rng(0)):
    p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), H_TRUE).reshape(-1, 2)
    return p + rng.normal(0, noise, p.shape)

def test_recovers_homography_within_twenty_centimetres():
    m = CourtMapper(GRID)
    h = m.update(_img(GRID, 1.0), np.full(15, 0.9), 0.0)
    assert h is not None and h.inliers >= 12 and h.confidence >= 0.4
    back = np.array([image_to_court(h.matrix, tuple(p)) for p in _img(GRID)])
    assert np.abs(back - GRID).max() < 0.2
    assert np.allclose(court_to_image(h.matrix, (14.0, 7.5)), _img(np.array([[14.0, 7.5]]))[0], atol=3)

def test_too_few_confident_points_is_none():
    conf = np.full(15, 0.2); conf[:5] = 0.9
    assert CourtMapper(GRID).update(_img(GRID), conf, 0.0) is None

def test_outliers_are_rejected():
    pts = _img(GRID, 1.0); pts[[1, 7, 11]] += (300, -200)
    h = CourtMapper(GRID).update(pts, np.full(15, 0.9), 0.0)
    back = np.array([image_to_court(h.matrix, tuple(p)) for p in _img(GRID)])
    assert np.abs(np.delete(back, [1, 7, 11], 0) - np.delete(GRID, [1, 7, 11], 0)).max() < 0.3

def test_smoothing_reduces_jitter_and_current_holds_two_seconds():
    raw, smooth, m = [], [], CourtMapper(GRID)
    for i in range(10):
        pts = _img(GRID, 3.0)
        raw.append(image_to_court(CourtMapper(GRID).update(pts, np.full(15, 0.9), i * 0.2).matrix, (960.0, 600.0)))
        smooth.append(image_to_court(m.update(pts, np.full(15, 0.9), i * 0.2).matrix, (960.0, 600.0)))
    assert np.std(np.array(smooth), axis=0).sum() < np.std(np.array(raw), axis=0).sum()
    assert m.current(1.8 + 1.9) is not None and m.current(1.8 + 2.1) is None

def test_template_loading_and_ordering(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump({"keypoints": [{"name": "a", "x_frac": 0.0, "y_frac": 0.0}, {"name": "b", "x_frac": 0.5, "y_frac": 1.0}]}))
    names, xy = load_template(p, L)
    assert names == ["a", "b"] and xy.tolist() == [[0.0, 0.0], [14.0, 15.0]]
    assert order_template(names, xy, ["b", "a"]).tolist() == [[14.0, 15.0], [0.0, 0.0]]
```

- [ ] **Step 2: Run to verify failure, implement, run the suite and lint, commit**

Commit: `feat(m1): court homography mapper with RANSAC, smoothing and template loading`.

---

### Task 14: Team assignment

**Files:**
- Create: `sports_watcher/perception/team.py`
- Test: `tests/unit/test_team.py`

**Interfaces:**
- Produces: `Embedder = Callable[[np.ndarray], np.ndarray]`; `parse_color(text: str) -> np.ndarray` (Lab from `#rrggbb` or one of `white, black, red, blue, green, yellow, orange, purple, navy, maroon, gold, teal, grey`; `ValueError` otherwise); `crop_color_lab(crop_bgr: np.ndarray) -> np.ndarray` (median Lab of the torso window: rows 25 % to 55 %, columns 25 % to 75 %); `map_clusters(cluster_lab: np.ndarray, home_lab: np.ndarray, away_lab: np.ndarray) -> dict[int, Team]` (the assignment of the two clusters to home and away with the smaller total Lab distance); `TeamAssigner(embed: Embedder, home_lab: np.ndarray | None, away_lab: np.ndarray | None, fit_after: int = 200, refit_every_s: float = 30.0, sample_every_s: float = 1.0)`: `observe(track_id: int, crop_bgr: np.ndarray, t_pts: float) -> None` (embeds the crop at the track's first observation and then at most once per `sample_every_s`; fits KMeans k=2 with `n_init=4, random_state=0` once `fit_after` samples exist and refits every `refit_every_s` on all samples kept, capped at 2,000), `team_of(track_id: int) -> Team` (`unknown` before the fit or for a track with no samples; otherwise the majority cluster label of the track's samples mapped through `map_clusters`; without operator colours clusters are `home`/`away` by cluster index, which `GameWatcher` logs as a warning), `fitted: bool`, `reset()`. `SiglipEmbedder(onnx_path: Path, providers: list[str] | None = None)`: `__call__(crop_bgr) -> np.ndarray` (shape `(768,)`, L2-normalised `pooler_output`); `SiglipEmbedder.preprocess(crop_bgr) -> np.ndarray` static: squash to 224 by 224, BGR to RGB, `float32`, `/255`, then `(x - 0.5) / 0.5`, NCHW `(1, 3, 224, 224)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_team.py
import numpy as np
from sports_watcher.perception.team import SiglipEmbedder, TeamAssigner, crop_color_lab, map_clusters, parse_color

RED, BLUE = np.full((100, 40, 3), (0, 0, 220), np.uint8), np.full((100, 40, 3), (220, 0, 0), np.uint8)
def fake_embed(crop):           # a 2-d embedding that separates by colour, with mild noise
    c = crop_color_lab(crop); return np.array([c[1], c[2]]) / 128 + np.random.default_rng(int(c.sum())).normal(0, 0.01, 2)

def _assigner(**kw):
    return TeamAssigner(fake_embed, parse_color("red"), parse_color("blue"), **kw)

def test_unknown_before_fit_then_split_by_colour():
    a = _assigner(fit_after=20)
    for i in range(10):
        a.observe(i, RED, 0.0); a.observe(100 + i, BLUE, 0.0)
    assert a.fitted and a.team_of(3) == "home" and a.team_of(103) == "away" and a.team_of(999) == "unknown"
    b = _assigner(fit_after=200); b.observe(1, RED, 0.0)
    assert b.fitted is False and b.team_of(1) == "unknown"

def test_samples_once_per_second_per_track():
    calls = []
    a = TeamAssigner(lambda c: (calls.append(1), np.zeros(2))[1], None, None, fit_after=10**6)
    for i in range(30): a.observe(1, RED, i / 30)
    assert len(calls) == 2            # first frame and the one-second mark

def test_majority_vote_survives_noisy_crops():
    a = _assigner(fit_after=20)
    for i in range(10): a.observe(i, RED, 0.0); a.observe(100 + i, BLUE, 0.0)
    for t in range(1, 6): a.observe(1, RED if t != 3 else BLUE, float(t))
    assert a.team_of(1) == "home"

def test_map_clusters_prefers_smaller_total_distance():
    red, blue = parse_color("red"), parse_color("blue")
    assert map_clusters(np.stack([blue, red]), red, blue) == {0: "away", 1: "home"}

def test_parse_color_hex_and_names():
    assert np.allclose(parse_color("#ff0000"), parse_color("red"))
    import pytest
    with pytest.raises(ValueError): parse_color("sparkly")

def test_siglip_preprocess_shape_and_range():
    x = SiglipEmbedder.preprocess(np.full((300, 120, 3), 255, np.uint8))
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32 and np.allclose(x, 1.0)
    assert np.allclose(SiglipEmbedder.preprocess(np.zeros((10, 10, 3), np.uint8)), -1.0)
```

- [ ] **Step 2: Run to verify failure, implement, run the suite and lint, commit**

Commit: `feat(m1): team assignment from crop embeddings with operator colour mapping; SigLIP preprocessing`.

---

### Task 15: Detector, ball crop refinement and keypoint model wrappers

**Files:**
- Create: `sports_watcher/perception/detector.py`, `sports_watcher/perception/ball.py`, `scripts/bench_perceiver.py`
- Modify: `sports_watcher/perception/models.py` (optional manifest fields `output_boxes: str = "dets"`, `output_logits: str = "labels"`, `output_keypoints: str = "keypoints"`)
- Test: `tests/unit/test_detector.py`

**Interfaces:**
- Produces (`detector.py`): `decode_detections(boxes: np.ndarray, logits: np.ndarray, image_hw: tuple[int, int], classes: dict[int, str], threshold: float = 0.3, max_dets: int = 100) -> list[Detection]` (`boxes` `(Q, 4)` cxcywh normalised to the input, `logits` `(Q, C)`; score is the sigmoid of the best class per query, class index maps through `classes`, indices absent from `classes` are dropped, boxes scaled to `image_hw` as xyxy and clipped); `preprocess(image_bgr: np.ndarray, resolution: int) -> np.ndarray` (squash to `resolution` square, RGB, `float32 / 255`, ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`, NCHW); `OnnxDetector(model_path: Path, manifest: ModelManifest, providers: list[str] | None = None)`: `detect(image_bgr) -> list[Detection]`, `resolution: int`; `default_providers() -> list[str]` (`CoreMLExecutionProvider` then `CPUExecutionProvider` when available). `OnnxKeypointModel(model_path, manifest, providers=None)`: `keypoints(image_bgr) -> tuple[np.ndarray, np.ndarray]` (`(K, 2)` pixels and `(K,)` confidences from the query with the highest class score; `output_keypoints` is `(1, Q, K, 3)` normalised x, y, confidence).
- Produces (`ball.py`): `crop_window(center_xy: tuple[float, float], crop_px: int, image_hw: tuple[int, int]) -> tuple[int, int, int, int]` (x0, y0, x1, y1, shifted to stay inside the image, always `crop_px` wide unless the image is smaller); `BallFinder(detector, crop_px: int = 384, ball_class: str = "ball")`: `find(image_bgr, predicted_xy: tuple[float, float] | None, full_dets: list[Detection]) -> Detection | None` (with a prediction: run `detector.detect` on the crop and return the best `ball` detection offset back to image coordinates; otherwise, or when the crop finds nothing, the best `ball` among `full_dets`).
- Produces: `scripts/bench_perceiver.py --models DIR --source CLIP --start 70 --frames 100 [--resolution N]` prints per-stage mean milliseconds (detector, ball crop, tracker, keypoints, team, probe) and the number of frames with a detected ball; results go in the notes file.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_detector.py
import numpy as np
from sports_watcher.perception.ball import BallFinder, crop_window
from sports_watcher.perception.detector import decode_detections, preprocess
from sports_watcher.perception.tracker import Detection

CLASSES = {1: "ball", 2: "player", 3: "referee", 4: "rim"}

def test_decode_scales_boxes_and_maps_classes():
    boxes = np.array([[0.5, 0.5, 0.2, 0.4], [0.1, 0.1, 0.05, 0.05], [0.9, 0.9, 0.1, 0.1]])
    logits = np.full((3, 5), -9.0); logits[0, 2] = 3.0; logits[1, 1] = 0.5; logits[2, 0] = 4.0
    dets = decode_detections(boxes, logits, (1080, 1920), CLASSES, threshold=0.3)
    assert len(dets) == 2
    player = next(d for d in dets if d.cls == "player")
    assert np.allclose(player.box, (1920 * 0.4, 1080 * 0.3, 1920 * 0.6, 1080 * 0.7)) and player.confidence > 0.95
    assert next(d for d in dets if d.cls == "ball").confidence < 0.7

def test_decode_threshold_and_max_dets():
    boxes = np.tile([[0.5, 0.5, 0.1, 0.1]], (10, 1)); logits = np.full((10, 5), -9.0); logits[:, 2] = np.linspace(-1, 3, 10)
    assert len(decode_detections(boxes, logits, (100, 100), CLASSES, threshold=0.5, max_dets=3)) == 3
    assert all(d.confidence >= 0.5 for d in decode_detections(boxes, logits, (100, 100), CLASSES, threshold=0.5))

def test_preprocess_shape_and_normalisation():
    x = preprocess(np.full((1080, 1920, 3), 255, np.uint8), 512)
    assert x.shape == (1, 3, 512, 512) and x.dtype == np.float32
    assert np.allclose(x[0, 0], (1 - 0.485) / 0.229, atol=1e-5)

def test_crop_window_clamps_to_image():
    assert crop_window((960.0, 540.0), 384, (1080, 1920)) == (768, 348, 1152, 732)
    assert crop_window((10.0, 10.0), 384, (1080, 1920)) == (0, 0, 384, 384)
    assert crop_window((1915.0, 1075.0), 384, (1080, 1920)) == (1536, 696, 1920, 1080)

class FakeDetector:
    def __init__(self): self.calls = []
    def detect(self, image):
        self.calls.append(image.shape)
        return [Detection((10.0, 20.0, 22.0, 32.0), "ball", 0.8)] if image.shape[0] == 384 else []

def test_ball_finder_offsets_crop_detection_back():
    d = FakeDetector(); f = BallFinder(d)
    img = np.zeros((1080, 1920, 3), np.uint8)
    det = f.find(img, (960.0, 540.0), [])
    assert d.calls == [(384, 384, 3)] and det.box == (778.0, 368.0, 790.0, 380.0)

def test_ball_finder_falls_back_to_full_frame():
    d = FakeDetector(); f = BallFinder(d)
    full = [Detection((1.0, 1.0, 9.0, 9.0), "ball", 0.4), Detection((0.0, 0.0, 50.0, 100.0), "player", 0.9)]
    assert f.find(np.zeros((1080, 1920, 3), np.uint8), None, full).confidence == 0.4 and d.calls == []
```

- [ ] **Step 2: Run to verify failure, implement, run the suite and lint, commit**

The ONNX session wrappers are covered by Task 17's integration test. Commit: `feat(m1): ONNX detector and keypoint wrappers, ball crop refinement, perception benchmark`.

---

### Task 16: Camera probe, its training script and the labels

**Files:**
- Modify: `sports_watcher/perception/camera_state.py` (add the probe)
- Create: `training/train_camera_probe.py`, `scripts/contact_sheet.py`, `data/labels/camera_state.jsonl` (gitignored; kept on disk)
- Test: `tests/unit/test_camera_state_tracker.py` (append)

**Interfaces:**
- Produces: `frame_features(inliers: int, n_players: int, median_box_h_frac: float) -> np.ndarray` = `[inliers / 33, n_players / 12, median_box_h_frac]` as `float32`; `CameraProbe(coef: np.ndarray, intercept: np.ndarray, classes: list[str])`: `predict(embedding: np.ndarray, features: np.ndarray) -> tuple[CameraHint, float]` (softmax over `coef @ concat(embedding, features) + intercept`), `save(path)`, `CameraProbe.load(path) -> CameraProbe` (JSON with `coef`, `intercept`, `classes`, `embedding_dim`).
- Produces: `training/train_camera_probe.py --labels data/labels/camera_state.jsonl --clips data/clips --models models --out models/camera/probe.json --fps 2 --holdout pba_ros_cvg_2500_2700.mp4`: decodes frames inside each labelled segment with PyAV at `--fps`, computes the SigLIP embedding and `frame_features` with the detector and court mapper from `models`, fits `sklearn.linear_model.LogisticRegression(max_iter=2000, C=1.0)`, prints held-out accuracy per class for the `--holdout` clip, saves the probe. Label lines: `{"clip": "<file name>", "start": <s>, "end": <s>, "state": "main|replay|closeup|graphics"}`.
- Produces: `scripts/contact_sheet.py --source CLIP --fps 1 --tiles 10x6 --out DIR [--start S --duration D]` (ffmpeg `fps=<fps>,scale=320:-1,tile=<tiles>`; tile index times `1/fps` is the time within the sheet).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_camera_state_tracker.py (append)
import numpy as np
from sports_watcher.perception.camera_state import CameraProbe, frame_features

def test_frame_features_scaling():
    assert np.allclose(frame_features(33, 12, 0.25), [1.0, 1.0, 0.25])

def test_probe_predicts_argmax_and_round_trips(tmp_path):
    coef = np.zeros((4, 5)); coef[2, 0] = 5.0; coef[3, 4] = 5.0
    p = CameraProbe(coef, np.zeros(4), ["main", "replay", "closeup", "graphics"])
    label, conf = p.predict(np.array([1.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    assert label == "closeup" and conf > 0.9
    assert p.predict(np.zeros(2), np.array([0.0, 0.0, 1.0]))[0] == "graphics"
    p.save(tmp_path / "probe.json")
    assert CameraProbe.load(tmp_path / "probe.json").predict(np.array([1.0, 0.0]), np.zeros(3))[0] == "closeup"
```

- [ ] **Step 2: Run to verify failure, implement the probe and the two scripts, run the suite and lint**

- [ ] **Step 3: Label the frames**

Make contact sheets for `data/clips/pba_ros_cvg_2700_3100.mp4` and `data/clips/lal_gsw_060_150.mp4` at 1 fps, write `data/labels/camera_state.jsonl` with one segment per camera stretch (the earlier study gives the coarse structure of the PBA clip: stats graphic 0 to 17 s, wipe, timeout huddle 20 to 39 s, close-ups 40 to 67 s, main camera 68 to 157 s with brief close-ups, free-throw graphic about 176 to 189 s, top-down replay 190 to 191 s, close-ups 192 to 209 s, play 210 to 239 s; refine every boundary on the sheets). Label `data/clips/pba_ros_cvg_2500_2700.mp4` the same way into `data/golden/pba_ros_cvg_q1.camera.jsonl` (committed, Task 18 uses it) and keep that clip as the held-out set.

- [ ] **Step 4: Commit**

Commit: `feat(m1): camera-state probe, probe training script and contact-sheet tool`. The probe itself is trained in Task 17 once the SigLIP export and the detector exist.

---

### Task 17: The ONNX perceiver, court template and the cache builder

**Files:**
- Create: `sports_watcher/perception/onnx_perceiver.py`, `sports_watcher/perception/offline.py`, `scripts/perceive.py`, `configs/court_keypoints/basketball-court-detection-2.yaml`
- Test: `tests/unit/test_offline_cache.py`, `tests/unit/test_court_template_config.py`, `tests/integration/test_onnx_perceiver.py`

**Interfaces:**
- Produces (`onnx_perceiver.py`): `OnnxPerceiver(models: ModelSet, league: LeagueConfig, home_color: str | None, away_color: str | None, template_path: Path | None = None, keypoint_every_s: float = 0.2, probe_every_s: float = 0.2, providers: list[str] | None = None)`: `version: str` (`models.version`), `perceive(frame) -> VisionFrame`, `reset()`, `timings: dict[str, float]` (running mean seconds per stage). Per frame: detector; player and referee detections to the `Tracker`; the rim is the highest-confidence `rim` detection, held as the median box of the last second and dropped after 2 s without one; `BallFilter.predict` then `BallFinder.find`, `BallFilter.update`, `BallFilter.current`; keypoints every `keypoint_every_s` into the `CourtMapper`, otherwise `current`; `feet_xy` through `image_to_court` when a homography exists; one crop per track into `TeamAssigner.observe` and `team_of` per track; every `probe_every_s` the SigLIP full-frame embedding plus `frame_features` into the `CameraProbe`, the last hint repeated in between; missing optional models degrade: no court model means `homography None`, no SigLIP means teams `unknown` and hint `main` at confidence 0.5, no probe means hint `main` at 0.5. `template_path` defaults to `configs/court_keypoints/<court manifest dataset name>.yaml` under `Settings().configs_dir`.
- Produces (`offline.py`): `build_cache(spec: SourceSpec, perceiver: Perceiver, path: Path, every_n: int | None = None, progress: Callable[[int], None] | None = None) -> CacheStats` (`CacheStats(frames_read: int, frames_perceived: int, seconds: float)`; wraps `perceiver` in `CachedPerceiver`, iterates `open_source(spec).frames()`, perceives every `every_n`-th frame with `every_n` derived as the watcher does when `None`).
- Produces: `scripts/perceive.py --source CLIP [--models DIR] [--out DIR] [--home-color C --away-color C] [--every N]` runs `build_cache` and prints the stats and `OnnxPerceiver.timings`.
- Produces: the court template yaml with the 33 keypoint names of `models/court/manifest.json` and their positions as fractions of court length and width (named from the dataset, positions from the FIBA court drawing: corners, sideline and baseline midpoints, key corners and free-throw line ends, three-point arc ends and tops, centre circle points).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_offline_cache.py
from pathlib import Path
from sports_watcher.ingest.base import SourceSpec
from sports_watcher.ingest.factory import register_source
from sports_watcher.perception.cache import CachedPerceiver
from sports_watcher.perception.offline import build_cache
from sports_watcher.perception.perceiver import ScriptedPerceiver
from tests.helpers.sources import StubSource

def test_build_cache_perceives_every_second_frame_at_sixty_fps(tmp_path):
    register_source("stub", lambda spec: StubSource(n=20, fps=60))
    p = ScriptedPerceiver([], version="v9")
    stats = build_cache(SourceSpec(kind="stub"), p, tmp_path / "c.v9.jsonl")
    assert (stats.frames_read, stats.frames_perceived) == (20, 10) and p.calls == list(range(0, 20, 2))
    assert CachedPerceiver(None, tmp_path / "c.v9.jsonl").version == "v9"
```

```python
# tests/unit/test_court_template_config.py
from pathlib import Path
from sports_watcher.perception.court import load_template
from tests.helpers.leagues import fixture_league

def test_court_template_has_33_unique_points_inside_the_court():
    names, xy = load_template(Path("configs/court_keypoints/basketball-court-detection-2.yaml"), fixture_league("pba"))
    assert len(names) == 33 and len(set(names)) == 33
    assert (xy[:, 0] >= 0).all() and (xy[:, 0] <= 28.0).all() and (xy[:, 1] >= 0).all() and (xy[:, 1] <= 15.0).all()
```

```python
# tests/integration/test_onnx_perceiver.py
import pytest
from pathlib import Path
MODELS, CLIP = Path("models"), Path("data/clips/pba_ros_cvg_2700_3100.mp4")

@pytest.mark.integration
def test_onnx_perceiver_on_main_camera_frames():
    from sports_watcher.perception.models import ModelSet
    ms = ModelSet.load(MODELS)
    if ms is None or not CLIP.exists():
        pytest.skip("needs models/ and the PBA clip")
    from sports_watcher.config import load_league
    from sports_watcher.ingest.base import SourceSpec
    from sports_watcher.ingest.factory import open_source
    from sports_watcher.perception.onnx_perceiver import OnnxPerceiver
    p = OnnxPerceiver(ms, load_league("pba"), "white", "navy")
    src = open_source(SourceSpec(kind="file", path=str(CLIP))); src.open()
    frames = [f for f in src.frames() if 90.0 <= f.pts_s < 90.5][::2]       # 15 perception frames of live play
    out = [p.perceive(f) for f in frames]
    assert len(out[-1].tracks) >= 6 and out[-1].rim is not None
    assert any(vf.homography is not None and vf.homography.confidence >= 0.4 for vf in out) or ms.court is None
    assert any(vf.ball is not None and vf.ball.source == "detected" for vf in out)
```

- [ ] **Step 2: Run to verify failure, implement, author the template, run the suite and lint**

Run the integration test with `./.venv/bin/python -m pytest tests/integration/test_onnx_perceiver.py -q` once weights are in `models/` (Task 3's Colab output copied in, SigLIP exported locally with `training/export_siglip.py`, probe trained with `training/train_camera_probe.py`).

- [ ] **Step 3: Measure and decide the detector size**

Run `scripts/bench_perceiver.py --models models --source data/clips/pba_ros_cvg_2700_3100.mp4 --start 70 --frames 100` for the trained detector; if two sizes were trained, for both. Record the per-stage milliseconds and ball-detected frame count in the notes file and keep the manifest of the chosen one in `models/detector`.

- [ ] **Step 4: Commit**

```bash
git add sports_watcher/perception/onnx_perceiver.py sports_watcher/perception/offline.py scripts/perceive.py configs/court_keypoints tests
git commit -m "feat(m1): ONNX perceiver, court keypoint template, offline cache builder"
```

---

### Task 18: Golden clips, truth files and the golden tests

**Files:**
- Create: `data/golden/pba_ros_cvg_vision.mp4` (gitignored), `data/golden/pba_ros_cvg_vision.meta.json`, `data/golden/pba_ros_cvg_vision.truth.jsonl`, `data/golden/pba_ros_cvg_q1.camera.jsonl`, `tests/golden/test_golden_camera_state.py`
- Modify: `tests/golden/test_golden_clips.py`, `data/golden/README.md`, `data/golden/pba_ros_cvg_q1.meta.json`

**Interfaces:**
- Meta gains `"checks": {"score_change": 0.9, "possession_start": 0.8, "shot_attempt": 0.8}` (type to minimum precision and recall) and `"home_color"`, `"away_color"`. The M0 clip's meta gets `"checks": {"score_change": 0.9}`.
- `test_golden_clips.py`: builds the watcher with `build_perceiver(Settings(), league, clip, home_color, away_color)` and `ocr_every_n_frames` from meta; scores every type in `checks`; vision types are skipped with reason `no perception cache` when the perceiver is `None`; asserts additionally that every confirmed `shot_made` has a confirmed `score_change` of the same team within 3 s.
- `test_golden_camera_state.py`: for each clip with a `.camera.jsonl`, runs the watcher with `log_camera_states=True` and asserts `camera_state_accuracy(...) >= 0.9` and that no confirmed `possession_start`, `shot_attempt`, `shot_made`, `rebound` or `turnover` has `t_pts` inside a `closeup`, `replay` or `graphics` segment; skipped without a perceiver.

- [ ] **Step 1: Cut the clip and write the camera truth**

```bash
ffmpeg -y -ss 60 -t 180 -i data/clips/pba_ros_cvg_2700_3100.mp4 -c:v libx264 -crf 18 -an data/golden/pba_ros_cvg_vision.mp4
```

Write `pba_ros_cvg_q1.camera.jsonl` from the Task 16 labels of `pba_ros_cvg_2500_2700.mp4` (same file as `pba_ros_cvg_q1.mp4`).

- [ ] **Step 2: Draft the vision truth**

`scripts/contact_sheet.py --source data/golden/pba_ros_cvg_vision.mp4 --fps 4 --tiles 10x6` for each live stretch; write one line per `possession_start` (team, t_pts of the first clear control), `shot_attempt` (team, t_pts of release, `zone`), `shot_made` or `shot_missed`, and the `score_change` lines, as `TruthEvent` JSON. Team names from the meta (`home` and `away` follow the scoreboard rows: top row away, bottom row home, as the M0 calibration notes). Hand the file to Stephen for a pass against the video before committing it.

- [ ] **Step 3: Write the failing golden tests, build the caches, run**

```bash
./.venv/bin/python scripts/perceive.py --source data/golden/pba_ros_cvg_vision.mp4 --home-color <from meta> --away-color <from meta>
./.venv/bin/python scripts/perceive.py --source data/golden/pba_ros_cvg_q1.mp4 --home-color <from meta> --away-color <from meta>
./.venv/bin/python -m pytest tests/golden -q
```

Expected: both golden tests run (not skipped) and pass the floors. If a floor fails, record the per-type scores in the notes, fix the rule or threshold with a unit test first, rebuild nothing (the cache is perception only) and rerun.

- [ ] **Step 4: Commit**

```bash
git add tests/golden data/golden/README.md data/golden/*.meta.json data/golden/*.truth.jsonl data/golden/*.camera.jsonl
git commit -m "test(m1): golden vision clip truth, camera-state truth and golden checks"
```

---

### Task 19: Weights download, docs, notes and the branch review

**Files:**
- Create: `scripts/download_models.py`, `docs/superpowers/plans/2026-10-31-sports-watcher-m1-vision.notes.md`
- Modify: `README.md`, `CLAUDE.md`, `.env.example`

**Interfaces:**
- `scripts/download_models.py [--tag m1-models] [--dest models]`: `gh release download <tag> --pattern 'models-*.zip' --dir <tmp>` then unzip into `--dest`; prints the manual alternative (copy the Colab zip's contents into `models/`) when `gh` is missing or the release has no asset. Publishing the release itself (`gh release create m1-models models-<date>.zip`) is a manual step Stephen runs after reviewing the weights' licences in their manifests.

- [ ] **Step 1: Write the download script and the docs**

`CLAUDE.md`: commands for `scripts/perceive.py`, `scripts/bench_perceiver.py`, `scripts/download_models.py`, the `train` extra and `training/README.md`; layout lines for `perception/` modules, `sports/basketball/{geometry,vision_state,confirmers,rules_*}.py`, `training/`, `data/cache/`, `data/labels/`; the rule "vision rules need the game clock running; free throws and tip-off are the exceptions". `README.md`: the M1 quick start (download or train weights, perceive a clip, run the watcher with colours). `.env.example`: `SW_MODELS_DIR=models`, `SW_CACHE_DIR=data/cache`.

- [ ] **Step 2: Write the notes file**

Rulings made during execution, detector measurements, golden scores per type, and deferred items, in the M0 notes format.

- [ ] **Step 3: Verify and commit**

```bash
./.venv/bin/python -m pytest -q && ./.venv/bin/ruff check .
git add scripts/download_models.py README.md CLAUDE.md .env.example docs/superpowers/plans/2026-10-31-sports-watcher-m1-vision.notes.md
git commit -m "docs(m1): weights download, commands, layout and execution notes"
```

- [ ] **Step 4: Whole-branch review and finish**

Use superpowers:requesting-code-review on `main..feat/m1-vision` against the spec, fix findings test-first, then superpowers:finishing-a-development-branch. Report to the starting session with the commit range, the golden scores and anything deferred.
