# Sports Watcher M0 Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A runnable Python service that ingests a basketball broadcast from a file, URL or YouTube live, reads the scoreboard, emits provisional/confirmed/retracted `score_change` and `period_*` events over an API and JSONL, and answers "first team to score" and "race to N" for a PBA recording.

**Architecture:** One process per game. A `FrameSource` adapter feeds a latest-frame reader; every Nth frame goes to a scoreboard OCR reader with majority voting; a sport plug-in turns scoreboard observations into event drafts; the event engine manages provisional → confirmed/retracted with a TTL and fans events out to sinks; a firsts fold answers the "first X" questions; FastAPI exposes jobs, a websocket and the read model. Vision, audio and the basketball possession rules are later plans (M1, M2) and plug into the same engine.

**Tech Stack:** Python 3.12, PyAV, NumPy, OpenCV headless, pydantic v2 + pydantic-settings, PyYAML, rapidocr-onnxruntime (scoreboard OCR), FastAPI + uvicorn, pytest, ruff. Optional: yt-dlp.

**Spec:** `docs/superpowers/specs/2026-10-31-sports-watcher-v1-design.md` (sections 3, 5.1, 5.2 scoreboard row, 5.4 to 5.9, 6, 7, 8, milestone M0).

## Global Constraints

- Python `>=3.12`; package name `sports_watcher`; project name `sports-watcher`.
- Runtime dependencies must be Apache-2.0, MIT or BSD licensed. No Ultralytics, no boxmot, no GPL packages.
- Events are keyed by `(game_id, seq)`; `seq` is strictly increasing per game. Every provisional event gets a terminal `confirmed` or `retracted` event whose `supersedes` is the provisional's `id`. Provisional TTL default `4.0` s of stream time.
- Latency classes and default behind-live estimates: `FILE` none, `PUSH` 0.5 s, `PULL` 1.0 s, `HLS` 15.0 s, `CAPTURE` 0.2 s.
- League values: PBA court 28.0 × 15.0 m, three-point radius 6.75 m, corner 6.60 m, period 600 s × 4, overtime 300 s, shot clock 24 s. NBA court 28.65 × 15.24 m, radius 7.24 m, corner 6.71 m, period 720 s × 4, overtime 300 s, shot clock 24 s.
- Secrets only in `.env` (gitignored); `.env.example` committed. Model weights and data under `models/` and `data/`, both gitignored.
- Every commit leaves `pytest -q` green and `ruff check .` clean. Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Use `./.venv/bin/python` for every command (`python3 -m venv .venv` once; `./.venv/bin/pip install -e ".[dev]"`).

## Review Focus

1. A single-frame OCR misread (78 read as 73 for one frame) must not produce a confirmed `score_change`; it may produce a provisional that is retracted. Pinned in Task 5 (voting) and Task 7 (retraction).
2. A score that goes down (official correction) must emit a `score_change` with a negative delta and `evidence.note == "correction"`, and the firsts fold must still report the first score correctly. Pinned in Task 6 and Task 8.
3. Clock strings vary: `10:00`, `7:05`, `0:59.8`, `59.8`, `:45`; period strings vary: `1st`, `Q1`, `1`, `OT`, `2OT`. Both parsers are table-tested in Task 5.
4. Stream PTS can jump backwards on an HLS discontinuity; wall-clock mapping must stay monotonic. Pinned in Task 2.
5. End of stream with provisional events pending must end the job in state `ended` and retract every pending event with reason `stream_ended`. Pinned in Task 7 and Task 9.

---

### Task 1: Project skeleton and configuration

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `README.md`, `CLAUDE.md`
- Create: `configs/watcher.yaml`, `configs/leagues/pba.yaml`, `configs/leagues/nba.yaml`
- Create: `sports_watcher/__init__.py`, `sports_watcher/config.py`
- Test: `tests/unit/test_config.py`, `tests/conftest.py` (empty for now)

**Interfaces:**
- Produces: `Settings` (pydantic-settings, env prefix `SW_`, fields `device: str = "cpu"`, `configs_dir: Path = Path("configs")`, `data_dir: Path = Path("data")`, `log_level: str = "INFO"`); `Roi(x, y, w, h)` as fractions of the frame; `BroadcasterConfig(scoreboard_roi: Roi, home_side: Literal["left","right"] = "left", shot_clock_visible: bool = True)`; `LeagueConfig(name, court_length_m, court_width_m, three_point_radius_m, three_point_corner_m, period_length_s, periods, overtime_length_s, shot_clock_s, asr_languages: list[str], broadcasters: dict[str, BroadcasterConfig])`; `load_league(name: str, configs_dir: Path | None = None) -> LeagueConfig`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_config.py
from sports_watcher.config import Settings, load_league

def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.device == "cpu" and s.configs_dir.name == "configs"

def test_load_pba_league():
    pba = load_league("pba")
    assert (pba.court_length_m, pba.court_width_m) == (28.0, 15.0)
    assert pba.shot_clock_s == 24 and pba.period_length_s == 600 and pba.periods == 4
    assert pba.broadcasters["one_sports"].home_side == "left"

def test_load_nba_league():
    nba = load_league("nba")
    assert nba.three_point_radius_m == 7.24 and nba.period_length_s == 720
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/unit/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: sports_watcher`

- [ ] **Step 3: Create `pyproject.toml`, package files and configs**

`pyproject.toml`: build backend `setuptools`; `[project]` name `sports-watcher`, version `0.1.0`, `requires-python = ">=3.12"`, dependencies `av>=14`, `numpy>=1.26`, `opencv-python-headless>=4.10`, `pydantic>=2.7`, `pydantic-settings>=2.3`, `pyyaml>=6`, `fastapi>=0.115`, `uvicorn[standard]>=0.30`, `websockets>=12`, `rapidocr-onnxruntime>=1.3`; extras `dev = ["pytest>=8", "httpx>=0.27", "pillow>=10", "ruff>=0.6"]`, `youtube = ["yt-dlp>=2025.1.1"]`; `[tool.pytest.ini_options] testpaths = ["tests"]`, markers `integration`; `[tool.ruff] line-length = 100, target-version = "py312"`. `.gitignore`: `.venv/`, `__pycache__/`, `*.pyc`, `.env`, `models/`, `data/`, `tmp/`, `.pytest_cache/`, `.ruff_cache/`, `*.egg-info/`. `configs/leagues/pba.yaml` and `nba.yaml` hold the Global Constraints values; PBA `broadcasters.one_sports.scoreboard_roi` starts at `{x: 0.02, y: 0.86, w: 0.40, h: 0.10}` (placeholder until Task 12 calibrates it) with `asr_languages: [en, tl]`; NBA `broadcasters.generic` with the same ROI and `asr_languages: [en]`. `configs/watcher.yaml`: `ocr_every_n_frames: 5`, `provisional_ttl_s: 4.0`, `vote_window: 5`. `README.md` and `CLAUDE.md`: title, one-paragraph purpose, the venv and test commands above, pointer to the spec.

- [ ] **Step 4: Implement `config.py`**

`load_league` reads `<configs_dir>/leagues/<name>.yaml` with `yaml.safe_load` and validates into `LeagueConfig`; raises `FileNotFoundError` with the path when missing.

- [ ] **Step 5: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/unit/test_config.py -v && ./.venv/bin/ruff check .`
Expected: 3 passed, ruff clean

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore .env.example README.md CLAUDE.md configs sports_watcher tests
git commit -m "feat: project skeleton, settings and league configs"
```

---

### Task 2: Stream clock and event model

**Files:**
- Create: `sports_watcher/clock.py`, `sports_watcher/events/__init__.py`, `sports_watcher/events/model.py`
- Test: `tests/unit/test_clock.py`, `tests/unit/test_event_model.py`

**Interfaces:**
- Produces: `LatencyClass(StrEnum)` with `FILE, PUSH, PULL, HLS, CAPTURE`; `BEHIND_LIVE_ESTIMATE_S: dict[LatencyClass, float | None]`; `StreamClock(latency_class)` with `observe(pts_s: float, wall: datetime) -> None`, `wall_for(pts_s: float) -> datetime`, `behind_live_s() -> float | None`.
- Produces: `EventStatus(StrEnum)` `provisional, confirmed, retracted`; `PlayerRef(track_id: int | None, name: str | None, jersey: str | None = None, confidence: float = 0.0)`; `CommentaryCueRef(t_pts: float, kind: str, text: str, confidence: float)`; `Evidence(score_before: tuple[int, int] | None = None, score_after: tuple[int, int] | None = None, points: int | None = None, ball_speed_mps: float | None = None, rim_distance_m: float | None = None, cues: list[CommentaryCueRef] = [], camera_state: str | None = None, detector_confidence: float | None = None, note: str | None = None)`; `SourceInfo(adapter: str, latency_class: LatencyClass)`; `Event` with the spec 5.7 fields plus `emitted_wall: datetime` and `reason: str | None`; methods `Event.confirmed_copy(seq, t_pts, t_wall, emitted_wall, confidence=None) -> Event` and `Event.retracted_copy(seq, t_pts, t_wall, emitted_wall, reason) -> Event`, both setting `supersedes=self.id` and a fresh `id`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_clock.py
def test_wall_for_is_monotonic_across_pts_discontinuity():
    c = StreamClock(LatencyClass.HLS)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    c.observe(100.0, t0); c.observe(101.0, t0 + timedelta(seconds=1))
    c.observe(5.0, t0 + timedelta(seconds=2))          # discontinuity: pts jumped back
    assert c.wall_for(6.0) > c.wall_for(101.0)
    assert c.behind_live_s() == 15.0

# tests/unit/test_event_model.py
def test_confirmed_copy_supersedes_and_roundtrips():
    e = make_event(type="score_change", status=EventStatus.provisional)
    c = e.confirmed_copy(seq=e.seq + 1, t_pts=e.t_pts + 0.2, t_wall=e.t_wall, emitted_wall=e.t_wall)
    assert c.supersedes == e.id and c.id != e.id and c.status == EventStatus.confirmed
    assert Event.model_validate_json(c.model_dump_json()) == c

def test_retracted_copy_carries_reason():
    r = make_event().retracted_copy(seq=2, t_pts=1.0, t_wall=NOW, emitted_wall=NOW, reason="ttl")
    assert r.status == EventStatus.retracted and r.reason == "ttl"
```

`make_event(**overrides)` is a helper in `tests/helpers/events.py` returning a valid provisional `Event` for `game_id="g1"`, `seq=1`, `team="home"`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/unit/test_clock.py tests/unit/test_event_model.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Implement `clock.py`**

`StreamClock` keeps an anchor `(pts, wall)`; `observe` re-anchors when `pts_s < last_pts - 1.0` (discontinuity) so `wall_for` of later pts is later than any earlier wall; `wall_for(pts)` = anchor wall + (pts − anchor pts). `behind_live_s()` returns `BEHIND_LIVE_ESTIMATE_S[latency_class]`.

- [ ] **Step 4: Implement `events/model.py`**

Pydantic v2 models as in the Interfaces block; `Event.id` default `uuid4().hex`; `Literal["home", "away", "unknown"]` for `team`; `model_config = ConfigDict(frozen=True)`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/unit -v && ./.venv/bin/ruff check .`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add sports_watcher/clock.py sports_watcher/events tests/unit/test_clock.py tests/unit/test_event_model.py tests/helpers
git commit -m "feat: stream clock with discontinuity handling and event model"
```

---

### Task 3: Ingest base, file adapter and latest-frame reader

**Files:**
- Create: `sports_watcher/ingest/__init__.py`, `sports_watcher/ingest/base.py`, `sports_watcher/ingest/file.py`, `sports_watcher/ingest/latest_frame.py`
- Modify: `tests/conftest.py` (add `sample_video` fixture)
- Test: `tests/unit/test_ingest_file.py`, `tests/unit/test_latest_frame.py`

**Interfaces:**
- Produces: `Frame(index: int, pts_s: float, wall: datetime, image: np.ndarray)` (BGR uint8, HxWx3); `AudioChunk(pts_s: float, pcm: np.ndarray, sample_rate: int = 16000)` (int16 mono); `SourceSpec(kind: Literal["file","url","youtube","stub"], path: str | None = None, url: str | None = None, realtime: bool = False)`; `FrameSource(Protocol)` with `latency_class`, `name: str`, `open()`, `frames() -> Iterator[Frame]`, `audio() -> Iterator[AudioChunk]`, `close()`; `FileSource(path, realtime=False, audio=True)`; `LatestFrameReader(frames: Iterator[Frame])` with `start() -> Self`, `read(timeout: float = 5.0) -> tuple[bool, Frame | None, int]`, `stop()`, `dropped_total: int`, `finished: bool`.

- [ ] **Step 1: Add the `sample_video` fixture**

`tests/conftest.py`: session-scoped fixture that runs `ffmpeg -y -f lavfi -i testsrc=size=320x240:rate=25 -f lavfi -i sine=frequency=440:sample_rate=16000 -t 2 -pix_fmt yuv420p -c:a aac <tmp>/sample.mp4` and returns the path; `pytest.skip` when `shutil.which("ffmpeg")` is None.

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit/test_ingest_file.py
def test_file_source_yields_frames_with_increasing_pts(sample_video):
    src = FileSource(sample_video); src.open()
    frames = list(src.frames())
    assert 48 <= len(frames) <= 52 and frames[0].image.shape == (240, 320, 3)
    assert all(b.pts_s > a.pts_s for a, b in zip(frames, frames[1:]))
    assert src.latency_class == LatencyClass.FILE

def test_file_source_audio_is_16k_mono_int16(sample_video):
    src = FileSource(sample_video); src.open()
    chunks = list(src.audio())
    total = sum(len(c.pcm) for c in chunks)
    assert chunks[0].sample_rate == 16000 and chunks[0].pcm.dtype == np.int16
    assert 30000 <= total <= 34000

# tests/unit/test_latest_frame.py
def test_latest_frame_reader_drops_when_consumer_is_slow():
    frames = (Frame(i, i / 25, NOW, np.zeros((2, 2, 3), np.uint8)) for i in range(200))
    r = LatestFrameReader(frames).start()
    got = []
    while True:
        ok, f, dropped = r.read(timeout=1.0)
        if not ok: break
        got.append(f.index); time.sleep(0.01)
    assert got[-1] == 199 and r.dropped_total > 0 and r.finished
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/unit/test_ingest_file.py tests/unit/test_latest_frame.py -v`
Expected: FAIL with import errors

- [ ] **Step 4: Implement `base.py`, `file.py`, `latest_frame.py`**

`FileSource.frames()` decodes video stream 0 with PyAV, `frame.to_ndarray(format="bgr24")`, `pts_s = float(frame.pts * stream.time_base)`, `wall = start_wall + pts_s` where `start_wall = datetime.now(UTC)` at `open()`; when `realtime=True` sleep so frames are released no faster than their pts. `audio()` opens the file a second time and uses `av.AudioResampler(format="s16", layout="mono", rate=16000)`. `LatestFrameReader` is the GodsEye design (condition variable, keep newest, count drops) adapted to pull from an iterator on a daemon thread; `finished` becomes true when the iterator is exhausted and the last frame was handed out.

- [ ] **Step 5: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/unit -v && ./.venv/bin/ruff check .`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add sports_watcher/ingest tests/conftest.py tests/unit/test_ingest_file.py tests/unit/test_latest_frame.py
git commit -m "feat: ingest base, file source and latest-frame reader"
```

---

### Task 4: URL and YouTube adapters and the source factory

**Files:**
- Create: `sports_watcher/ingest/url.py`, `sports_watcher/ingest/youtube.py`, `sports_watcher/ingest/factory.py`
- Test: `tests/unit/test_ingest_url.py`, `tests/unit/test_ingest_factory.py`

**Interfaces:**
- Consumes: `FrameSource`, `Frame`, `AudioChunk`, `SourceSpec`, `LatencyClass`, `FileSource`.
- Produces: `latency_class_for(url: str) -> LatencyClass` (`rtsp` → PULL; `rtmp`, `srt`, `udp` → PUSH; `http`/`https` → HLS; anything else → FILE); `options_for(url: str) -> dict[str, str]` (base `{"fflags": "nobuffer", "flags": "low_delay", "probesize": "500000", "analyzeduration": "1000000"}`, plus `rtsp_transport: tcp` for rtsp, plus `live_start_index: -1` for http(s)); `UrlSource(url: str, options: dict[str, str] | None = None)` sharing the decode loop with `FileSource` (factor it into `ingest/_decode.py` `decode_frames(container, start_wall)` and `decode_audio(container)`); `resolve_with_ytdlp(watch_url: str) -> str` (runs `yt-dlp -g -f "best[protocol^=m3u8]/best" <url>`, returns first line, raises `RuntimeError` with stderr on failure); `YouTubeSource(watch_url: str, resolver: Callable[[str], str] = resolve_with_ytdlp)` with `latency_class = HLS`; `register_source(kind: str, factory: Callable[[SourceSpec], FrameSource]) -> None`; `open_source(spec: SourceSpec) -> FrameSource` (opens it before returning).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_ingest_url.py
@pytest.mark.parametrize("url,cls", [
    ("rtsp://h/x", LatencyClass.PULL), ("srt://h:9000", LatencyClass.PUSH),
    ("rtmp://h/live", LatencyClass.PUSH), ("udp://@:1234", LatencyClass.PUSH),
    ("https://h/live.m3u8", LatencyClass.HLS), ("/tmp/a.mp4", LatencyClass.FILE)])
def test_latency_class_for(url, cls): assert latency_class_for(url) == cls

def test_options_for_rtsp_and_hls():
    assert options_for("rtsp://h/x")["rtsp_transport"] == "tcp"
    assert options_for("https://h/a.m3u8")["live_start_index"] == "-1"
    assert "rtsp_transport" not in options_for("https://h/a.m3u8")

def test_url_source_reads_local_path(sample_video):
    src = UrlSource(str(sample_video)); src.open()
    assert len(list(src.frames())) >= 48

def test_youtube_source_uses_injected_resolver(sample_video):
    src = YouTubeSource("https://youtube.com/watch?v=abc", resolver=lambda u: str(sample_video))
    assert src.latency_class == LatencyClass.HLS and src.url == str(sample_video)

# tests/unit/test_ingest_factory.py
def test_factory_opens_file_and_custom_kind(sample_video):
    assert isinstance(open_source(SourceSpec(kind="file", path=str(sample_video))), FileSource)
    register_source("stub", lambda spec: StubSource())
    assert isinstance(open_source(SourceSpec(kind="stub")), StubSource)
```

`StubSource` lives in `tests/helpers/sources.py`: yields `n` black frames at 25 fps from a list and no audio; used again in Tasks 9 and 10.

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/unit/test_ingest_url.py tests/unit/test_ingest_factory.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Implement `url.py`, `youtube.py`, `factory.py` and refactor `file.py` onto `_decode.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/unit -v && ./.venv/bin/ruff check .`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add sports_watcher/ingest tests
git commit -m "feat: URL and YouTube sources with low-delay options and source factory"
```

---

### Task 5: Scoreboard reader

**Files:**
- Create: `sports_watcher/perception/__init__.py`, `sports_watcher/perception/scoreboard.py`
- Test: `tests/unit/test_scoreboard_parse.py`, `tests/unit/test_scoreboard_reader.py`, `tests/integration/test_scoreboard_ocr.py`, `tests/helpers/scoreboard_images.py`, `tests/helpers/text_readers.py`

**Interfaces:**
- Consumes: `Roi`, `BroadcasterConfig`.
- Produces (test helpers, reused by Tasks 9 and 10): `boxes(home: int, away: int, clock: str, period: str) -> list[TextBox]` in `tests/helpers/scoreboard_images.py` building boxes laid out as in the parse test; `SeqTextReader(seq: list[list[TextBox]])` returning entries in order, and `ScriptedTextReader(by_frame: dict[int, list[TextBox]])` returning the entry with the largest key not greater than the number of calls so far, both in `tests/helpers/text_readers.py`; constants `BLANK` (a 80x640 black image), `B1`, `B2` (left and right score bboxes) and `BROADCASTER` (a `BroadcasterConfig` with a full-frame ROI).
- Produces: `ScoreboardState(home: int | None, away: int | None, clock_s: float | None, period: int | None, shot_clock_s: float | None, confidence: float, raw: list[str])`; `TextBox = tuple[str, tuple[int, int, int, int], float]` (text, bbox xywh in the crop, confidence); `TextReader(Protocol).read(image_bgr) -> list[TextBox]`; `RapidOcrTextReader()` wrapping `rapidocr_onnxruntime.RapidOCR`; `crop_roi(image, roi) -> np.ndarray`; `parse_clock(text) -> float | None`; `parse_period(text) -> int | None`; `parse_scoreboard(boxes: list[TextBox], home_side: str, shot_clock_visible: bool) -> ScoreboardState`; `ScoreboardReader(reader: TextReader, broadcaster: BroadcasterConfig, vote_window: int = 5)` with `read(frame_bgr) -> ScoreboardState` returning the majority vote over the last `vote_window` parses (per field; `confidence` = agreement fraction of the winning tuple).

- [ ] **Step 1: Write the failing parser tests**

```python
@pytest.mark.parametrize("s,v", [("10:00", 600.0), ("7:05", 425.0), ("0:59.8", 59.8),
                                 ("59.8", 59.8), (":45", 45.0), ("78", None), ("Q3", None)])
def test_parse_clock(s, v): assert parse_clock(s) == v

@pytest.mark.parametrize("s,v", [("1st", 1), ("Q1", 1), ("1", 1), ("2nd", 2), ("4TH", 4),
                                 ("OT", 5), ("2OT", 6), ("78", None)])
def test_parse_period(s, v): assert parse_period(s) == v

def test_parse_scoreboard_home_left():
    boxes = [("78", (10, 0, 40, 30), .9), ("65", (200, 0, 40, 30), .9),
             ("7:31", (100, 0, 60, 20), .9), ("3rd", (100, 30, 30, 15), .9), ("14", (300, 0, 20, 15), .9)]
    st = parse_scoreboard(boxes, home_side="left", shot_clock_visible=True)
    assert (st.home, st.away, st.clock_s, st.period, st.shot_clock_s) == (78, 65, 451.0, 3, 14.0)
```

Scores are the two integer boxes with the largest text height; the one with the smaller x is the `home_side` team. A standalone integer ≤ 24 that is not a score is the shot clock when `shot_clock_visible`.

- [ ] **Step 2: Write the failing voting test**

```python
def test_reader_majority_vote_ignores_single_misread():
    seq = [[("78",B1,.9),("65",B2,.9)]]*2 + [[("73",B1,.9),("65",B2,.9)]] + [[("78",B1,.9),("65",B2,.9)]]*2
    r = ScoreboardReader(SeqTextReader(seq), broadcaster=BROADCASTER, vote_window=5)
    states = [r.read(BLANK) for _ in seq]
    assert states[2].home == 78 and states[-1].home == 78 and states[-1].confidence == 0.8
```

- [ ] **Step 3: Write the failing OCR integration test**

`tests/helpers/scoreboard_images.py`: `render_scoreboard(home: int, away: int, clock: str, period: str, size=(640, 80)) -> np.ndarray` draws white text on a dark bar with PIL's default font at height 40 px. Test (marked `integration`, skipped if `rapidocr_onnxruntime` import fails): `RapidOcrTextReader` + `parse_scoreboard` on `render_scoreboard(78, 65, "7:31", "3rd")` returns home 78, away 65, clock 451, period 3.

- [ ] **Step 4: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/unit/test_scoreboard_parse.py tests/unit/test_scoreboard_reader.py tests/integration -v`
Expected: FAIL with import errors

- [ ] **Step 5: Implement `scoreboard.py`**

`parse_clock` regex `^(\d{1,2}):(\d{2})(\.\d)?$`, `^:?(\d{1,2})(\.\d)?$` for sub-minute; `parse_period` regex `^(\d)(st|nd|rd|th)?$`, `^Q(\d)$`, `^(\d?)OT$` (OT → 5, nOT → 4 + n). Voting keeps a `deque(maxlen=vote_window)` of parsed states and takes the mode per field.

- [ ] **Step 6: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests -v && ./.venv/bin/ruff check .`
Expected: all passed (integration may skip on a machine without the OCR package)

- [ ] **Step 7: Commit**

```bash
git add sports_watcher/perception tests
git commit -m "feat: scoreboard OCR reader with clock and period parsing and majority voting"
```

---

### Task 6: Game state, sport plug-in interface and basketball scoreboard rules

**Files:**
- Create: `sports_watcher/state/__init__.py`, `sports_watcher/state/game_state.py`, `sports_watcher/sports/__init__.py`, `sports_watcher/sports/base.py`, `sports_watcher/sports/basketball/__init__.py`, `sports_watcher/sports/basketball/plugin.py`, `sports_watcher/sports/basketball/rules.py`
- Test: `tests/unit/test_basketball_scoreboard_rules.py`

**Interfaces:**
- Consumes: `ScoreboardState`, `LeagueConfig`, `Evidence`.
- Produces: `GameState` dataclass (`period: int | None = None`, `clock_s: float | None = None`, `shot_clock_s: float | None = None`, `home_score: int = 0`, `away_score: int = 0`, `scoreboard_confidence: float = 0.0`, `camera_state: str = "main"`, `possession: Literal["home","away","unknown"] = "unknown"`, `t_pts: float = 0.0`; method `score() -> tuple[int, int]`); `Observation(frame_index: int, t_pts: float, t_wall: datetime, scoreboard: ScoreboardState | None = None, camera_state: str = "main")`; `Verdict = Literal["pending", "confirmed", "retracted"]`; `Confirmer = Callable[[EventDraft, GameState, Observation], Verdict]`; `EventDraft(type: str, team: str, t_pts: float, evidence: Evidence, confidence: float, confirmer: Confirmer, ttl_s: float = 4.0, player: PlayerRef | None = None, court_xy: tuple[float, float] | None = None, zone: str | None = None)`; `Rule = Callable[[GameState, GameState, Observation], list[EventDraft]]` (prev state, new state, observation); `SportPlugin(Protocol)` with `name: str`, `league: LeagueConfig`, `update_state(state: GameState, obs: Observation) -> GameState`, `rules() -> list[Rule]`; `BasketballPlugin(league: LeagueConfig)`; rules `score_change_rule`, `period_start_rule`, `period_end_rule`; confirmer factory `scoreboard_agrees(field: str, expected, reads: int = 2) -> Confirmer`.

- [ ] **Step 1: Write the failing tests**

```python
def run(seq):  # seq: list[(t_pts, home, away, clock_s, period)] -> list[EventDraft]
    plugin = BasketballPlugin(load_league("pba")); state = GameState(); out = []
    for t, h, a, c, p in seq:
        obs = Observation(0, t, NOW, ScoreboardState(home=h, away=a, clock_s=c, period=p, shot_clock_s=None, confidence=1.0, raw=[]))
        new = plugin.update_state(state, obs)
        for rule in plugin.rules(): out += rule(state, new, obs)
        state = new
    return out

def test_home_two_points_emits_score_change():
    d = run([(0, 0, 0, 600, 1), (1, 2, 0, 598, 1)])
    assert [x.type for x in d] == ["period_start", "score_change"]
    sc = d[-1]; assert sc.team == "home" and sc.evidence.points == 2 and sc.evidence.score_after == (2, 0)

def test_negative_delta_is_a_correction():
    sc = [x for x in run([(0, 10, 8, 500, 2), (1, 8, 8, 499, 2)]) if x.type == "score_change"][0]
    assert sc.evidence.points == -2 and sc.evidence.note == "correction"

def test_period_boundaries_including_overtime():
    d = run([(0, 50, 50, 0.0, 4), (2, 50, 50, 0.0, 4), (3, 50, 50, 300.0, 5)])
    assert [x.type for x in d] == ["period_end", "period_start"]
    assert d[-1].evidence.note == "overtime"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/unit/test_basketball_scoreboard_rules.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Implement `game_state.py`, `sports/base.py`, basketball `plugin.py` and `rules.py`**

`update_state` copies the state and applies the scoreboard fields when `obs.scoreboard` is present and `confidence >= 0.6`. `score_change_rule` fires when `new.score() != prev.score()` (skip when prev is the initial zero state and new is also zero); team is the side whose number changed (both changed → `unknown`); `points` = delta of that side; negative → `note="correction"`; confirmer `scoreboard_agrees("score", new.score(), reads=2)`. `period_start_rule` fires on the first observation with a period, on period increment, or on clock reset to `period_length_s` (or `overtime_length_s` when period > periods, with `note="overtime"`); `period_end_rule` fires when clock reaches 0.0 and stays 0.0 on the next observation at least 1 s later, once per period. `scoreboard_agrees` returns `confirmed` after `reads` later observations whose field equals `expected`, `retracted` when an observation disagrees, else `pending`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/unit -v && ./.venv/bin/ruff check .`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add sports_watcher/state sports_watcher/sports tests/unit/test_basketball_scoreboard_rules.py
git commit -m "feat: game state, sport plug-in interface and basketball scoreboard rules"
```

---

### Task 7: Event engine with provisional lifecycle, and sinks

**Files:**
- Create: `sports_watcher/events/engine.py`, `sports_watcher/events/fuser.py`, `sports_watcher/events/sinks.py`
- Test: `tests/unit/test_event_engine.py`, `tests/unit/test_sinks.py`

**Interfaces:**
- Consumes: `SportPlugin`, `GameState`, `Observation`, `EventDraft`, `Event`, `StreamClock`, `SourceInfo`.
- Produces: `PendingEvent(event: Event, draft: EventDraft, deadline_pts: float)`; `resolve_pending(pending: list[PendingEvent], state: GameState, obs: Observation, now_pts: float) -> tuple[list[tuple[PendingEvent, Verdict | "ttl"]], list[PendingEvent]]` (resolved, still pending); `EventSink(Protocol)` with `emit(event: Event) -> None` and `close() -> None`; `MemorySink()` (`.events: list[Event]`), `JsonlSink(path: Path)`, `QueueSink(queue: queue.Queue)`, `Broadcaster()` with `subscribe() -> queue.Queue`, `unsubscribe(q)`, `emit`; `EventEngine(plugin: SportPlugin, game_id: str, source: SourceInfo, clock: StreamClock, ttl_s: float = 4.0, sinks: Sequence[EventSink] = ())` with `step(obs: Observation) -> list[Event]`, `finish(t_pts: float, t_wall: datetime) -> list[Event]`, `state: GameState`, `seq: int`.

- [ ] **Step 1: Write the failing tests**

```python
def test_provisional_then_confirmed_with_supersedes(engine, obs_seq):
    e1 = engine.step(obs_seq[0]); e2 = engine.step(obs_seq[1])   # score 0-0 then 2-0
    prov = [e for e in e2 if e.type == "score_change"][0]
    assert prov.status == "provisional"
    later = engine.step(obs_seq[2]) + engine.step(obs_seq[3])      # two agreeing reads
    conf = [e for e in later if e.supersedes == prov.id][0]
    assert conf.status == "confirmed" and conf.seq > prov.seq

def test_retracted_when_scoreboard_flips_back(engine):
    ...  # 0-0, 2-0 (provisional), 0-0 -> retracted with reason "scoreboard_disagrees"

def test_ttl_expiry_retracts(engine):
    ...  # 2-0 provisional, then observations without scoreboard for 4.5 s of pts -> retracted reason "ttl"

def test_finish_retracts_pending_with_stream_ended(engine):
    ...  # provisional pending, engine.finish(...) -> retracted reason "stream_ended"

def test_seq_strictly_increasing_and_jsonl_sink_writes_every_event(tmp_path):
    ...  # MemorySink seqs == sorted unique; JsonlSink file has len(events) lines
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/unit/test_event_engine.py tests/unit/test_sinks.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Implement `fuser.py`, `sinks.py`, `engine.py`**

`step`: `new = plugin.update_state(state, obs)`; drafts from all rules; each draft becomes a provisional `Event` (`emitted_wall = datetime.now(UTC)`, `t_wall = clock.wall_for(obs.t_pts)`, `behind_live_s = clock.behind_live_s()`) appended to pending with `deadline_pts = obs.t_pts + draft.ttl_s`; then `resolve_pending` over existing pending (confirmer verdicts, then TTL by `now_pts >= deadline_pts`); emit all through every sink; return them. `finish` retracts all pending with `reason="stream_ended"` and closes sinks. `Broadcaster.emit` puts into each subscriber queue with `put_nowait`, dropping on `Full`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/unit -v && ./.venv/bin/ruff check .`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add sports_watcher/events tests/unit/test_event_engine.py tests/unit/test_sinks.py
git commit -m "feat: event engine with provisional, confirmed and retracted lifecycle and sinks"
```

---

### Task 8: Firsts read model

**Files:**
- Create: `sports_watcher/events/firsts.py`
- Test: `tests/unit/test_firsts.py`

**Interfaces:**
- Consumes: `Event`, `LeagueConfig`.
- Produces: `FirstRef(team: str, player: PlayerRef | None, t_pts: float, seq: int)`; `FirstScore(FirstRef)` plus `kind: Literal["ft", "two", "three", "unknown"]`; `PossessionResult(period: int | None, team: str, result: str, t_pts: float)`; `Firsts(first_possession: str | None = None, tip_off_winner: str | None = None, first_shot_attempt: FirstRef | None = None, first_score: FirstScore | None = None, first_three: FirstRef | None = None, race_to: dict[int, dict[int, str | None]] = {}, possessions: list[PossessionResult] = [])` where `race_to[period][10|20]` is the team that reached that period's running total first; `fold_firsts(events: Iterable[Event], league: LeagueConfig) -> Firsts` using confirmed events only (a provisional superseded by a retraction counts for nothing).

- [ ] **Step 1: Write the failing tests**

```python
def test_first_score_and_race_to_from_confirmed_events():
    evs = confirmed_scores([(1, "home", 2, (2, 0)), (2, "away", 3, (2, 3)), (3, "home", 1, (3, 3)),
                            (4, "away", 2, (3, 5)), (5, "home", 3, (6, 5))] + ten_more_home_twos, period=1)
    f = fold_firsts(evs, PBA)
    assert f.first_score.team == "home" and f.first_score.kind == "two"
    assert f.first_three.team == "away"
    assert f.race_to[1][10] == "home" and f.race_to[1][20] in ("home", None)

def test_retracted_and_provisional_scores_are_ignored():
    prov, retr = provisional_and_retraction(team="away", points=3)
    f = fold_firsts([prov, retr], PBA)
    assert f.first_score is None and f.first_three is None

def test_correction_does_not_become_first_score():
    evs = confirmed_scores([(1, "home", 2, (2, 0)), (2, "home", -2, (0, 0)), (3, "away", 2, (0, 2))], period=1)
    assert fold_firsts(evs, PBA).first_score.team == "home"   # first score stands as reported; correction is recorded
    assert fold_firsts(evs, PBA).race_to[1][10] is None
```

Decision pinned by the last test: corrections adjust running totals used for `race_to` but do not rewrite `first_score`; the spec treats the firsts as what was observed, with retractions as the correction mechanism.

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/unit/test_firsts.py -v`
Expected: FAIL with import error

- [ ] **Step 3: Implement `firsts.py`**

Kind from `evidence.points`: 1 → `ft`, 2 → `two`, 3 → `three`, else `unknown`. Running totals per period reset at `period_start`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/unit -v && ./.venv/bin/ruff check .`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add sports_watcher/events/firsts.py tests/unit/test_firsts.py
git commit -m "feat: firsts read model folded from confirmed events"
```

---

### Task 9: Game watcher runner and CLI

**Files:**
- Create: `sports_watcher/watcher.py`, `scripts/run_watch.py`, `tests/helpers/synth_video.py`
- Test: `tests/unit/test_watcher.py`, `tests/integration/test_watcher_e2e.py`

**Interfaces:**
- Consumes: `open_source`, `LatestFrameReader`, `ScoreboardReader`, `TextReader`, `BasketballPlugin`, `EventEngine`, sinks, `StreamClock`, `fold_firsts`, `Settings`, `LeagueConfig`.
- Produces: `WatcherStatus(state: Literal["starting","running","ended","failed"], frames_read: int, frames_dropped: int, frames_processed: int, reconnects: int, t_pts: float, period: int | None, clock_s: float | None, score: tuple[int, int], events_emitted: int, error: str | None)`; `GameWatcher(spec: SourceSpec, league: LeagueConfig, broadcaster: str, game_id: str, home: str, away: str, sinks: Sequence[EventSink] = (), text_reader: TextReader | None = None, broadcaster_config: BroadcasterConfig | None = None, ocr_every_n_frames: int = 5, ttl_s: float = 4.0, max_reconnects: int = 5)` with `run() -> None` (blocking), `start_in_thread() -> threading.Thread`, `stop() -> None`, `status() -> WatcherStatus`, `events: list[Event]`, `events_since(seq: int) -> list[Event]`, `firsts() -> Firsts`, `broadcaster: Broadcaster`.
- Produces: `write_scoreboard_video(path: Path, timeline: list[tuple[float, int, int, str, str]], duration_s: float, fps: int = 25, size: tuple[int, int] = (640, 360)) -> Roi` drawing a scoreboard bar in the bottom-left and returning its ROI; `timeline` entries are `(t_s, home, away, clock, period)` taking effect from `t_s`.

- [ ] **Step 1: Write the failing unit test with a fake text reader**

```python
def test_watcher_emits_confirmed_score_change_and_ends(sample_video):
    reader = ScriptedTextReader({0: boxes(0, 0, "10:00", "1st"), 20: boxes(2, 0, "9:50", "1st")})  # by frame index
    w = GameWatcher(SourceSpec(kind="file", path=str(sample_video)), load_league("pba"), "one_sports",
                    game_id="g1", home="TNT", away="Magnolia", text_reader=reader, ocr_every_n_frames=1)
    w.run()
    types = [(e.type, e.status) for e in w.events]
    assert ("score_change", "confirmed") in types and w.status().state == "ended"
    assert w.firsts().first_score.team == "home"
    assert not any(e.status == "provisional" and e.id not in {x.supersedes for x in w.events} for e in w.events)
```

- [ ] **Step 2: Write the failing integration test with real OCR**

`write_scoreboard_video(tmp/"synth.mp4", [(0, 0, 0, "10:00", "1st"), (1.0, 2, 0, "9:50", "1st"), (2.0, 2, 3, "9:40", "1st")], duration_s=3.0)`; run `GameWatcher` with the default `RapidOcrTextReader` and a `BroadcasterConfig` whose ROI is the returned one; assert two confirmed `score_change` events with `t_pts` within 0.5 s of 1.0 and 2.0, and `first_score.team == "home"`. Marked `integration`.

- [ ] **Step 2b: Write the failing reconnect test**

```python
def test_watcher_reconnects_url_sources_and_emits_stream_gap():
    attempts = iter([FailingStubSource(fail_after=10), StubSource(n=20)])
    register_source("stub", lambda spec: next(attempts))
    w = GameWatcher(SourceSpec(kind="stub", url="rtsp://x"), load_league("pba"), "one_sports", game_id="g2",
                    home="A", away="B", text_reader=ScriptedTextReader({}), max_reconnects=2)
    w.run()
    assert w.status().reconnects == 1 and w.status().state == "ended"
    assert [e.type for e in w.events if e.type == "stream_gap"] == ["stream_gap"]
```

`FailingStubSource` (in `tests/helpers/sources.py`) raises `OSError("connection reset")` from `frames()` after `fail_after` frames. Reconnect applies only when `spec.kind` is `url`, `youtube` or `stub` with a `url`; backoff 1, 2, 4, 8 s capped at 8, at most `max_reconnects`; each reconnect emits a `stream_gap` event (team `unknown`, status `confirmed`, `evidence.note` = the error text) and bumps `reconnects`. A `file` source never reconnects.

- [ ] **Step 3: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/unit/test_watcher.py tests/integration/test_watcher_e2e.py -v`
Expected: FAIL with import errors

- [ ] **Step 4: Implement `watcher.py`, `synth_video.py` and `scripts/run_watch.py`**

`run()`: open source, start `LatestFrameReader`, loop `read()`; every `ocr_every_n_frames` processed frames call `ScoreboardReader.read` and build an `Observation`; call `engine.step` on every processed frame (scoreboard `None` between reads so TTLs still expire); on reader exhaustion call `engine.finish` and set state `ended`; any exception sets `failed` with the message and still calls `finish`. `scripts/run_watch.py`: argparse `--source` (path or URL; `youtube:` prefix selects `YouTubeSource`), `--league pba`, `--broadcaster one_sports`, `--home`, `--away`, `--out events.jsonl`, `--realtime`; prints the firsts as JSON at exit.

- [ ] **Step 5: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests -v && ./.venv/bin/ruff check .`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add sports_watcher/watcher.py scripts/run_watch.py tests
git commit -m "feat: game watcher runner, synthetic scoreboard video helper and CLI"
```

---

### Task 10: FastAPI service

**Files:**
- Create: `sports_watcher/api/__init__.py`, `sports_watcher/api/main.py`, `sports_watcher/api/schemas.py`, `sports_watcher/api/jobs.py`, `sports_watcher/api/routes/__init__.py`, `sports_watcher/api/routes/games.py`, `sports_watcher/api/routes/health.py`
- Test: `tests/unit/test_api.py`

**Interfaces:**
- Consumes: `GameWatcher`, `WatcherStatus`, `SourceSpec`, `load_league`, `Settings`, `Event`, `Firsts`.
- Produces: `CreateGameRequest(source: SourceSpec, league: str = "pba", broadcaster: str = "one_sports", home: str, away: str, home_color: str | None = None, away_color: str | None = None)`; `CreateGameResponse(game_id: str)`; `GameStatusResponse(game_id, home, away, league, status: WatcherStatus, recent_events: list[Event])` (last 20); `JobManager(settings: Settings, text_reader_factory: Callable[[], TextReader] = RapidOcrTextReader)` with `create(req: CreateGameRequest) -> str`, `get(game_id) -> GameWatcher` (raises `KeyError`), `stop(game_id)`, `list_ids() -> list[str]`; `create_app(settings: Settings | None = None, jobs: JobManager | None = None) -> FastAPI`; routes `GET /api/v1/health`, `POST /api/v1/games`, `GET /api/v1/games`, `GET /api/v1/games/{id}`, `GET /api/v1/games/{id}/events?since_seq=0`, `WS /api/v1/games/{id}/events`, `GET /api/v1/games/{id}/firsts`, `POST /api/v1/games/{id}/stop`; unknown id → 404.

- [ ] **Step 1: Write the failing tests**

```python
def test_game_lifecycle_over_http_and_websocket(sample_video):
    register_source("stub", lambda spec: StubSource(n=60))
    app = create_app(jobs=JobManager(Settings(_env_file=None), text_reader_factory=lambda: ScriptedTextReader({0: boxes(0,0,"10:00","1st"), 10: boxes(3,0,"9:55","1st")})))
    c = TestClient(app)
    gid = c.post("/api/v1/games", json={"source": {"kind": "stub"}, "home": "TNT", "away": "Magnolia"}).json()["game_id"]
    with c.websocket_connect(f"/api/v1/games/{gid}/events") as ws:
        first = ws.receive_json(); assert first["game_id"] == gid
    wait_until(lambda: c.get(f"/api/v1/games/{gid}").json()["status"]["state"] == "ended", timeout=10)
    evs = c.get(f"/api/v1/games/{gid}/events", params={"since_seq": 0}).json()
    assert any(e["type"] == "score_change" and e["status"] == "confirmed" for e in evs)
    assert c.get(f"/api/v1/games/{gid}/firsts").json()["first_score"]["team"] == "home"
    assert c.get("/api/v1/games/nope").status_code == 404 and c.get("/api/v1/health").json()["status"] == "ok"
```

`JobManager` takes `text_reader_factory: Callable[[], TextReader] = RapidOcrTextReader` so tests inject the scripted reader.

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/unit/test_api.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Implement the API package**

Websocket handler subscribes to `watcher.broadcaster`, sends `event.model_dump(mode="json")` as it arrives, and closes when the watcher state is `ended` or `failed` and the queue is empty. Settings and `JobManager` live on `app.state`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests -v && ./.venv/bin/ruff check .`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add sports_watcher/api tests/unit/test_api.py
git commit -m "feat: FastAPI service with game jobs, event websocket and firsts endpoint"
```

---

### Task 11: Replay rig, truth files and evaluation

**Files:**
- Create: `sports_watcher/replay/__init__.py`, `sports_watcher/replay/rig.py`, `sports_watcher/replay/truth.py`, `sports_watcher/eval/__init__.py`, `sports_watcher/eval/score.py`, `scripts/evaluate.py`, `scripts/replay_push.py`
- Test: `tests/unit/test_replay.py`, `tests/unit/test_eval.py`

**Interfaces:**
- Consumes: `Event`, `Firsts`.
- Produces: `ffmpeg_push_command(input_path: Path, url: str, loop: bool = True) -> list[str]` (`ffmpeg -re [-stream_loop -1] -i <in> -c copy -f rtsp -rtsp_transport tcp <url>` for `rtsp://`, `-f mpegts <url>` for `srt://` and `udp://`); `mediamtx_compose_yaml() -> str` (service `mediamtx` image `bluenviron/mediamtx:latest`, ports `8554:8554` and `8890:8890/udp`); `start_push(input_path, url, loop=True) -> subprocess.Popen`; `TruthEvent(t_pts: float, type: str, team: str, player: str | None = None, points: int | None = None)`; `load_truth(path) -> list[TruthEvent]`, `save_truth(path, events)`; `TypeScore(tp: int, fp: int, fn: int, precision: float, recall: float, latency_p50_s: float | None, latency_p95_s: float | None)`; `EvalReport(per_type: dict[str, TypeScore], retraction_rate: float, firsts_accuracy: float | None)`; `match_events(events: list[Event], truth: list[TruthEvent], tolerance_s: float = 1.0) -> list[tuple[TruthEvent, Event | None]]` (confirmed events only, greedy nearest within tolerance, each event used once); `score(events, truth, tolerance_s=1.0, firsts: Firsts | None = None, truth_firsts: dict | None = None) -> EvalReport` where latency for a matched pair is `event.emitted_wall - event.t_wall` and `retraction_rate` = retracted ÷ provisional.

- [ ] **Step 1: Write the failing tests**

```python
def test_push_command_for_rtsp_and_srt(tmp_path):
    cmd = ffmpeg_push_command(tmp_path / "g.mp4", "rtsp://localhost:8554/game")
    assert cmd[:2] == ["ffmpeg", "-re"] and "-stream_loop" in cmd and cmd[-1] == "rtsp://localhost:8554/game"
    assert "mpegts" in ffmpeg_push_command(tmp_path / "g.mp4", "srt://localhost:8890", loop=False)

def test_truth_roundtrip(tmp_path): ...  # save_truth then load_truth equals input

def test_score_precision_recall_and_latency():
    truth = [TruthEvent(10.0, "score_change", "home", points=2), TruthEvent(30.0, "score_change", "away", points=3)]
    events = [confirmed("score_change", t_pts=10.4, emitted_delay_s=0.3), confirmed("score_change", t_pts=55.0, emitted_delay_s=0.2),
              provisional_then_retracted("score_change", t_pts=40.0)]
    r = score(events, truth)
    s = r.per_type["score_change"]
    assert (s.tp, s.fp, s.fn) == (1, 1, 1) and s.precision == 0.5 and s.recall == 0.5
    assert s.latency_p50_s == pytest.approx(0.3) and r.retraction_rate == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/unit/test_replay.py tests/unit/test_eval.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Implement `rig.py`, `truth.py`, `score.py` and the two scripts**

`scripts/evaluate.py --events events.jsonl --truth truth.jsonl [--firsts firsts.json --truth-firsts truth_firsts.json]` prints a table (type, tp, fp, fn, P, R, p50, p95) and writes `report.json`. `scripts/replay_push.py --input game.mp4 --url rtsp://localhost:8554/game` prints the compose YAML hint when the URL is unreachable and otherwise runs `start_push` until Ctrl-C.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests -v && ./.venv/bin/ruff check .`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add sports_watcher/replay sports_watcher/eval scripts/evaluate.py scripts/replay_push.py tests
git commit -m "feat: replay rig helpers, truth files and evaluation report"
```

---

### Task 12: ROI calibration tool, golden test scaffold, docs and end-to-end run

**Files:**
- Create: `scripts/dump_frame.py`, `tests/golden/__init__.py`, `tests/golden/test_golden_clips.py`, `data/golden/README.md` (committed despite `data/` being ignored: add `!data/golden/README.md` to `.gitignore`)
- Modify: `README.md`, `CLAUDE.md`, `configs/leagues/pba.yaml` (calibrated ROI if a recording is available)

**Interfaces:**
- Consumes: `open_source`, `Roi`, `GameWatcher`, `score`, `load_truth`.
- Produces: `scripts/dump_frame.py --source <path|url> --t <seconds> --out frame.png [--roi x,y,w,h]` writing the frame and, with `--roi`, a second file `frame_roi.png` of the crop; golden test that, for each `data/golden/<clip>.mp4` with a sibling `<clip>.truth.jsonl`, runs `GameWatcher` and asserts `score_change` recall ≥ 0.9 and precision ≥ 0.9, skipping when the directory is empty.

- [ ] **Step 1: Write the golden test and the README for `data/golden`**

The README states the clip naming, the truth JSONL format from Task 11, and that clips are never committed.

- [ ] **Step 2: Run the suite to verify the golden test skips cleanly**

Run: `./.venv/bin/python -m pytest tests -v`
Expected: golden test reported as skipped, everything else passed

- [ ] **Step 3: Implement `scripts/dump_frame.py`**

- [ ] **Step 4: Update `README.md` and `CLAUDE.md`**

README: purpose; install; run the API (`./.venv/bin/uvicorn sports_watcher.api.main:create_app --factory --port 8000`); run the CLI on a recording; calibrate the scoreboard ROI with `dump_frame.py` and edit `configs/leagues/pba.yaml`; evaluate against a truth file; replay rig. CLAUDE.md: the commands above, the layout from spec section 8, the rule that `tests/integration` and `tests/golden` need the OCR package or clips, and "events are only ever appended; corrections are retractions".

- [ ] **Step 5: End-to-end check on a real recording (manual, if one is available)**

Run: `./.venv/bin/python scripts/run_watch.py --source data/pba_sample.mp4 --league pba --home TNT --away Magnolia --out tmp/events.jsonl`
Expected: `score_change` events appear with the right teams during the first minutes; adjust `scoreboard_roi` with `dump_frame.py` until the OCR reads all four fields. Record the calibrated ROI in `pba.yaml`. If no recording is on disk, note that in the commit message and leave the placeholder ROI.

- [ ] **Step 6: Full verification and commit**

Run: `./.venv/bin/python -m pytest tests -v && ./.venv/bin/ruff check .`
Expected: all passed or skipped, ruff clean

```bash
git add -A
git commit -m "feat: ROI calibration tool, golden test scaffold and docs for M0"
```

---

## After M0

Separate plans, each written after the previous milestone ships and reviewed the same way:

- M1 vision events: detector, tracker, team assignment, court homography, camera-state classifier; possession, shot attempt, made or missed, rebound, turnover, free throws; golden clips for each.
- M2 commentary: faster-whisper streaming, cue extractor, fusion into confidence and player attribution.
- M3 hardening: capture and screen adapters, replay-rig latency report, retraction tuning, Linux GPU build.
