from fastapi.testclient import TestClient

from sports_watcher.api.jobs import JobManager
from sports_watcher.api.main import create_app
from sports_watcher.ingest.factory import register_source
from tests.helpers.leagues import fixture_settings
from tests.helpers.scoreboard_images import boxes
from tests.helpers.sources import StubSource
from tests.helpers.text_readers import ScriptedTextReader
from tests.helpers.wait import wait_until


def scripted():
    return ScriptedTextReader({0: boxes(0, 0, "10:00", "1st"), 10: boxes(3, 0, "9:55", "1st")})


def make_client(n_frames: int = 200) -> tuple[TestClient, JobManager]:
    register_source("stub", lambda spec: StubSource(n=n_frames))
    jobs = JobManager(fixture_settings(), text_reader_factory=scripted, ocr_every_n_frames=5)
    return TestClient(create_app(settings=fixture_settings(), jobs=jobs)), jobs


def test_game_lifecycle_over_http_and_websocket():
    c, _ = make_client()
    body = {"source": {"kind": "stub"}, "home": "TNT", "away": "Magnolia"}
    created = c.post("/api/v1/games", json=body)
    assert created.status_code == 201
    gid = created.json()["game_id"]
    with c.websocket_connect(f"/api/v1/games/{gid}/events") as ws:
        first = ws.receive_json()
        assert first["game_id"] == gid and first["seq"] >= 1
    wait_until(lambda: c.get(f"/api/v1/games/{gid}").json()["status"]["state"] == "ended")
    status = c.get(f"/api/v1/games/{gid}").json()
    assert status["home"] == "TNT" and status["league"] == "pba"
    assert status["status"]["score"] == [3, 0] and len(status["recent_events"]) <= 20
    evs = c.get(f"/api/v1/games/{gid}/events", params={"since_seq": 0}).json()
    assert any(e["type"] == "score_change" and e["status"] == "confirmed" for e in evs)
    last_seq = evs[-1]["seq"]
    assert c.get(f"/api/v1/games/{gid}/events", params={"since_seq": last_seq}).json() == []
    assert c.get(f"/api/v1/games/{gid}/firsts").json()["first_score"]["team"] == "home"
    assert gid in c.get("/api/v1/games").json()["game_ids"]
    assert c.get("/api/v1/games/nope").status_code == 404
    assert c.get("/api/v1/health").json()["status"] == "ok"


def test_websocket_streams_backfill_then_closes_when_the_game_ends():
    c, _ = make_client(n_frames=60)
    gid = c.post("/api/v1/games", json={"source": {"kind": "stub"}, "home": "A", "away": "B"}).json()["game_id"]
    wait_until(lambda: c.get(f"/api/v1/games/{gid}").json()["status"]["state"] == "ended")
    expected = [e["seq"] for e in c.get(f"/api/v1/games/{gid}/events").json()]
    with c.websocket_connect(f"/api/v1/games/{gid}/events") as ws:
        got = [ws.receive_json()["seq"] for _ in expected]
    assert got == expected


def test_stop_endpoint_ends_a_running_job():
    c, _ = make_client(n_frames=10_000_000)
    gid = c.post("/api/v1/games", json={"source": {"kind": "stub"}, "home": "A", "away": "B"}).json()["game_id"]
    wait_until(lambda: c.get(f"/api/v1/games/{gid}").json()["status"]["state"] == "running")
    assert c.post(f"/api/v1/games/{gid}/stop").status_code == 202
    wait_until(lambda: c.get(f"/api/v1/games/{gid}").json()["status"]["state"] == "ended")
    assert c.post("/api/v1/games/nope/stop").status_code == 404


def test_create_rejects_unknown_league_and_broadcaster():
    c, _ = make_client()
    bad_league = c.post("/api/v1/games", json={"source": {"kind": "stub"}, "home": "A", "away": "B", "league": "nope"})
    assert bad_league.status_code == 400 and "league" in bad_league.json()["detail"]
    bad_bc = c.post("/api/v1/games", json={"source": {"kind": "stub"}, "home": "A", "away": "B", "broadcaster": "nope"})
    assert bad_bc.status_code == 400 and "broadcaster" in bad_bc.json()["detail"]


def test_websocket_client_disconnect_releases_the_subscription():
    c, jobs = make_client(n_frames=10_000_000)
    gid = c.post("/api/v1/games", json={"source": {"kind": "stub"}, "home": "A", "away": "B"}).json()["game_id"]
    with c.websocket_connect(f"/api/v1/games/{gid}/events") as ws:
        ws.receive_json()
    watcher = jobs.get(gid).watcher
    wait_until(lambda: watcher.broadcaster.subscriber_count == 0, timeout=5)
    c.post(f"/api/v1/games/{gid}/stop")
    wait_until(lambda: c.get(f"/api/v1/games/{gid}").json()["status"]["state"] == "ended")


def test_create_builds_a_perceiver_from_the_request_colours(monkeypatch):
    import sports_watcher.api.jobs as jobs_module

    calls = []

    def fake_build(settings, league, clip, home_color, away_color, use_cache=True):
        calls.append((home_color, away_color, clip))

    monkeypatch.setattr(jobs_module, "build_perceiver", fake_build)
    c, _ = make_client(n_frames=5)
    body = {"source": {"kind": "stub"}, "home": "TNT", "away": "Magnolia", "home_color": "white", "away_color": "navy"}
    assert c.post("/api/v1/games", json=body).status_code == 201
    assert calls == [("white", "navy", None)]
