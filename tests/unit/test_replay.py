from sports_watcher.replay.rig import ffmpeg_push_command, mediamtx_compose_yaml
from sports_watcher.replay.truth import TruthEvent, load_truth, save_truth


def test_push_command_for_rtsp_and_srt(tmp_path):
    cmd = ffmpeg_push_command(tmp_path / "g.mp4", "rtsp://localhost:8554/game")
    assert cmd[:2] == ["ffmpeg", "-re"] and "-stream_loop" in cmd
    assert cmd[-1] == "rtsp://localhost:8554/game" and "rtsp" in cmd and "-rtsp_transport" in cmd
    srt = ffmpeg_push_command(tmp_path / "g.mp4", "srt://localhost:8890", loop=False)
    assert "mpegts" in srt and "-stream_loop" not in srt and srt[-1] == "srt://localhost:8890"


def test_mediamtx_compose_yaml_mentions_image_and_ports():
    y = mediamtx_compose_yaml()
    assert "bluenviron/mediamtx" in y and "8554:8554" in y and "8890:8890/udp" in y


def test_truth_roundtrip(tmp_path):
    events = [
        TruthEvent(t_pts=10.0, type="score_change", team="home", points=2),
        TruthEvent(t_pts=30.5, type="period_end", team="unknown"),
        TruthEvent(t_pts=40.0, type="score_change", team="away", player="J. Doe", points=3),
    ]
    path = tmp_path / "truth.jsonl"
    save_truth(path, events)
    assert load_truth(path) == events and len(path.read_text().splitlines()) == 3
