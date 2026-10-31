import pytest

from sports_watcher.config import BroadcasterConfig, load_league
from sports_watcher.events.model import EventStatus
from sports_watcher.ingest.base import SourceSpec
from tests.helpers.synth_video import write_scoreboard_video

pytestmark = pytest.mark.integration


def test_watcher_reads_synthetic_scoreboard_with_real_ocr(tmp_path):
    pytest.importorskip("rapidocr_onnxruntime")
    from sports_watcher.perception.scoreboard import RapidOcrTextReader
    from sports_watcher.watcher import GameWatcher

    path = tmp_path / "synth.mp4"
    roi = write_scoreboard_video(
        path,
        [(0.0, 0, 0, "10:00", "1st"), (1.0, 2, 0, "9:50", "1st"), (2.0, 2, 3, "9:40", "1st")],
        duration_s=3.0,
    )
    w = GameWatcher(
        SourceSpec(kind="file", path=str(path)), load_league("pba"), "one_sports",
        game_id="e2e", home="TNT", away="Magnolia", text_reader=RapidOcrTextReader(),
        broadcaster_config=BroadcasterConfig(scoreboard_roi=roi, shot_clock_visible=False),
        ocr_every_n_frames=2,
    )
    w.run()
    confirmed = [e for e in w.events if e.type == "score_change" and e.status == EventStatus.confirmed]
    assert [(e.team, e.evidence.points) for e in confirmed] == [("home", 2), ("away", 3)]
    assert abs(confirmed[0].t_pts - 1.0) < 0.5 and abs(confirmed[1].t_pts - 2.0) < 0.5
    assert w.firsts().first_score.team == "home" and w.firsts().first_three.team == "away"
    assert w.status().state == "ended"
