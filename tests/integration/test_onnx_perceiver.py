"""The real perceiver on live-play frames. Skips without weights or the PBA clip."""

from pathlib import Path

import pytest

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
    src = open_source(SourceSpec(kind="file", path=str(CLIP)))
    out = []
    try:
        for frame in src.frames():
            if frame.pts_s >= 90.5:
                break
            if frame.pts_s >= 90.0 and frame.index % 2 == 0:  # 15 perception frames of live play
                out.append(p.perceive(frame))
    finally:
        src.close()
    assert len(out[-1].tracks) >= 6 and out[-1].rim is not None
    assert ms.court is None or any(vf.homography is not None and vf.homography.confidence >= 0.4 for vf in out)
    assert any(vf.ball is not None and vf.ball.source == "detected" for vf in out)
