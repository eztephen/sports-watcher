"""The SigLIP export plus the trained camera probe on real frames. Skips without them."""

from pathlib import Path

import numpy as np
import pytest

SIGLIP, PROBE = Path("models/siglip/vision.onnx"), Path("models/camera/probe.json")
CLIP = Path("data/clips/pba_ros_cvg_2500_2700.mp4")  # the hold-out clip
# (seconds, expected state) from data/golden/pba_ros_cvg_q1.camera.jsonl, well inside segments
SAMPLES = [(1.5, "main"), (40.0, "main"), (5.5, "closeup"), (65.0, "closeup"), (114.0, "graphics")]


def _frame_at(clip: Path, t: float) -> np.ndarray:
    import av

    with av.open(str(clip)) as container:
        stream = container.streams.video[0]
        container.seek(int(t / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is not None and float(frame.pts * stream.time_base) >= t:
                return frame.to_ndarray(format="bgr24")
    raise AssertionError(f"no frame at {t}")


@pytest.mark.integration
def test_probe_classifies_held_out_frames():
    if not (SIGLIP.is_file() and PROBE.is_file() and CLIP.is_file()):
        pytest.skip("needs models/siglip, models/camera/probe.json and the hold-out clip")
    from sports_watcher.perception.camera_state import CameraProbe, frame_features
    from sports_watcher.perception.team import SiglipEmbedder

    embed = SiglipEmbedder(SIGLIP)
    probe = CameraProbe.load(PROBE)
    got = [probe.predict(embed(_frame_at(CLIP, t)), frame_features(0, 0, 0.0))[0] for t, _ in SAMPLES]
    hits = sum(g == want for g, (_, want) in zip(got, SAMPLES, strict=True))
    assert hits >= 4, list(zip([t for t, _ in SAMPLES], got, strict=True))
