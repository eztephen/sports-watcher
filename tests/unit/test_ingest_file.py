import itertools

import numpy as np

from sports_watcher.clock import LatencyClass
from sports_watcher.ingest.file import FileSource


def test_file_source_yields_frames_with_increasing_pts(sample_video):
    src = FileSource(sample_video)
    src.open()
    try:
        frames = list(src.frames())
    finally:
        src.close()
    assert 48 <= len(frames) <= 52 and frames[0].image.shape == (240, 320, 3)
    assert frames[0].image.dtype == np.uint8
    assert all(b.pts_s > a.pts_s for a, b in itertools.pairwise(frames))
    assert all(b.index == a.index + 1 for a, b in itertools.pairwise(frames))
    assert src.latency_class == LatencyClass.FILE and src.name == "file"


def test_file_source_audio_is_16k_mono_int16(sample_video):
    src = FileSource(sample_video)
    src.open()
    try:
        chunks = list(src.audio())
    finally:
        src.close()
    total = sum(len(c.pcm) for c in chunks)
    assert chunks[0].sample_rate == 16000 and chunks[0].pcm.dtype == np.int16
    assert chunks[0].pcm.ndim == 1
    assert 30000 <= total <= 34000
