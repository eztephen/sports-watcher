import time
from datetime import UTC, datetime

import numpy as np

from sports_watcher.ingest.base import Frame
from sports_watcher.ingest.latest_frame import LatestFrameReader

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _frames(n: int):
    for i in range(n):
        yield Frame(index=i, pts_s=i / 25, wall=NOW, image=np.zeros((2, 2, 3), np.uint8))


def test_latest_frame_reader_drops_when_consumer_is_slow():
    r = LatestFrameReader(_frames(200)).start()
    got = []
    while True:
        ok, f, _dropped = r.read(timeout=1.0)
        if not ok:
            break
        got.append(f.index)
        time.sleep(0.01)
    assert got[-1] == 199 and r.dropped_total > 0 and r.finished
    assert len(got) + r.dropped_total == 200


def test_latest_frame_reader_delivers_every_frame_to_a_fast_consumer():
    r = LatestFrameReader(_frames(5)).start()
    got = []
    while True:
        ok, f, _ = r.read(timeout=1.0)
        if not ok:
            break
        got.append(f.index)
    assert r.finished and got[-1] == 4 and len(got) + r.dropped_total == 5


def test_latest_frame_reader_stop_is_idempotent():
    r = LatestFrameReader(_frames(3)).start()
    r.stop()
    r.stop()
    ok, _, _ = r.read(timeout=0.1)
    assert ok is False
