"""Live sources: the scoreboard is read on stream time and perception never blocks it."""

import itertools
import time

from sports_watcher.clock import LatencyClass
from sports_watcher.events.model import EventStatus
from sports_watcher.ingest.base import SourceSpec
from sports_watcher.ingest.factory import register_source
from sports_watcher.perception.perceiver import ScriptedPerceiver
from sports_watcher.watcher import GameWatcher
from tests.helpers.leagues import fixture_league
from tests.helpers.scoreboard_images import boxes
from tests.helpers.sources import StubSource
from tests.helpers.text_readers import SeqTextReader

SPEED = 10.0  # the stub plays ten times faster than its timestamps


class LiveStub(StubSource):
    """A push source: frames arrive on the wall clock whether or not anyone is ready."""

    latency_class = LatencyClass.PUSH

    def frames(self):
        for frame in super().frames():
            time.sleep(1.0 / (self.fps * SPEED))
            yield frame


class SlowPerceiver(ScriptedPerceiver):
    def __init__(self, seconds: float) -> None:
        super().__init__([])
        self._seconds = seconds

    def perceive(self, frame):
        time.sleep(self._seconds)
        return super().perceive(frame)


class ClockReader(SeqTextReader):
    """A scoreboard whose home score goes from 0 to 2 at the fifteenth read."""

    def __init__(self) -> None:
        super().__init__([[]])

    def read(self, image_bgr):
        self.calls += 1
        return boxes(0 if self.calls < 15 else 2, 0, f"9:{59 - self.calls // 5:02d}", "1st")


def _live_watcher(perceiver, reader, n=300):
    register_source("stub", lambda spec: LiveStub(n=n, fps=30))
    w = GameWatcher(
        SourceSpec(kind="stub", url="udp://x"), fixture_league(), "one_sports", "g", "H", "A",
        text_reader=reader, perceiver=perceiver,
    )
    seen = []
    step = w._plugin.update_state

    def record(state, obs):
        seen.append((obs.frame_index, obs.t_pts, obs.scoreboard is not None, obs.vision.frame_index if obs.vision else None))
        return step(state, obs)

    w._plugin.update_state = record
    return w, seen


def test_live_scoreboard_is_read_five_times_a_second_whatever_perception_costs():
    reader = ClockReader()
    w, _ = _live_watcher(SlowPerceiver(0.1), reader)  # one perception per second of stream
    w.run()
    assert w.status().state == "ended"
    assert reader.calls >= 30  # ten seconds of stream; it was 2 with OCR on every fifth processed frame
    assert (("score_change", EventStatus.confirmed)) in [(e.type, e.status) for e in w.events]


def test_live_vision_is_stepped_with_its_own_frame_and_in_stream_order():
    w, seen = _live_watcher(SlowPerceiver(0.05), ClockReader())
    w.run()
    with_vision = [s for s in seen if s[3] is not None]
    assert len(with_vision) >= 5
    assert all(frame == vision for frame, _, _, vision in with_vision)
    times = [t for _, t, _, _ in seen]
    assert times == sorted(times)
    assert len(seen) > 3 * len(with_vision)  # frames between perceptions are still observed


def test_live_without_a_perceiver_reads_the_scoreboard_on_stream_time():
    reader = ClockReader()
    w, seen = _live_watcher(None, reader)
    w.run()
    reads = [t for _, t, sb, _ in seen if sb]
    assert 30 <= len(reads) <= 55 and min(b - a for a, b in itertools.pairwise(reads)) >= 0.19
