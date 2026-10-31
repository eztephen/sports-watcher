from sports_watcher.config import Roi
from sports_watcher.perception.scoreboard import ScoreboardReader, crop_roi
from tests.helpers.scoreboard_images import B1, B2, BLANK, BROADCASTER
from tests.helpers.text_readers import SeqTextReader


def test_reader_majority_vote_ignores_single_misread():
    good = [("78", B1, 0.9), ("65", B2, 0.9)]
    bad = [("73", B1, 0.9), ("65", B2, 0.9)]
    seq = [good, good, bad, good, good]
    r = ScoreboardReader(SeqTextReader(seq), broadcaster=BROADCASTER, vote_window=5)
    states = [r.read(BLANK) for _ in seq]
    assert states[2].home == 78 and states[-1].home == 78
    assert states[-1].confidence == 0.8 and states[-1].away == 65


def test_reader_follows_a_real_change_after_majority():
    before = [("78", B1, 0.9), ("65", B2, 0.9)]
    after = [("80", B1, 0.9), ("65", B2, 0.9)]
    r = ScoreboardReader(SeqTextReader([before] * 5 + [after] * 5), broadcaster=BROADCASTER)
    homes = [r.read(BLANK).home for _ in range(10)]
    assert homes[:5] == [78] * 5 and homes[-1] == 80 and 80 in homes[6:9]


def test_crop_roi_uses_fractions_of_the_frame():
    import numpy as np

    frame = np.zeros((100, 200, 3), np.uint8)
    crop = crop_roi(frame, Roi(x=0.5, y=0.25, w=0.25, h=0.5))
    assert crop.shape == (50, 50, 3)


class SeqLineReader:
    """read_line returns scripted (text, conf) pairs in call order; read() is unused."""

    def __init__(self, lines):
        self._lines = list(lines)
        self.calls = 0

    def read(self, image_bgr):
        raise AssertionError("detection path must not be used when field ROIs are set")

    def read_line(self, image_bgr):
        text, conf = self._lines[self.calls % len(self._lines)]
        self.calls += 1
        return text, conf


def _fields_broadcaster():
    from sports_watcher.config import BroadcasterConfig, FieldRois, Roi

    box = Roi(x=0.0, y=0.0, w=0.1, h=0.5)
    return BroadcasterConfig(
        scoreboard_roi=Roi(x=0.0, y=0.0, w=1.0, h=1.0),
        home_side="right",
        fields=FieldRois(home=box, away=box, clock=box, period=box, shot_clock=box),
    )


def test_reader_uses_field_rois_with_recognition_only_reads():
    reader = SeqLineReader([("2", 0.99), ("7", 0.99), ("10:05", 0.99), ("1sT", 0.84), ("6", 0.95)])
    r = ScoreboardReader(reader, broadcaster=_fields_broadcaster(), vote_window=1)
    st = r.read(BLANK)
    assert (st.home, st.away, st.clock_s, st.period, st.shot_clock_s) == (2, 7, 605.0, 1, 6.0)
    assert st.raw == ["2", "7", "10:05", "1sT", "6"] and st.confidence == 1.0
    assert reader.calls == 5


def test_field_reads_tolerate_garbage_and_still_vote():
    good = [("2", 0.99), ("7", 0.99), ("10:05", 0.99), ("1ST", 0.9), ("6", 0.95)]
    garbage = [("C", 0.3), ("7", 0.99), ("", 0.0), ("WEST", 0.5), ("", 0.0)]
    reader = SeqLineReader(good + good + garbage + good + good)
    r = ScoreboardReader(reader, broadcaster=_fields_broadcaster(), vote_window=5)
    states = [r.read(BLANK) for _ in range(5)]
    assert states[2].home == 2 and states[-1].home == 2 and states[-1].confidence == 0.8
    assert states[2].clock_s == 605.0  # latest non-None clock survives a blank read


def test_field_rois_without_shot_clock_field():
    from sports_watcher.config import BroadcasterConfig, FieldRois, Roi

    box = Roi(x=0.0, y=0.0, w=0.1, h=0.5)
    bc = BroadcasterConfig(scoreboard_roi=box, fields=FieldRois(home=box, away=box, clock=box))
    reader = SeqLineReader([("30", 0.9), ("28", 0.9), ("0:59.8", 0.9)])
    st = ScoreboardReader(reader, broadcaster=bc, vote_window=1).read(BLANK)
    assert (st.home, st.away, st.clock_s, st.period, st.shot_clock_s) == (30, 28, 59.8, None, None)
    assert reader.calls == 3
