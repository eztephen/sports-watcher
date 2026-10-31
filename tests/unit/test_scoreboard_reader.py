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
