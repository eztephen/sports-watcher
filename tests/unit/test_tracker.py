from sports_watcher.perception.tracker import BallFilter, Detection, Tracker, iou


def _box(x, y, w=40, h=100):
    return (x, y, x + w, y + h)


def test_iou():
    assert iou(_box(0, 0), _box(0, 0)) == 1.0 and iou(_box(0, 0), _box(100, 100)) == 0.0


def test_constant_motion_keeps_one_id_and_publishes_after_three_hits():
    tr = Tracker()
    out = [tr.update([Detection(_box(10 * i, 50), "player", 0.9)], i / 30) for i in range(30)]
    assert out[0] == [] and out[1] == [] and len(out[2]) == 1
    assert {t.track_id for frame in out[2:] for t in frame} == {1}


def test_ten_frame_occlusion_reassociates_same_id():
    tr = Tracker()
    for i in range(10):
        tr.update([Detection(_box(10 * i, 50), "player", 0.9)], i / 30)
    for i in range(10, 20):
        tr.update([], i / 30)
    out = tr.update([Detection(_box(200, 50), "player", 0.9)], 20 / 30)
    assert [t.track_id for t in out] == [1]


def test_track_dropped_after_one_second():
    tr = Tracker()
    for i in range(5):
        tr.update([Detection(_box(0, 0), "player", 0.9)], i / 30)
    for i in range(5, 40):
        tr.update([], i / 30)
    out = tr.update([Detection(_box(0, 0), "player", 0.9)], 40 / 30)
    assert out == [] or out[0].track_id == 2


def test_crossing_players_keep_their_ids():
    tr = Tracker()
    ids = []
    for i in range(40):
        a = Detection(_box(10 * i, 100), "player", 0.9)  # left to right
        b = Detection(_box(400 - 10 * i, 100), "player", 0.9)  # right to left
        out = tr.update([a, b], i / 30)
        if out:
            ids.append(dict((t.track_id, round(t.box[0])) for t in out))
    first, last = ids[0], ids[-1]
    assert first[1] < first[2] and last[1] > last[2]


def test_ball_filter_predicts_and_drops():
    bf = BallFilter()
    for i in range(10):
        bf.update((10.0 * i, 100.0 - 2.0 * i), 6.0, 0.8, i / 30)
    assert bf.current(9 / 30).source == "detected"
    pred = bf.predict(12 / 30)
    assert abs(pred[0] - 120.0) < 3 and abs(pred[1] - 76.0) < 3
    assert bf.current(12 / 30).source == "predicted"
    assert bf.current(9 / 30 + 0.6) is None


def test_ball_filter_rejects_far_detection():
    bf = BallFilter()
    for i in range(5):
        bf.update((10.0 * i, 100.0), 6.0, 0.8, i / 30)
    assert bf.update((500.0, 500.0), 6.0, 0.9, 5 / 30) is False
    assert bf.update((50.0, 100.0), 6.0, 0.9, 5 / 30) is True
