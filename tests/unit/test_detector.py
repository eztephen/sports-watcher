import numpy as np

from sports_watcher.perception.ball import BallFinder, crop_window
from sports_watcher.perception.detector import decode_detections, preprocess
from sports_watcher.perception.tracker import Detection

CLASSES = {0: "ball", 1: "player", 2: "referee", 3: "rim"}  # logit columns; column 4 is unused


def test_decode_scales_boxes_and_maps_classes():
    boxes = np.array([[0.5, 0.5, 0.2, 0.4], [0.1, 0.1, 0.05, 0.05], [0.9, 0.9, 0.1, 0.1]])
    logits = np.full((3, 5), -9.0)
    logits[0, 1] = 3.0
    logits[1, 0] = 0.5
    logits[2, 4] = 4.0  # the unused last column is dropped however confident
    dets = decode_detections(boxes, logits, (1080, 1920), CLASSES, threshold=0.3)
    assert len(dets) == 2
    player = next(d for d in dets if d.cls == "player")
    assert np.allclose(player.box, (1920 * 0.4, 1080 * 0.3, 1920 * 0.6, 1080 * 0.7))
    assert player.confidence > 0.95
    assert next(d for d in dets if d.cls == "ball").confidence < 0.7


def test_decode_threshold_and_max_dets():
    boxes = np.tile([[0.5, 0.5, 0.1, 0.1]], (10, 1))
    logits = np.full((10, 5), -9.0)
    logits[:, 2] = np.linspace(-1, 3, 10)
    assert len(decode_detections(boxes, logits, (100, 100), CLASSES, threshold=0.5, max_dets=3)) == 3
    assert all(d.confidence >= 0.5 for d in decode_detections(boxes, logits, (100, 100), CLASSES, threshold=0.5))


def test_preprocess_shape_and_normalisation():
    x = preprocess(np.full((1080, 1920, 3), 255, np.uint8), 512)
    assert x.shape == (1, 3, 512, 512) and x.dtype == np.float32
    assert np.allclose(x[0, 0], (1 - 0.485) / 0.229, atol=1e-5)


def test_crop_window_clamps_to_image():
    assert crop_window((960.0, 540.0), 384, (1080, 1920)) == (768, 348, 1152, 732)
    assert crop_window((10.0, 10.0), 384, (1080, 1920)) == (0, 0, 384, 384)
    assert crop_window((1915.0, 1075.0), 384, (1080, 1920)) == (1536, 696, 1920, 1080)


class FakeDetector:
    def __init__(self):
        self.calls = []

    def detect(self, image):
        self.calls.append(image.shape)
        if image.shape[0] == 384:
            return [Detection((10.0, 20.0, 22.0, 32.0), "ball", 0.8)]
        return []


def test_ball_finder_offsets_crop_detection_back():
    d = FakeDetector()
    f = BallFinder(d)
    img = np.zeros((1080, 1920, 3), np.uint8)
    det = f.find(img, (960.0, 540.0), [])
    assert d.calls == [(384, 384, 3)] and det.box == (778.0, 368.0, 790.0, 380.0)


def test_ball_finder_falls_back_to_full_frame():
    d = FakeDetector()
    f = BallFinder(d)
    full = [Detection((1.0, 1.0, 9.0, 9.0), "ball", 0.4), Detection((0.0, 0.0, 50.0, 100.0), "player", 0.9)]
    assert f.find(np.zeros((1080, 1920, 3), np.uint8), None, full).confidence == 0.4 and d.calls == []


def test_ball_finder_rejects_oversized_balls():
    class HugeBallDetector:
        def detect(self, image):
            return [Detection((100.0, 100.0, 310.0, 310.0), "ball", 0.95)]  # 210 px: a body, not a ball

    f = BallFinder(HugeBallDetector())
    img = np.zeros((1080, 1920, 3), np.uint8)
    assert f.find(img, (200.0, 200.0), []) is None
    big = Detection((0.0, 0.0, 120.0, 120.0), "ball", 0.9)
    small = Detection((500.0, 500.0, 522.0, 522.0), "ball", 0.4)
    assert f.find(img, None, [big, small]) is small


def test_rfdetr_sessions_prefer_the_cpu_provider():
    # RF-DETR graphs split into ~100 CoreML partitions and run about twice as slow there as on
    # the CPU (Small 512: 300 ms vs 650 ms per frame on an M-series Mac); SigLIP is the opposite.
    from sports_watcher.perception.detector import default_providers

    assert default_providers(prefer_cpu=True)[0] == "CPUExecutionProvider"
    assert "CPUExecutionProvider" in default_providers()


IMG = np.zeros((1080, 1920, 3), np.uint8)


def test_the_crop_pass_wins_over_a_full_frame_ball():
    # Trusting a confident full-frame ball without the crop was tried and cost events.
    d = FakeDetector()
    full = Detection((900.0, 500.0, 930.0, 530.0), "ball", 0.9)
    out = BallFinder(d).find(IMG, (200.0, 200.0), [full])
    assert len(d.calls) == 1 and out is not None and out.confidence == 0.8


def test_no_crop_pass_once_the_ball_has_been_lost():
    # The filter extrapolates for ever; a crop around a position minutes old finds nothing
    # and cost as much as the whole detector.
    d = FakeDetector()
    assert BallFinder(d).find(IMG, (200.0, 200.0), [], tracking=False) is None
    assert d.calls == []
    weak = Detection((900.0, 500.0, 930.0, 530.0), "ball", 0.35)
    assert BallFinder(d).find(IMG, (200.0, 200.0), [weak], tracking=False) == weak


def test_ball_filter_says_whether_the_ball_was_seen_lately():
    from sports_watcher.perception.tracker import BallFilter

    f = BallFilter()
    assert f.seen_within(0.0, 1.0) is False
    f.update((100.0, 100.0), 10.0, 0.9, 0.0)
    assert f.seen_within(0.9, 1.0) is True and f.seen_within(1.1, 1.0) is False
