from sports_watcher.perception.camera_state import CameraStateTracker


def test_first_hint_sets_state_and_changes_need_half_a_second():
    t = CameraStateTracker()
    assert t.state == "main"
    assert t.update("closeup", 0.9, 0.0) == "closeup"
    assert t.update("main", 0.9, 0.1) == "closeup"
    assert t.update("main", 0.9, 0.4) == "closeup"
    assert t.update("main", 0.9, 0.6) == "main"


def test_flicker_does_not_switch():
    t = CameraStateTracker()
    t.update("main", 0.9, 0.0)
    for i in range(10):
        assert t.update("graphics" if i % 2 else "main", 0.9, 0.1 * i) == "main"


def test_low_confidence_hints_are_ignored():
    t = CameraStateTracker()
    t.update("main", 0.9, 0.0)
    assert t.update("graphics", 0.1, 1.0) == "main" and t.update("graphics", 0.1, 2.0) == "main"


def test_frame_features_scaling():
    import numpy as np

    from sports_watcher.perception.camera_state import frame_features

    assert np.allclose(frame_features(33, 12, 0.25), [1.0, 1.0, 0.25])


def test_probe_predicts_argmax_and_round_trips(tmp_path):
    import numpy as np

    from sports_watcher.perception.camera_state import CameraProbe

    coef = np.zeros((4, 5))
    coef[2, 0] = 5.0
    coef[3, 4] = 5.0
    p = CameraProbe(coef, np.zeros(4), ["main", "replay", "closeup", "graphics"])
    label, conf = p.predict(np.array([1.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    assert label == "closeup" and conf > 0.9
    assert p.predict(np.zeros(2), np.array([0.0, 0.0, 1.0]))[0] == "graphics"
    p.save(tmp_path / "probe.json")
    assert CameraProbe.load(tmp_path / "probe.json").predict(np.array([1.0, 0.0]), np.zeros(3))[0] == "closeup"
