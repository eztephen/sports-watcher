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


def test_leaving_main_for_replay_takes_longer_than_a_misread():
    """A replay shot by the main camera looks like live play frame for frame, so a short
    run of replay hints during play is a misread; a real replay outlasts it."""
    t = CameraStateTracker()
    t.update("main", 0.9, 0.0)
    for i in range(1, 16):  # 1.5 s of replay hints
        assert t.update("replay", 0.6, 0.1 * i) == "main"
    assert t.update("main", 0.9, 1.6) == "main"
    for i in range(26):
        t.update("replay", 0.6, 2.0 + 0.1 * i)
    assert t.state == "replay"


def test_replay_after_a_close_up_or_a_graphic_switches_at_the_usual_hold():
    t = CameraStateTracker()
    t.update("closeup", 0.9, 0.0)
    t.update("replay", 0.6, 1.0)
    assert t.update("replay", 0.6, 1.6) == "replay"


def test_main_to_close_up_keeps_the_usual_hold():
    t = CameraStateTracker()
    t.update("main", 0.9, 0.0)
    t.update("closeup", 0.9, 1.0)
    assert t.update("closeup", 0.9, 1.6) == "closeup"
