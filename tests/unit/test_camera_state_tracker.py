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
