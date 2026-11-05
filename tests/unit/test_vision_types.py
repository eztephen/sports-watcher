from datetime import UTC, datetime

import numpy as np

from sports_watcher.perception.types import (
    BallObs,
    Homography,
    Track,
    VisionFrame,
    empty_vision,
    feet_of,
    vision_from_dict,
    vision_to_dict,
)


def _frame():
    return VisionFrame(
        frame_index=7,
        t_pts=0.233,
        camera_hint="main",
        camera_confidence=0.8,
        tracks=[Track(3, (10.0, 20.0, 50.0, 120.0), "player", "home", (4.0, 7.5), 0.9)],
        ball=BallObs((30.0, 15.0), 6.0, (4.1, 7.4), "detected", 0.7),
        rim=(900.0, 300.0, 940.0, 320.0),
        homography=Homography(np.eye(3), 12, 2.5, 0.85),
    )


def test_round_trip_through_dict():
    vf = _frame()
    back = vision_from_dict(vision_to_dict(vf))
    assert back.tracks == vf.tracks and back.ball == vf.ball and back.rim == vf.rim
    assert back.homography.inliers == 12 and np.allclose(back.homography.matrix, np.eye(3))
    assert back.camera_hint == "main" and back.frame_index == 7


def test_empty_vision_has_no_detections():
    vf = empty_vision(1, 0.0)
    assert vf.tracks == [] and vf.ball is None and vf.homography is None
    assert vf.camera_hint == "main"


def test_feet_of_is_bottom_centre():
    assert feet_of((10.0, 20.0, 50.0, 120.0)) == (30.0, 120.0)


def test_game_state_and_observation_defaults():
    from sports_watcher.sports.base import Observation
    from sports_watcher.state.game_state import GameState

    s = GameState()
    assert s.holder is None and s.clock_running is False and s.clock_state == "unknown"
    assert s.tracks == []
    obs = Observation(frame_index=0, t_pts=0.0, t_wall=datetime.now(UTC))
    assert obs.vision is None


def test_evidence_vision_fields_default_none():
    from sports_watcher.events.model import Evidence

    e = Evidence()
    assert e.homography_confidence is None and e.clock_state is None and e.track_id is None
