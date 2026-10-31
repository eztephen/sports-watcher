import pytest

from tests.helpers.scoreboard_images import render_scoreboard

pytestmark = pytest.mark.integration


def test_rapidocr_reads_rendered_scoreboard():
    pytest.importorskip("rapidocr_onnxruntime")
    from sports_watcher.perception.scoreboard import RapidOcrTextReader, parse_scoreboard

    img = render_scoreboard(78, 65, "7:31", "3rd")
    boxes = RapidOcrTextReader().read(img)
    st = parse_scoreboard(boxes, home_side="left", shot_clock_visible=False)
    assert (st.home, st.away, st.clock_s, st.period) == (78, 65, 451.0, 3), st.raw
