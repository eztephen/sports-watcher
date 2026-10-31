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


def test_rapidocr_recognition_only_reads_single_digit_scores():
    pytest.importorskip("rapidocr_onnxruntime")
    from sports_watcher.perception.scoreboard import RapidOcrTextReader

    img = render_scoreboard(7, 2, "10:05", "1st")  # 640x80: scores at x 20-60 and 530-570
    reader = RapidOcrTextReader()
    away_text, away_conf = reader.read_line(img[5:70, 10:80])
    home_text, home_conf = reader.read_line(img[5:70, 520:600])
    assert (away_text, home_text) == ("7", "2") and min(away_conf, home_conf) > 0.8
