import pytest

from sports_watcher.clock import LatencyClass
from sports_watcher.ingest.url import UrlSource, latency_class_for, options_for
from sports_watcher.ingest.youtube import YouTubeSource


@pytest.mark.parametrize(
    "url,cls",
    [
        ("rtsp://h/x", LatencyClass.PULL),
        ("srt://h:9000", LatencyClass.PUSH),
        ("rtmp://h/live", LatencyClass.PUSH),
        ("udp://@:1234", LatencyClass.PUSH),
        ("https://h/live.m3u8", LatencyClass.HLS),
        ("http://h/live.mpd", LatencyClass.HLS),
        ("/tmp/a.mp4", LatencyClass.FILE),
    ],
)
def test_latency_class_for(url, cls):
    assert latency_class_for(url) == cls


def test_options_for_rtsp_and_hls():
    rtsp = options_for("rtsp://h/x")
    hls = options_for("https://h/a.m3u8")
    assert rtsp["rtsp_transport"] == "tcp" and "live_start_index" not in rtsp
    assert hls["live_start_index"] == "-1" and "rtsp_transport" not in hls
    for opts in (rtsp, hls):
        assert opts["fflags"] == "nobuffer" and opts["flags"] == "low_delay"
        assert opts["probesize"] == "500000" and opts["analyzeduration"] == "1000000"


def test_url_source_reads_local_path(sample_video):
    src = UrlSource(str(sample_video))
    src.open()
    try:
        frames = list(src.frames())
    finally:
        src.close()
    assert len(frames) >= 48 and src.name == "url"


def test_youtube_source_uses_injected_resolver(sample_video):
    src = YouTubeSource("https://youtube.com/watch?v=abc", resolver=lambda u: str(sample_video))
    assert src.latency_class == LatencyClass.HLS and src.name == "youtube"
    src.open()
    assert src.url == str(sample_video)
    src.close()
