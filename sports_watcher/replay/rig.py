"""Push a recording through ffmpeg as if it were a live feed (RTSP via mediamtx, or
SRT/UDP MPEG-TS), so latency can be measured against a stream we control."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path
from urllib.parse import urlparse


def ffmpeg_push_command(input_path: Path, url: str, loop: bool = True) -> list[str]:
    cmd = ["ffmpeg", "-re"]
    if loop:
        cmd += ["-stream_loop", "-1"]
    cmd += ["-i", str(input_path), "-c", "copy"]
    if urlparse(url).scheme == "rtsp":
        cmd += ["-f", "rtsp", "-rtsp_transport", "tcp"]
    else:
        cmd += ["-f", "mpegts"]
    cmd.append(url)
    return cmd


def mediamtx_compose_yaml() -> str:
    return textwrap.dedent(
        """\
        services:
          mediamtx:
            image: bluenviron/mediamtx:latest
            ports:
              - "8554:8554"        # RTSP
              - "8890:8890/udp"    # SRT
              - "1935:1935"        # RTMP
            restart: unless-stopped
        """
    )


def start_push(input_path: Path, url: str, loop: bool = True) -> subprocess.Popen:
    return subprocess.Popen(
        ffmpeg_push_command(input_path, url, loop),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
