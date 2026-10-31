"""Shared fixtures."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two seconds of 320x240 colour bars at 25 fps with a 440 Hz tone, made by ffmpeg."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    out = tmp_path_factory.mktemp("video") / "sample.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
        "-t", "2", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac",
        "-shortest", str(out),
    ]
    subprocess.run(cmd, check=True)
    return out
