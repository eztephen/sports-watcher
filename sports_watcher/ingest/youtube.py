"""YouTube live or VOD, resolved to its HLS manifest with yt-dlp. Development use only."""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from sports_watcher.clock import LatencyClass
from sports_watcher.ingest.url import UrlSource, options_for

YTDLP_FORMAT = "best[protocol^=m3u8]/best"


def _ytdlp_binary() -> str:
    local = Path(sys.executable).parent / "yt-dlp"
    if local.is_file():
        return str(local)
    found = shutil.which("yt-dlp")
    if found is None:
        raise RuntimeError("yt-dlp is not installed (pip install 'sports-watcher[youtube]')")
    return found


def resolve_with_ytdlp(watch_url: str) -> str:
    """Return the direct manifest URL for a YouTube watch URL."""
    cmd = [_ytdlp_binary(), "-g", "-f", YTDLP_FORMAT, "--no-warnings", watch_url]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90, check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"yt-dlp failed for {watch_url}: {proc.stderr.strip()}")
    return proc.stdout.strip().splitlines()[0]


class YouTubeSource(UrlSource):
    name = "youtube"

    def __init__(
        self, watch_url: str, resolver: Callable[[str], str] = resolve_with_ytdlp
    ) -> None:
        super().__init__(watch_url)
        self.watch_url = watch_url
        self.latency_class = LatencyClass.HLS
        self._resolver = resolver

    def open(self) -> None:
        self.url = self._resolver(self.watch_url)
        self._options = options_for(self.url)
        super().open()
