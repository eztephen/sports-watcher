#!/usr/bin/env python
"""Loop a recording into a local mediamtx (RTSP) or an SRT/UDP listener as a live feed."""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

from sports_watcher.replay.rig import mediamtx_compose_yaml, start_push


def reachable(url: str, timeout: float = 1.0) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "rtsp" or not parsed.hostname:
        return True  # SRT/UDP have no handshake to probe; just push
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 8554), timeout=timeout):
            return True
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--url", default="rtsp://localhost:8554/game")
    ap.add_argument("--no-loop", action="store_true")
    args = ap.parse_args(argv)

    if not reachable(args.url):
        print(f"{args.url} is not reachable. Start mediamtx first, e.g. docker compose with:\n", file=sys.stderr)
        print(mediamtx_compose_yaml(), file=sys.stderr)
        return 2
    proc = start_push(args.input, args.url, loop=not args.no_loop)
    print(f"pushing {args.input} to {args.url} (pid {proc.pid}); Ctrl-C to stop", file=sys.stderr)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
