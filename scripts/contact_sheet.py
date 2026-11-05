#!/usr/bin/env python
"""Contact sheets for labelling: one image per tile grid, ``fps`` frames per second.

  ./.venv/bin/python scripts/contact_sheet.py --source data/clips/x.mp4 --fps 1 --tiles 10x6 --out tmp/sheets

Tile k (row-major, from 0) of sheet n shows the frame at (n * tiles + k) / fps seconds
after ``--start``.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--tiles", default="10x6")
    ap.add_argument("--width", type=int, default=320, help="tile width in pixels")
    ap.add_argument("--start", type=float, default=None, help="seconds into the clip")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.source).stem
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if args.start is not None:
        cmd += ["-ss", str(args.start)]
    if args.duration is not None:
        cmd += ["-t", str(args.duration)]
    cmd += [
        "-i", args.source,
        "-vf", f"fps={args.fps},scale={args.width}:-1,tile={args.tiles}",
        "-q:v", "3", str(args.out / f"{stem}_%02d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    print(f"sheets in {args.out}: tile k of sheet n is at ({args.start or 0} + (n*{args.tiles}+k)/{args.fps}) s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
