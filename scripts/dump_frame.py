#!/usr/bin/env python
"""Save one frame of a recording or stream, optionally with a cropped ROI, to calibrate
the scoreboard region. ROI is x,y,w,h as fractions of the frame.

  ./.venv/bin/python scripts/dump_frame.py --source data/game.mp4 --t 120 --out tmp/frame.png --roi 0.02,0.86,0.40,0.10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import av
import cv2
import numpy as np

from sports_watcher.config import Roi
from sports_watcher.perception.scoreboard import crop_roi


def grab_frame(source: str, t_s: float) -> np.ndarray:
    """Seek (files) or read forward (streams) to the first frame at or after ``t_s``."""
    with av.open(source) as container:
        stream = container.streams.video[0]
        if container.format.name not in {"rtsp", "hls", "applehttp", "mpegts"} and stream.duration:
            container.seek(int(t_s / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            if float(frame.pts * stream.time_base) >= t_s:
                return frame.to_ndarray(format="bgr24")
    raise SystemExit(f"no frame at or after t={t_s}s in {source}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--t", type=float, default=0.0, help="seconds into the recording")
    ap.add_argument("--out", default="tmp/frame.png")
    ap.add_argument("--roi", help="x,y,w,h as fractions, e.g. 0.02,0.86,0.40,0.10")
    args = ap.parse_args(argv)

    frame = grab_frame(args.source, args.t)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    h, w = frame.shape[:2]
    print(f"wrote {out} ({w}x{h})")
    if args.roi:
        x, y, rw, rh = (float(v) for v in args.roi.split(","))
        roi = Roi(x=x, y=y, w=rw, h=rh)
        crop = crop_roi(frame, roi)
        roi_out = out.with_name(out.stem + "_roi" + out.suffix)
        cv2.imwrite(str(roi_out), crop)
        print(f"wrote {roi_out} ({crop.shape[1]}x{crop.shape[0]}) = pixels x{round(x * w)} y{round(y * h)} w{round(rw * w)} h{round(rh * h)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
