#!/usr/bin/env python
"""`scripts/perceive.py` with every ONNX call memoised on disk (a development tool).

Building a perception cache costs about 25 s of compute per second of clip, nearly all
of it in the models. A change to the tracker, the team reading or the court mapping
leaves the models' inputs untouched, so this wrapper stores each model output under a
hash of its input and replays it: the first build of a clip records, every later build
takes a few minutes. The key is the model file's name and size plus the input tensor;
new weights or a changed crop simply miss and are computed.

  ./.venv/bin/python scripts/perceive_memo.py --source data/golden/x.mp4 --home-color white --away-color navy

Takes the options of `scripts/perceive.py`. The store is `data/cache/onnx_memo/memo.sqlite`
(gitignored, about 100 MB per minute of clip); delete it to start over.
"""

from __future__ import annotations

import hashlib
import io
import runpy
import sqlite3
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

STORE = Path("data/cache/onnx_memo/memo.sqlite")
COMMIT_EVERY = 200


def install(db: sqlite3.Connection, stats: dict[str, int]) -> None:
    """Wrap ``InferenceSession`` so ``run`` answers from ``db`` when it has seen the input."""
    init, run = ort.InferenceSession.__init__, ort.InferenceSession.run

    def memo_init(self, path_or_bytes, *args, **kwargs):
        init(self, path_or_bytes, *args, **kwargs)
        path = Path(str(path_or_bytes))
        self._memo_tag = f"{path.name}:{path.stat().st_size}".encode()

    def memo_run(self, output_names, input_feed, *args, **kwargs):
        digest = hashlib.sha1(self._memo_tag, usedforsecurity=False)
        digest.update(repr(output_names).encode())
        for name in sorted(input_feed):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(input_feed[name]).tobytes())
        key = digest.digest()
        row = db.execute("select v from memo where k=?", (key,)).fetchone()
        if row is not None:
            stats["hit"] += 1
            with np.load(io.BytesIO(row[0])) as stored:
                return [stored[f"o{i}"] for i in range(len(stored.files))]
        outputs = run(self, output_names, input_feed, *args, **kwargs)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **{f"o{i}": np.asarray(o) for i, o in enumerate(outputs)})
        db.execute("insert or replace into memo values (?,?)", (key, buffer.getvalue()))
        stats["miss"] += 1
        if stats["miss"] % COMMIT_EVERY == 0:
            db.commit()
        return outputs

    ort.InferenceSession.__init__, ort.InferenceSession.run = memo_init, memo_run


def main() -> int:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(STORE)
    db.execute("create table if not exists memo (k blob primary key, v blob)")
    stats = {"hit": 0, "miss": 0}
    install(db, stats)
    script = Path(__file__).with_name("perceive.py")
    sys.argv = [str(script), *sys.argv[1:]]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as done:
        return int(done.code or 0)
    finally:
        db.commit()
        print(f"onnx memo: {stats['hit']} replayed, {stats['miss']} computed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
