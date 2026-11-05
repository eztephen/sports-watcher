#!/usr/bin/env python
"""Fetch the vision weights from a GitHub release of this repository into models/.

  ./.venv/bin/python scripts/download_models.py [--tag m1-models] [--dest models]

The release holds one zip (models-<date>.zip) whose contents are the models/ layout:
detector/model.onnx + manifest.json, court/..., siglip/vision.onnx, camera/probe.json.
Publishing it is a manual step after the licences in the manifests are checked:

  zip -r models-$(date +%F).zip models -x "models/*/run/*"
  gh release create m1-models models-$(date +%F).zip --title "M1 vision weights" --notes "see manifests"

Without `gh` (https://cli.github.com) or a release, copy the zip's contents into models/
by hand; the watcher only needs the files, not the release.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="m1-models")
    ap.add_argument("--dest", type=Path, default=Path("models"))
    ap.add_argument("--repo", default=None, help="owner/name; default: the current checkout's origin")
    args = ap.parse_args(argv)

    if shutil.which("gh") is None:
        print("gh is not installed: copy the zip's contents into models/ by hand (see --help)")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["gh", "release", "download", args.tag, "--pattern", "models-*.zip", "--dir", tmp]
        if args.repo:
            cmd += ["-R", args.repo]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr.strip() or "download failed")
            print("no release asset: copy the zip's contents into models/ by hand (see --help)")
            return 1
        zips = sorted(Path(tmp).glob("models-*.zip"))
        if not zips:
            print("the release has no models-*.zip asset")
            return 1
        with zipfile.ZipFile(zips[-1]) as zf:
            members = [m for m in zf.namelist() if not m.endswith("/")]
            for m in members:
                rel = Path(m)
                if rel.parts and rel.parts[0] == "models":
                    rel = Path(*rel.parts[1:])
                if not rel.parts or ".." in rel.parts:
                    continue
                target = args.dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        print(f"weights from {zips[-1].name} are under {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
