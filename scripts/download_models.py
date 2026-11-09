#!/usr/bin/env python
"""Fetch the vision weights from a GitHub release of this repository into models/.

  ./.venv/bin/python scripts/download_models.py [--tag m1-models] [--dest models]

The release holds one zip (models-<date>.zip) whose contents are the models/ layout:
detector/model.onnx + manifest.json, court/..., siglip/vision.onnx, camera/probe.json.
Publishing it is a manual step after the licences in the manifests are checked:

  zip -r models-$(date +%F).zip models -x "models/*/run/*"
  gh release create m1-models models-$(date +%F).zip --title "M1 vision weights" --notes "see manifests"

Without `gh` (https://cli.github.com) or a release, pass the zip you downloaded from Colab:

  ./.venv/bin/python scripts/download_models.py --zip ~/Downloads/models.zip

The watcher only needs the files under models/, not the release.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


def install_zip(zip_path: Path, dest: Path) -> list[str]:
    """Unpack a models zip into ``dest``: a leading ``models/`` is stripped, training
    ``run/`` folders and any path that escapes ``dest`` are skipped. Returns the paths written."""
    written: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for m in zf.namelist():
            if m.endswith("/"):
                continue
            rel = Path(m)
            if rel.parts and rel.parts[0] == "models":
                rel = Path(*rel.parts[1:])
            if not rel.parts or rel.is_absolute() or ".." in rel.parts or "run" in rel.parts:
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            written.append(rel.as_posix())
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="m1-models")
    ap.add_argument("--dest", type=Path, default=Path("models"))
    ap.add_argument("--repo", default=None, help="owner/name; default: the current checkout's origin")
    ap.add_argument("--zip", type=Path, default=None, help="a models zip on disk instead of a release")
    args = ap.parse_args(argv)

    if args.zip is not None:
        written = install_zip(args.zip, args.dest)
        print(f"installed {len(written)} files from {args.zip} under {args.dest}")
        return 0 if written else 1
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
        written = install_zip(zips[-1], args.dest)
        print(f"{len(written)} files from {zips[-1].name} are under {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
