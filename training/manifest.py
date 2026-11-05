"""Write the manifest.json that sits next to every exported model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(model_path: Path, **fields: Any) -> Path:
    """Write ``manifest.json`` beside ``model_path`` with its sha256 filled in."""
    model_path = Path(model_path)
    manifest = {**fields, "sha256": sha256_of(model_path)}
    out = model_path.parent / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
