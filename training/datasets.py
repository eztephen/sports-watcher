"""Download the Roboflow datasets and remap their classes to what the watcher detects.

Both datasets are CC BY 4.0 on Roboflow Universe (workspace ``roboflow-jvuqo``):
``basketball-player-detection-2`` (players, ball, rim, referees and action classes) and
``basketball-court-detection-2`` (one ``court`` class with 33 keypoints).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PLAYER_CLASS_MAP: dict[str, str | None] = {
    "ball": "ball",
    "player": "player",
    "referee": "referee",
    "rim": "rim",
    "number": None,
    "player-dribble": "player",
    "player-fall": "player",
    "player-jump-shot": "player",
    "player-layup": "player",
    "player-screen": "player",
    "player-shot-block": "player",
}
TARGET_CLASSES = ["ball", "player", "referee", "rim"]  # ids 1..4 in this order
SPLITS = ("train", "valid", "test")
COCO_FILE = "_annotations.coco.json"


def remap_coco(
    src: Path, dst: Path, class_map: dict[str, str | None], target_classes: list[str]
) -> dict[str, int]:
    """Rewrite a COCO file so its categories are ``target_classes`` with ids 1..n.

    Annotations of classes mapped to ``None`` are dropped; classes absent from the map are
    dropped too (the Roboflow super-category among them). Returns annotation counts per
    target class.
    """
    data = json.loads(Path(src).read_text(encoding="utf-8"))
    target_id = {name: i + 1 for i, name in enumerate(target_classes)}
    old_to_new: dict[int, int] = {}
    for cat in data.get("categories", []):
        mapped = class_map.get(cat["name"])
        if mapped is not None:
            old_to_new[cat["id"]] = target_id[mapped]
    counts = dict.fromkeys(target_classes, 0)
    annotations = []
    for ann in data.get("annotations", []):
        new_id = old_to_new.get(ann["category_id"])
        if new_id is None:
            continue
        annotations.append({**ann, "category_id": new_id})
        counts[target_classes[new_id - 1]] += 1
    data["categories"] = [{"id": i + 1, "name": n, "supercategory": "none"} for i, n in enumerate(target_classes)]
    data["annotations"] = annotations
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(json.dumps(data), encoding="utf-8")
    return counts


def remap_split_dirs(
    dataset_dir: Path,
    class_map: dict[str, str | None] = PLAYER_CLASS_MAP,
    target_classes: list[str] = TARGET_CLASSES,
) -> dict[str, dict[str, int]]:
    """Apply ``remap_coco`` in place to every split's ``_annotations.coco.json``."""
    out: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        path = Path(dataset_dir) / split / COCO_FILE
        if path.is_file():
            out[split] = remap_coco(path, path, class_map, target_classes)
    return out


def download_roboflow(workspace: str, project: str, version: int, fmt: str, dest: Path) -> Path:
    """Download one dataset version with the roboflow SDK (key from ``ROBOFLOW_API_KEY``)."""
    from roboflow import Roboflow  # training extra

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("ROBOFLOW_API_KEY is not set (put it in .env or the Colab environment)")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=key)
    rf.workspace(workspace).project(project).version(version).download(fmt, location=str(dest))
    return dest
