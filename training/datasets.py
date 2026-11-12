"""Download the Roboflow datasets and remap their classes to what the watcher detects.

Both datasets are CC BY 4.0 on Roboflow Universe (workspace ``roboflow-jvuqo``):
``basketball-player-detection-2`` (players, ball, rim, referees and action classes) and
``basketball-court-detection-2`` (one ``court`` class with 33 keypoints).
"""

from __future__ import annotations

import json
import os
import shutil
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


def _keypoint_category(coco: dict) -> dict:
    for cat in coco["categories"]:
        if cat.get("keypoints"):
            return cat
    raise ValueError("no category with a keypoints list")


def merge_keypoint_sets(base_dir: Path, extra_dir: Path, out_dir: Path, repeat: int = 1) -> dict[str, dict[str, int]]:
    """Write ``out_dir`` as the base keypoint set plus the extra one, split by split.

    The extra set (a few labelled frames of the target broadcast) must name the same
    keypoints; its triplets are reordered to the base order and its images are listed
    ``repeat`` times so a small set weighs against a large base. Images are copied."""
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        base_json = Path(base_dir) / split / COCO_FILE
        if not base_json.is_file():
            continue
        coco = json.loads(base_json.read_text(encoding="utf-8"))
        cat = _keypoint_category(coco)
        names = [str(n) for n in cat["keypoints"]]
        dst = Path(out_dir) / split
        dst.mkdir(parents=True, exist_ok=True)
        for im in coco["images"]:
            shutil.copy2(base_json.parent / im["file_name"], dst / im["file_name"])
        n_base, n_extra = len(coco["images"]), 0
        extra_json = Path(extra_dir) / split / COCO_FILE
        if extra_json.is_file():
            extra = json.loads(extra_json.read_text(encoding="utf-8"))
            extra_cat = _keypoint_category(extra)
            extra_names = [str(n) for n in extra_cat["keypoints"]]
            if sorted(extra_names) != sorted(names):
                odd = sorted(set(extra_names) ^ set(names))
                raise ValueError(f"the extra set's keypoints differ from the base set's: {odd}")
            order = [extra_names.index(n) for n in names]
            by_image: dict[int, list[dict]] = {}
            for ann in extra["annotations"]:
                if ann["category_id"] == extra_cat["id"]:
                    by_image.setdefault(ann["image_id"], []).append(ann)
            next_image = max((im["id"] for im in coco["images"]), default=-1) + 1
            next_ann = max((a["id"] for a in coco["annotations"]), default=0) + 1
            for im in extra["images"]:
                shutil.copy2(extra_json.parent / im["file_name"], dst / im["file_name"])
            n_extra = len(extra["images"])
            for _ in range(repeat):
                for im in extra["images"]:
                    coco["images"].append({**im, "id": next_image})
                    for ann in by_image.get(im["id"], []):
                        kp = ann["keypoints"]
                        reordered = [v for i in order for v in kp[3 * i : 3 * i + 3]]
                        coco["annotations"].append(
                            {**ann, "id": next_ann, "image_id": next_image, "category_id": cat["id"], "keypoints": reordered}
                        )
                        next_ann += 1
                    next_image += 1
        (dst / COCO_FILE).write_text(json.dumps(coco), encoding="utf-8")
        counts[split] = {"base": n_base, "extra": n_extra, "repeat": repeat}
    return counts


def download_roboflow(workspace: str, project: str, version: int, fmt: str, dest: Path) -> Path:
    """Download one dataset version with the roboflow SDK (key from ``ROBOFLOW_API_KEY``)."""
    from roboflow import Roboflow  # training extra

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("ROBOFLOW_API_KEY is not set (put it in .env or the Colab environment)")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)  # the client skips an existing location
    rf = Roboflow(api_key=key)
    rf.workspace(workspace).project(project).version(version).download(fmt, location=str(dest), overwrite=True)
    return dest
