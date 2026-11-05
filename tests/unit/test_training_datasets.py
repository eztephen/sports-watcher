import json

from training.datasets import PLAYER_CLASS_MAP, TARGET_CLASSES, remap_coco


def _ann(i, cat):
    return {"id": i, "image_id": 1, "category_id": cat, "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0}


def test_remap_merges_player_actions_and_drops_numbers(tmp_path):
    src = tmp_path / "a.json"
    src.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "f.jpg", "width": 10, "height": 10}],
        "categories": [
            {"id": 0, "name": "basketball", "supercategory": "none"}, {"id": 1, "name": "ball"},
            {"id": 2, "name": "number"}, {"id": 3, "name": "player-layup"}, {"id": 4, "name": "rim"},
            {"id": 5, "name": "referee"}, {"id": 6, "name": "player"},
        ],
        "annotations": [_ann(1, 1), _ann(2, 2), _ann(3, 3), _ann(4, 6)],
    }))
    dst = tmp_path / "b.json"
    counts = remap_coco(src, dst, PLAYER_CLASS_MAP, TARGET_CLASSES)
    out = json.loads(dst.read_text())
    assert [c["name"] for c in out["categories"]] == ["ball", "player", "referee", "rim"]
    assert [c["id"] for c in out["categories"]] == [1, 2, 3, 4]
    assert counts == {"ball": 1, "player": 2, "referee": 0, "rim": 0}
    assert sorted(a["category_id"] for a in out["annotations"]) == [1, 2, 2]
