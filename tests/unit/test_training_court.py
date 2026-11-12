import json

import pytest

from training.datasets import merge_keypoint_sets


def _coco(names, files, kp):
    return {
        "info": {}, "licenses": [],
        "categories": [{"id": 0, "name": "x"}, {"id": 1, "name": "court", "keypoints": names, "skeleton": []}],
        "images": [{"id": i, "file_name": f, "width": 10, "height": 10} for i, f in enumerate(files)],
        "annotations": [
            {"id": i + 1, "image_id": i, "category_id": 1, "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0,
             "keypoints": list(kp)}
            for i in range(len(files))
        ],
    }


def _split(root, split, coco):
    d = root / split
    d.mkdir(parents=True)
    for im in coco["images"]:
        (d / im["file_name"]).write_bytes(b"jpg")
    (d / "_annotations.coco.json").write_text(json.dumps(coco))


def test_merge_adds_the_extra_set_to_train_reordered_by_name_and_repeated(tmp_path):
    base, extra, out = tmp_path / "nba", tmp_path / "pba", tmp_path / "out"
    for split in ("train", "valid"):
        _split(base, split, _coco(["01", "02"], ["a.jpg"], [1, 1, 2, 2, 2, 2]))
    # the extra set lists the same keypoints in another order
    _split(extra, "train", _coco(["02", "01"], ["p.jpg", "q.jpg"], [5, 5, 2, 7, 7, 1]))
    counts = merge_keypoint_sets(base, extra, out, repeat=3)
    assert counts == {"train": {"base": 1, "extra": 2, "repeat": 3}, "valid": {"base": 1, "extra": 0, "repeat": 3}}
    train = json.loads((out / "train" / "_annotations.coco.json").read_text())
    assert train["categories"][1]["keypoints"] == ["01", "02"]
    assert len(train["images"]) == 1 + 2 * 3 == len(train["annotations"])
    assert len({im["id"] for im in train["images"]}) == 7 and len({a["id"] for a in train["annotations"]}) == 7
    by_image = {a["image_id"]: a for a in train["annotations"]}
    extra_images = [im for im in train["images"] if im["file_name"] in ("p.jpg", "q.jpg")]
    assert len(extra_images) == 6
    assert all(by_image[im["id"]]["keypoints"] == [7, 7, 1, 5, 5, 2] for im in extra_images)
    assert (out / "train" / "p.jpg").is_file() and (out / "train" / "a.jpg").is_file()
    assert len(json.loads((out / "valid" / "_annotations.coco.json").read_text())["images"]) == 1


def test_merge_rejects_an_extra_set_with_other_keypoints(tmp_path):
    base, extra = tmp_path / "nba", tmp_path / "pba"
    _split(base, "train", _coco(["01", "02"], ["a.jpg"], [1, 1, 2, 2, 2, 2]))
    _split(extra, "train", _coco(["01", "03"], ["p.jpg"], [1, 1, 2, 2, 2, 2]))
    with pytest.raises(ValueError, match="03"):
        merge_keypoint_sets(base, extra, tmp_path / "out")


def test_untie_redraws_only_rows_that_copy_an_earlier_row():
    torch = pytest.importorskip("torch")
    from training.train_court import untie_duplicate_rows

    g = torch.Generator().manual_seed(0)
    first = torch.randn(17, 8, generator=g)
    p = torch.nn.Parameter(first.repeat(2, 1)[:33].clone())  # how RF-DETR resizes 17 rows to 33
    assert untie_duplicate_rows(p, generator=g) == 16
    assert torch.equal(p[:17], first)
    assert len({tuple(r.tolist()) for r in p.detach()}) == 33
    assert 0.3 < float(p.detach()[17:].std() / first.std()) < 3.0
    assert untie_duplicate_rows(p, generator=g) == 0


def test_untie_model_touches_only_per_keypoint_tables():
    torch = pytest.importorskip("torch")
    from training.train_court import untie_keypoint_slots

    m = torch.nn.Module()
    m.table = torch.nn.Parameter(torch.randn(17, 4).repeat(2, 1)[:33].clone())
    m.other = torch.nn.Parameter(torch.zeros(5, 4))  # duplicate rows, not a keypoint table
    m.bias = torch.nn.Parameter(torch.zeros(33))
    assert untie_keypoint_slots(m, 33) == {"table": 16}
    assert float(m.other.abs().sum()) == 0.0 and float(m.bias.abs().sum()) == 0.0
