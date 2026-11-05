import json
from pathlib import Path

from sports_watcher.perception.models import ModelSet, load_manifest


def _write(root: Path, name: str, sha: str, **extra) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"x")
    manifest = {"name": name, "family": "rfdetr", "licence": "Apache-2.0", "sha256": sha, **extra}
    (d / "manifest.json").write_text(json.dumps(manifest))


def test_model_set_requires_detector(tmp_path):
    assert ModelSet.load(tmp_path) is None


def test_model_set_loads_optional_parts(tmp_path):
    _write(
        tmp_path, "detector", "a" * 64,
        classes={"1": "ball", "2": "player", "3": "referee", "4": "rim"}, resolution=512,
    )
    ms = ModelSet.load(tmp_path)
    assert ms is not None and ms.court is None and ms.siglip is None and ms.camera_probe is None
    assert ms.detector_manifest.classes[2] == "player"
    _write(tmp_path, "court", "b" * 64, family="rfdetr-keypoint", keypoints=[f"k{i}" for i in range(33)])
    ms2 = ModelSet.load(tmp_path)
    assert ms2.court_manifest.keypoints[32] == "k32"
    assert ms2.version != ms.version and len(ms2.version) == 12


def test_load_manifest_validates(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"name": "x", "family": "siglip", "licence": "Apache-2.0", "sha256": "c" * 64}))
    assert load_manifest(p).family == "siglip"


def test_model_set_rejects_a_manifest_whose_sha_does_not_match(tmp_path):
    import hashlib

    import pytest

    _write(tmp_path, "detector", "a" * 64, resolution=512)
    with pytest.raises(ValueError, match="sha256"):
        ModelSet.load(tmp_path, verify=True)
    (tmp_path / "detector" / "manifest.json").write_text(
        json.dumps({"name": "detector", "family": "rfdetr", "licence": "Apache-2.0", "sha256": hashlib.sha256(b"x").hexdigest()})
    )
    assert ModelSet.load(tmp_path, verify=True) is not None


def test_settings_version_changes_with_colours():
    from sports_watcher.perception.models import settings_version

    a = settings_version("abc", home_color="white", away_color="navy")
    b = settings_version("abc", home_color="navy", away_color="white")
    assert a != b and len(a) == 12 and settings_version("abc", home_color="white", away_color="navy") == a
