import importlib.util
import json
import zipfile
from pathlib import Path

SCRIPT = Path("scripts/download_models.py")


def _module():
    spec = importlib.util.spec_from_file_location("download_models", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_install_zip_strips_the_models_prefix_and_skips_run_dirs(tmp_path):
    zpath = tmp_path / "models-2026-11-10.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("models/detector/model.onnx", b"onnx")
        zf.writestr("models/detector/manifest.json", json.dumps({"name": "detector"}))
        zf.writestr("models/detector/run/last.ckpt", b"big")
        zf.writestr("models/court/model.onnx", b"onnx2")
        zf.writestr("../escape.txt", b"no")
    dest = tmp_path / "models"
    installed = _module().install_zip(zpath, dest)
    assert (dest / "detector" / "model.onnx").read_bytes() == b"onnx"
    assert (dest / "court" / "model.onnx").read_bytes() == b"onnx2"
    assert not (dest / "detector" / "run").exists()
    assert not (tmp_path / "escape.txt").exists()
    assert sorted(installed) == ["court/model.onnx", "detector/manifest.json", "detector/model.onnx"]
