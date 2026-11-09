# Training the vision models

Three models, all exported to ONNX under `models/` (gitignored). Training runs on Google
Colab's free tier (T4) or on the Mac with `--device mps`; the scripts are the same.

| Model | Script | Data | Output |
|---|---|---|---|
| Detector (ball, player, referee, rim) | `train_detector.py` | `roboflow-jvuqo/basketball-player-detection-2`, CC BY 4.0 | `models/detector/model.onnx` + `manifest.json` |
| Court keypoints (33) | `train_court.py` | `roboflow-jvuqo/basketball-court-detection-2` v19, CC BY 4.0 | `models/court/model.onnx` + `manifest.json` |
| SigLIP vision tower | `export_siglip.py` | `google/siglip-base-patch16-224`, Apache-2.0, no training | `models/siglip/vision.onnx` |
| Camera probe | `train_camera_probe.py` | frames labelled from the clips on disk | `models/camera/probe.json` |

Only the Apache-2.0 RF-DETR sizes are used (Nano, Small, Medium, and the keypoint preview).

## Colab

1. New notebook, runtime type GPU (T4).
2. In a cell:

```
!pip -q install "rfdetr[train,onnx]" roboflow
!git clone <this repository> sports-watcher
%cd sports-watcher
%env ROBOFLOW_API_KEY=<your key>
```

3. Download and remap the player set (pick the latest version shown on Roboflow Universe):

```
from pathlib import Path
from training.datasets import download_roboflow, remap_split_dirs
download_roboflow("roboflow-jvuqo", "basketball-player-detection-2", <version>, "coco", Path("data/datasets/players"))
print(remap_split_dirs(Path("data/datasets/players")))
download_roboflow("roboflow-jvuqo", "basketball-court-detection-2", 19, "coco", Path("data/datasets/court"))
```

4. Train and export:

```
!python -m training.train_detector --dataset-dir data/datasets/players --dataset-version <version> \
    --model small --resolution 512 --epochs 30 --batch 4 --grad-accum 4 --workers 2 --device cuda --out models/detector
!python -m training.train_court --dataset-dir data/datasets/court --epochs 40 --batch 4 --grad-accum 4 --workers 2 --device cuda --out models/court
!python -m training.export_siglip --out models/siglip/vision.onnx
```

5. Zip and download: `!zip -r models-$(date +%F).zip models -x "models/*/run/*"`, then put
   the zip's contents under `models/` on the Mac (or publish it as a GitHub release asset
   and run `scripts/download_models.py`).

A second detector size for the benchmark: rerun step 4 with `--model nano --resolution 384
--out models/detector-nano` and compare with `scripts/bench_perceiver.py`.

## Mac (fallback)

```
./.venv/bin/pip install -e ".[train]"
./.venv/bin/python -m training.train_detector --dataset-dir data/datasets/players --model small \
    --resolution 512 --epochs 30 --batch 2 --grad-accum 8 --device mps --out models/detector
```

Expect hours rather than minutes on an M1 Pro.

## Licences

Both datasets are CC BY 4.0: the manifest written next to each model carries the
attribution line. Check the provenance of the images inside the sets before any
commercial deployment (spec addendum, open question 3).
