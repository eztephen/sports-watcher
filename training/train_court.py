#!/usr/bin/env python
"""Fine-tune the RF-DETR keypoint preview model on the basketball court set (33 keypoints).

Colab:  python training/train_court.py --dataset-dir data/datasets/court --epochs 40 \
            --batch 8 --device cuda --out models/court
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

from training.manifest import write_manifest

DATASET = "roboflow-jvuqo/basketball-court-detection-2"
ATTRIBUTION = "Roboflow Universe basketball-court-detection-2 (roboflow-jvuqo), CC BY 4.0"
NUM_KEYPOINTS = 33


def keypoint_names(dataset_dir: Path) -> list[str]:
    coco = json.loads((dataset_dir / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    for cat in coco["categories"]:
        names = cat.get("keypoints")
        if names:
            return [str(n) for n in names]
    raise SystemExit("no category with a keypoints list in the train annotations")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--dataset-version", default="19")
    ap.add_argument("--resolution", type=int, default=576, help="divisible by 32")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("models/court"))
    args = ap.parse_args(argv)
    if args.resolution % 32:
        raise SystemExit("--resolution must be divisible by 32")
    names = keypoint_names(args.dataset_dir)
    if len(names) != NUM_KEYPOINTS:
        raise SystemExit(f"expected {NUM_KEYPOINTS} keypoints, the dataset lists {len(names)}")

    from rfdetr import RFDETRKeypointPreview  # training extra

    model = RFDETRKeypointPreview(
        num_classes=1, num_keypoints_per_class=[NUM_KEYPOINTS], resolution=args.resolution
    )
    model.train(
        dataset_dir=str(args.dataset_dir),
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        device=args.device,
        output_dir=str(args.out / "run"),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    model.export(
        output_dir=str(args.out),
        format="onnx",
        shape=(args.resolution, args.resolution),
        batch_size=1,
        opset_version=17,
        simplify=False,
    )
    final = args.out / "model.onnx"
    shutil.move(str(args.out / "inference_model.onnx"), str(final))
    manifest = write_manifest(
        final,
        name="court",
        family="rfdetr-keypoint",
        size="keypoint-preview",
        resolution=args.resolution,
        classes={1: "court"},
        keypoints=names,
        dataset=DATASET,
        dataset_version=str(args.dataset_version),
        trained_on=date.today().isoformat(),
        licence="Apache-2.0 (model); CC BY 4.0 (data)",
        attribution=ATTRIBUTION,
        output_boxes="dets",
        output_logits="labels",
        output_keypoints="keypoints",
    )
    print(f"wrote {final} and {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
