#!/usr/bin/env python
"""Fine-tune RF-DETR on the basketball player set and export ONNX plus a manifest.

Colab:  python -m training.train_detector --dataset-dir data/datasets/players --model small \
            --resolution 512 --epochs 30 --batch 8 --grad-accum 2 --device cuda --out models/detector
Mac:    same with --device mps --batch 2 --grad-accum 8
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run by path or as a module

import argparse
import shutil
from datetime import date

from training.datasets import TARGET_CLASSES
from training.manifest import write_manifest

DATASET = "roboflow-jvuqo/basketball-player-detection-2"
ATTRIBUTION = "Roboflow Universe basketball-player-detection-2 (roboflow-jvuqo), CC BY 4.0"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, default=None, help="required unless --checkpoint is given")
    ap.add_argument("--dataset-version", default="unknown")
    ap.add_argument("--model", choices=["nano", "small", "medium"], default="small")
    ap.add_argument("--resolution", type=int, default=512, help="divisible by 32")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=4, help="data loader workers")
    ap.add_argument("--checkpoint", type=Path, default=None, help="export this checkpoint instead of training")
    ap.add_argument("--note", default=None, help="free text recorded in the manifest")
    ap.add_argument("--out", type=Path, default=Path("models/detector"))
    args = ap.parse_args(argv)
    if args.resolution % 32:
        raise SystemExit("--resolution must be divisible by 32")
    if args.checkpoint is None and args.dataset_dir is None:
        raise SystemExit("--dataset-dir is required unless --checkpoint is given")

    from rfdetr import RFDETRMedium, RFDETRNano, RFDETRSmall  # training extra

    cls = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium}[args.model]
    if args.checkpoint is not None:
        model = cls(resolution=args.resolution, pretrain_weights=str(args.checkpoint))
    else:
        model = cls(resolution=args.resolution)
        model.train(
            dataset_dir=str(args.dataset_dir),
            epochs=args.epochs,
            batch_size=args.batch,
            grad_accum_steps=args.grad_accum,
            lr=args.lr,
            device=args.device,
            num_workers=args.workers,
            output_dir=str(args.out / "run"),
        )
    args.out.mkdir(parents=True, exist_ok=True)
    model.export(
        output_dir=str(args.out),
        format="onnx",
        shape=(args.resolution, args.resolution),
        batch_size=1,
        opset_version=17,
        output_name="model",
    )
    final = args.out / "model.onnx"
    if not final.is_file():  # older rfdetr releases ignore output_name and pick their own
        candidates = sorted(args.out.glob("*.onnx"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise SystemExit(f"no .onnx written under {args.out}")
        shutil.move(str(candidates[-1]), str(final))
    manifest = write_manifest(
        final,
        name="detector",
        family="rfdetr",
        size=args.model,
        resolution=args.resolution,
        classes={i + 1: n for i, n in enumerate(TARGET_CLASSES)},
        dataset=DATASET,
        dataset_version=str(args.dataset_version),
        trained_on=date.today().isoformat(),
        licence="Apache-2.0 (model); CC BY 4.0 (data)",
        attribution=ATTRIBUTION,
        output_boxes="dets",
        output_logits="labels",
        note=args.note,
    )
    print(f"wrote {final} and {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
