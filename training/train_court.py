#!/usr/bin/env python
"""Fine-tune the RF-DETR keypoint preview model on the basketball court set (33 keypoints).

Colab:  python -m training.train_court --dataset-dir data/datasets/court --epochs 40 \
            --batch 8 --device cuda --out models/court
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run by path or as a module

import argparse
import json
import shutil
from datetime import date

from training.manifest import logit_classes, write_manifest

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


def untie_duplicate_rows(param, generator=None) -> int:
    """Redraw, in place, every row of ``param`` that exactly copies an earlier row.

    RF-DETR resizes its 17-keypoint tables to 33 rows by tiling, so slot ``i + 17`` starts
    as slot ``i``. Such twins predict the same point with a 50/50 confidence and a few
    epochs do not pull them apart. A fresh row is drawn with the mean and spread of the
    table's original rows. Returns the number of rows redrawn."""
    import torch

    with torch.no_grad():
        seen: dict[bytes, int] = {}
        copies = []
        for i, row in enumerate(param.detach().cpu()):
            key = row.numpy().tobytes()
            if key in seen:
                copies.append(i)
            else:
                seen[key] = i
        if not copies:
            return 0
        originals = param[sorted(seen.values())]
        mean = originals.mean()
        std = originals.std() if originals.numel() > 1 else originals.new_tensor(1.0)
        noise = torch.randn((len(copies), *param.shape[1:]), generator=generator, dtype=param.dtype)
        param[copies] = (mean + std * noise.to(param.device)).to(param.dtype)
    return len(copies)


def untie_keypoint_slots(module, num_keypoints: int) -> dict[str, int]:
    """Untie every per-keypoint table (``num_keypoints`` rows, two or more dimensions)."""
    out: dict[str, int] = {}
    for name, param in module.named_parameters():
        if param.dim() >= 2 and param.shape[0] == num_keypoints:
            n = untie_duplicate_rows(param)
            if n:
                out[name] = n
    return out


def install_untie_hook(num_keypoints: int) -> list[dict[str, int]]:
    """Untie the tiled keypoint slots whenever RF-DETR resizes its keypoint head.

    The resize happens while the pretrained weights load, at a point the training API
    does not expose, so the model class's resize method is wrapped. Returns a list that
    receives one report per resize to ``num_keypoints`` slots."""
    from rfdetr.models.lwdetr import LWDETR

    reports: list[dict[str, int]] = []
    original = LWDETR.reinitialize_keypoint_head

    def wrapped(self, num_keypoints_per_class):
        original(self, num_keypoints_per_class)
        if num_keypoints_per_class and sum(num_keypoints_per_class) == num_keypoints:
            report = untie_keypoint_slots(self, num_keypoints)
            reports.append(report)
            print(f"untied tiled keypoint slots: {report}")

    LWDETR.reinitialize_keypoint_head = wrapped
    return reports


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
    ap.add_argument("--workers", type=int, default=4, help="data loader workers")
    ap.add_argument("--out", type=Path, default=Path("models/court"))
    args = ap.parse_args(argv)
    if args.resolution % 32:
        raise SystemExit("--resolution must be divisible by 32")
    names = keypoint_names(args.dataset_dir)
    if len(names) != NUM_KEYPOINTS:
        raise SystemExit(f"expected {NUM_KEYPOINTS} keypoints, the dataset lists {len(names)}")

    from rfdetr import RFDETRKeypointPreview  # training extra

    untied = install_untie_hook(NUM_KEYPOINTS)
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
        num_workers=args.workers,
        output_dir=str(args.out / "run"),
    )
    if not any(untied):
        raise SystemExit("the keypoint head was never untied: the model would keep its twin slots; not exporting")
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
        name="court",
        family="rfdetr-keypoint",
        size="keypoint-preview",
        resolution=args.resolution,
        classes=logit_classes(["court"]),
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
