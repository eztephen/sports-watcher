#!/usr/bin/env python
"""Export the SigLIP vision tower (google/siglip-base-patch16-224, Apache-2.0) to ONNX.

    python -m training.export_siglip --out models/siglip/vision.onnx
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run by path or as a module

import argparse
from datetime import date

from training.manifest import write_manifest

MODEL_ID = "google/siglip-base-patch16-224"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("models/siglip/vision.onnx"))
    args = ap.parse_args(argv)

    import torch  # training extra
    from transformers import SiglipVisionModel

    model = SiglipVisionModel.from_pretrained(MODEL_ID).eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    class Pooled(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.inner(pixel_values=pixel_values).pooler_output

    torch.onnx.export(
        Pooled(model),
        torch.zeros(1, 3, 224, 224),
        str(args.out),
        input_names=["pixel_values"],
        output_names=["pooler_output"],
        opset_version=17,
        dynamo=False,
    )
    manifest = write_manifest(
        args.out,
        name="siglip",
        family="siglip",
        size="base-patch16-224",
        resolution=224,
        dataset=MODEL_ID,
        trained_on=date.today().isoformat(),
        licence="Apache-2.0",
        attribution="google/siglip-base-patch16-224 (Apache-2.0)",
    )
    print(f"wrote {args.out} and {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
