#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check the final trainer runtime without requiring a particular GPU model."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "python": "3.11.15",
    "torch": "2.11.0+cu130",
    "torch_cuda": "13.0",
    "nnunet": "2.6.4",
    "dynamic-network-architectures": "0.4.3",
    "batchgenerators": "0.25.1",
    "batchgeneratorsv2": "0.3.2",
}
NNUNET_COMMIT = "f6d221d1b79cd2173650f78f97ecfee273e0cf86"
PAPER_GPU = "NVIDIA GeForce RTX 5090"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict", action="store_true", help="Fail on software, CUDA, AMP, or import differences."
    )
    parser.add_argument(
        "--exact-hardware",
        action="store_true",
        help="Additionally require the RTX 5090 used for the paper training run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[tuple[str, str, str, bool]] = []

    def add(name: str, actual: object, expected: object, ok: bool) -> None:
        rows.append((name, str(actual), str(expected), bool(ok)))

    version = platform.python_version()
    add("Python", version, EXPECTED["python"], version == EXPECTED["python"])
    try:
        import torch

        add("PyTorch", torch.__version__, EXPECTED["torch"], torch.__version__ == EXPECTED["torch"])
        add("torch CUDA", torch.version.cuda, EXPECTED["torch_cuda"], torch.version.cuda == EXPECTED["torch_cuda"])
        cuda_ok = torch.cuda.is_available()
        add("CUDA available", cuda_ok, True, cuda_ok)
        gpu = torch.cuda.get_device_name(0) if cuda_ok else "unavailable"
        add("CUDA device", gpu, "any CUDA-capable NVIDIA GPU", cuda_ok)
        if cuda_ok:
            try:
                left = torch.ones((16, 16), device="cuda", dtype=torch.float32)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    result = left @ left
                amp_ok = result.dtype == torch.float16 and torch.isfinite(result).all().item()
                add("CUDA AMP", result.dtype, torch.float16, amp_ok)
            except Exception as exc:
                add("CUDA AMP", repr(exc), "float16 autocast succeeds", False)
        else:
            add("CUDA AMP", "not tested", "float16 autocast succeeds", False)
        if args.exact_hardware:
            add("paper GPU", gpu, PAPER_GPU, gpu == PAPER_GPU)
    except Exception as exc:
        add("PyTorch import", repr(exc), "import succeeds", False)

    for package in (
        "nnunetv2",
        "dynamic-network-architectures",
        "batchgenerators",
        "batchgeneratorsv2",
    ):
        expected = EXPECTED["nnunet" if package == "nnunetv2" else package]
        try:
            actual = importlib.metadata.version(package)
            add(package, actual, expected, actual == expected)
        except Exception as exc:
            add(package, repr(exc), expected, False)

    checkout = REPO_ROOT / "third_party" / "nnUNet"
    if (checkout / ".git").is_dir():
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        add("nnU-Net commit", actual, NNUNET_COMMIT, actual == NNUNET_COMMIT)

    try:
        from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot import (
            BottleneckSoftMoEAdapter,
            nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot,
        )

        add(
            "trainer import",
            nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot.__name__,
            "nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot",
            True,
        )
        add("dense adapter import", BottleneckSoftMoEAdapter.__name__, "BottleneckSoftMoEAdapter", True)
    except Exception as exc:
        add("trainer import", repr(exc), "imports", False)

    failed = [row for row in rows if not row[3]]
    for name, actual, expected, ok in rows:
        print(f"{'PASS' if ok else 'FAIL'} {name}: actual={actual!r} expected={expected!r}")
    if failed and (args.strict or args.exact_hardware):
        print(f"FAIL: {len(failed)} required environment check(s) failed.", file=sys.stderr)
        return 1
    if failed:
        print(f"WARNING: {len(failed)} non-strict environment difference(s).")
    else:
        print("PASS: final ResEnc-M K=4 runtime checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
