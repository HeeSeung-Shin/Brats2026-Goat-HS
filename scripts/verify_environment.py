#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
SOURCE_HASHES = {
    "src/nnunet_overlays/nnunetv2/training/nnUNetTrainer/nnUNetTrainerNoDeepSupervision.py": "d89f051248ff58f3af7e5e9a4456751949d89d08c710f9e4722e0bead890d564",
    "src/nnunet_overlays/nnunetv2/training/nnUNetTrainer/variants/nnUNetTrainerPseudoWeighted_500ep_LR5e4.py": "856d6cf33b698afd4863aa5d0afff74fdc2cb8e8a52320423c3e0f41419e96f3",
    "src/nnunet_overlays/nnunetv2/training/nnUNetTrainer/variants/nnUNetTrainer_D007ETAwareRegionLoss500.py": "9fe0a480ca74021a9956c610581106bf4b009641b81d746e006c8c53255d28e9",
    "src/nnunet_overlays/nnunetv2/training/nnUNetTrainer/variants/nnUNetTrainer_D007ETTCRegionAuxHead500.py": "4072edcc05d9136942e380efe659b298f04418f53488b78a3161f5e0ef24a096",
    "src/nnunet_overlays/nnunetv2/training/nnUNetTrainer/variants/nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot.py": "3fb3e6de8ce5669583c09263b24f6c82abcdb592b6a880efafedf859d74846c6",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    rows: list[tuple[str, str, str, bool]] = []

    def add(name: str, actual: object, expected: object, ok: bool) -> None:
        rows.append((name, str(actual), str(expected), bool(ok)))

    add("Python", platform.python_version(), EXPECTED["python"], platform.python_version() == EXPECTED["python"])
    try:
        import torch
        add("PyTorch", torch.__version__, EXPECTED["torch"], torch.__version__ == EXPECTED["torch"])
        add("torch CUDA", torch.version.cuda, EXPECTED["torch_cuda"], torch.version.cuda == EXPECTED["torch_cuda"])
        cuda_ok = torch.cuda.is_available()
        add("CUDA available", cuda_ok, True, cuda_ok)
        gpu = torch.cuda.get_device_name(0) if cuda_ok else "unavailable"
        add("GPU", gpu, "NVIDIA GeForce RTX 5090", gpu == "NVIDIA GeForce RTX 5090")
    except Exception as exc:
        add("PyTorch import", repr(exc), "import succeeds", False)

    for package in ("nnunetv2", "dynamic-network-architectures", "batchgenerators", "batchgeneratorsv2"):
        expected = EXPECTED["nnunet" if package == "nnunetv2" else package]
        try:
            actual = importlib.metadata.version(package)
            add(package, actual, expected, actual == expected)
        except Exception as exc:
            add(package, repr(exc), expected, False)

    for relative, expected in SOURCE_HASHES.items():
        path = REPO_ROOT / relative
        actual = digest(path) if path.is_file() else "missing"
        add(relative, actual, expected, actual == expected)

    checkout = REPO_ROOT / "third_party" / "nnUNet"
    if (checkout / ".git").is_dir():
        actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        add("nnU-Net commit", actual, NNUNET_COMMIT, actual == NNUNET_COMMIT)

    try:
        from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot import (
            BottleneckSoftMoEAdapter,
            nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot,
        )
        add("trainer import", nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot.__name__,
            "nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot", True)
        add("SoftMoE adapter import", BottleneckSoftMoEAdapter.__name__, "BottleneckSoftMoEAdapter", True)
    except Exception as exc:
        add("trainer import", repr(exc), "imports", False)

    failed = [row for row in rows if not row[3]]
    for name, actual, expected, ok in rows:
        print(f"{'PASS' if ok else 'FAIL'} {name}: actual={actual!r} expected={expected!r}")
    if failed and args.strict:
        print(f"FAIL: {len(failed)} strict environment check(s) failed.", file=sys.stderr)
        return 1
    if failed:
        print(f"WARNING: {len(failed)} non-strict environment difference(s).")
    else:
        print("PASS: audited ResEnc-M SoftMoE K=4 environment matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
