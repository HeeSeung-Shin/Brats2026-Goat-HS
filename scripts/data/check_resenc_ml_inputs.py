#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from mlconsensus_common import (
    EXPECTED_CHANNELS,
    EXPECTED_UNLABELED_CASES,
    HS_SEG_NNUNET_PREDICT,
    IMAGES_UN,
    OUT_ROOT,
    RESENC_L_RESULTS,
    RESENC_M_RESULTS,
    case_ids_from_images,
    modalities_for_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate inputs for ResEnc-M/L 5fold pseudo labeling.")
    parser.add_argument("--imagesUn", default=str(IMAGES_UN))
    parser.add_argument("--resenc-m-results", default=str(RESENC_M_RESULTS))
    parser.add_argument("--resenc-l-results", default=str(RESENC_L_RESULTS))
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--nnunet-predict", default=str(HS_SEG_NNUNET_PREDICT))
    parser.add_argument("--expected-cases", type=int, default=EXPECTED_UNLABELED_CASES)
    parser.add_argument("--check-help", action="store_true", help="Run nnUNetv2_predict -h, not just executable checks.")
    return parser.parse_args()


def add_error(errors: list[str], message: str) -> None:
    errors.append(f"ERROR: {message}")


def add_warning(warnings: list[str], message: str) -> None:
    warnings.append(f"WARNING: {message}")


def check_model_checkpoints(root: Path, label: str, errors: list[str]) -> None:
    if not root.is_dir():
        add_error(errors, f"{label} result root does not exist: {root}")
        return
    for fold in range(5):
        checkpoint = root / f"fold_{fold}" / "checkpoint_best.pth"
        if not checkpoint.is_file():
            add_error(errors, f"Missing {label} fold_{fold}/checkpoint_best.pth: {checkpoint}")


def main() -> None:
    args = parse_args()
    images_un = Path(args.imagesUn)
    out_root = Path(args.out_root)
    nnunet_predict = Path(args.nnunet_predict)
    errors: list[str] = []
    warnings: list[str] = []

    if not images_un.is_dir():
        add_error(errors, f"imagesUn does not exist: {images_un}")
        cases: list[str] = []
    else:
        cases = case_ids_from_images(images_un)
        if len(cases) != args.expected_cases:
            add_error(errors, f"imagesUn case count is {len(cases)}, expected {args.expected_cases}")
        for case_id in cases:
            modalities = modalities_for_case(images_un, case_id)
            present = sorted(modalities)
            if present != EXPECTED_CHANNELS:
                add_error(errors, f"{case_id}: expected modalities {EXPECTED_CHANNELS}, found {present}")

    check_model_checkpoints(Path(args.resenc_m_results), "ResEnc-M", errors)
    check_model_checkpoints(Path(args.resenc_l_results), "ResEnc-L", errors)

    if not nnunet_predict.is_file():
        add_error(errors, f"nnUNetv2_predict was not found: {nnunet_predict}")
    elif not os.access(nnunet_predict, os.X_OK):
        add_error(errors, f"nnUNetv2_predict is not executable: {nnunet_predict}")
    elif args.check_help:
        result = subprocess.run([str(nnunet_predict), "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            add_error(errors, f"nnUNetv2_predict -h failed with exit code {result.returncode}: {result.stderr[:500]}")

    if out_root.exists():
        add_warning(warnings, f"OUT_ROOT already exists. Scripts will not overwrite it unless --overwrite is used: {out_root}")

    for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        if not os.environ.get(name):
            add_warning(warnings, f"{name} is not set in the current shell. Bash scripts export the expected default.")

    print("ResEnc-M/L pseudo labeling input check")
    print("=" * 48)
    print(f"python: {sys.executable}")
    print(f"imagesUn: {images_un}")
    print(f"imagesUn cases: {len(cases)}")
    print(f"ResEnc-M results: {Path(args.resenc_m_results)}")
    print(f"ResEnc-L results: {Path(args.resenc_l_results)}")
    print(f"nnUNetv2_predict: {nnunet_predict}")
    print(f"OUT_ROOT: {out_root}")
    print("")

    for warning in warnings:
        print(warning)
    for error in errors:
        print(error)

    if errors:
        raise SystemExit(1)
    print("CHECK_PASSED")


if __name__ == "__main__":
    main()
