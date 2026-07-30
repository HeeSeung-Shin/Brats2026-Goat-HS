#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from mlconsensus_common import (
    DATASET005_ROOT,
    DATASET007_ROOT,
    EXPECTED_CHANNELS,
    FILE_ENDING,
    OUT_ROOT,
    VALID_LABELS,
    case_ids_from_images,
    copy_or_link,
    ensure_dir,
    expected_channels_from_dataset_json,
    load_json,
    load_nifti,
    modalities_for_case,
    read_case_ids,
    read_csv_rows,
    str2bool,
    utc_timestamp,
    valid_label_ids_from_dataset_json,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Dataset007 from Dataset005 GT plus ResEnc-M/L consensus pseudo labels.")
    parser.add_argument("--source_dataset_root", "--source-dataset-root", default=str(DATASET005_ROOT))
    parser.add_argument("--pseudolabel_root", "--pseudolabel-root", default=str(OUT_ROOT))
    parser.add_argument("--target_dataset_root", "--target-dataset-root", default=str(DATASET007_ROOT))
    parser.add_argument("--selection", default="strict", choices=["strict", "relaxed"])
    parser.add_argument("--link_mode", "--link-mode", default="symlink", choices=["symlink", "copy"])
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--copy_unselected_imagesUn", "--copy-unselected-imagesUn", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def ensure_clean_target(target_root: Path, overwrite: bool) -> None:
    if target_root.exists() and any(target_root.iterdir()):
        if not overwrite:
            raise RuntimeError(f"{target_root} already exists and is not empty. Re-run with --overwrite true to replace it.")
        shutil.rmtree(target_root)
    ensure_dir(target_root)


def link_case_images(src_folder: Path, dst_folder: Path, src_case_id: str, dst_case_id: str, channels: list[str], link_mode: str) -> None:
    for channel in channels:
        src = src_folder / f"{src_case_id}_{channel}{FILE_ENDING}"
        if not src.is_file():
            raise RuntimeError(f"Missing modality file: {src}")
        dst = dst_folder / f"{dst_case_id}_{channel}{FILE_ENDING}"
        copy_or_link(src, dst, link_mode)


def validate_label(path: Path, valid_labels: set[int]) -> None:
    if not path.is_file():
        raise RuntimeError(f"Missing label: {path}")
    _, data = load_nifti(path)
    arr = np.asarray(data)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise RuntimeError(f"Label is not 3D: {path}, shape={arr.shape}")
    invalid = sorted(set(int(v) for v in np.unique(arr)) - valid_labels)
    if invalid:
        raise RuntimeError(f"Label contains invalid values {invalid}: {path}")


def qc_by_case(pseudolabel_root: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(pseudolabel_root / "qc" / "pseudolabel_qc_ml.csv")
    return {row["case_id"]: row for row in rows if row.get("case_id")}


def metric(row: dict[str, str], key: str) -> str:
    return row.get(key, "")


def build_manifest_row(
    case_id: str,
    role: str,
    source_image: str,
    source_label: str,
    selected: bool,
    recommended_pseudo_weight: Any,
    qc_row: dict[str, str] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    qc_row = qc_row or {}
    return {
        "case_id": case_id,
        "role": role,
        "source_image": source_image,
        "source_label": source_label,
        "selected": selected,
        "recommended_pseudo_weight": recommended_pseudo_weight,
        "dice_ml_et": metric(qc_row, "dice_ML_ET"),
        "dice_ml_tc": metric(qc_row, "dice_ML_TC"),
        "dice_ml_wt": metric(qc_row, "dice_ML_WT"),
        "mean_fg_confidence": metric(qc_row, "mean_fg_confidence"),
        "mean_fg_entropy": metric(qc_row, "mean_fg_normalized_entropy"),
        "mean_fg_margin": metric(qc_row, "mean_fg_margin"),
        "foreground_ratio": metric(qc_row, "foreground_ratio_fused"),
        "notes": notes or metric(qc_row, "notes"),
    }


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_dataset_root)
    pseudolabel_root = Path(args.pseudolabel_root)
    target_root = Path(args.target_dataset_root)
    images_tr = source_root / "imagesTr"
    labels_tr = source_root / "labelsTr"
    images_un = source_root / "imagesUn"
    fused_labels = pseudolabel_root / "fused_labels"
    manifests = pseudolabel_root / "manifests"

    dataset_json_path = source_root / "dataset.json"
    if not dataset_json_path.is_file():
        raise RuntimeError(f"Missing source dataset.json: {dataset_json_path}")
    dataset_json = load_json(dataset_json_path)
    channels = expected_channels_from_dataset_json(dataset_json) or EXPECTED_CHANNELS
    valid_labels = valid_label_ids_from_dataset_json(dataset_json) or VALID_LABELS

    original_cases = case_ids_from_images(images_tr)
    unlabeled_cases = case_ids_from_images(images_un)
    if not original_cases:
        raise RuntimeError(f"No original labeled cases found in {images_tr}")
    if not unlabeled_cases:
        raise RuntimeError(f"No unlabeled cases found in {images_un}")

    strict_cases = set(read_case_ids(manifests / "pseudo_labeled_ml_strict_cases.txt"))
    relaxed_cases = set(read_case_ids(manifests / "pseudo_labeled_ml_relaxed_cases.txt"))
    exclude_cases = set(read_case_ids(manifests / "exclude_recommended_cases.txt"))
    selected_source_cases = sorted(strict_cases if args.selection == "strict" else relaxed_cases)
    if not selected_source_cases:
        raise RuntimeError(f"No selected pseudo-label cases for selection={args.selection}. Run fusion/QC first.")

    missing_labels = [case_id for case_id in selected_source_cases if not (fused_labels / f"{case_id}{FILE_ENDING}").is_file()]
    if missing_labels:
        raise RuntimeError(f"Missing fused labels for selected cases. first={missing_labels[:5]}")

    ensure_clean_target(target_root, args.overwrite)
    target_images_tr = ensure_dir(target_root / "imagesTr")
    target_labels_tr = ensure_dir(target_root / "labelsTr")
    target_images_un = ensure_dir(target_root / "imagesUn")

    original_set = set(original_cases)
    source_to_target_pseudo: dict[str, str] = {}
    for case_id in selected_source_cases:
        target_case_id = f"pseudo_{case_id}" if case_id in original_set else case_id
        source_to_target_pseudo[case_id] = target_case_id

    rows: list[dict[str, Any]] = []
    qc = qc_by_case(pseudolabel_root)

    for case_id in original_cases:
        link_case_images(images_tr, target_images_tr, case_id, case_id, channels, args.link_mode)
        src_label = labels_tr / f"{case_id}{FILE_ENDING}"
        validate_label(src_label, valid_labels)
        copy_or_link(src_label, target_labels_tr / f"{case_id}{FILE_ENDING}", args.link_mode)
        rows.append(
            build_manifest_row(
                case_id=case_id,
                role="original_labeled",
                source_image=str(images_tr / f"{case_id}_*{FILE_ENDING}"),
                source_label=str(src_label),
                selected=True,
                recommended_pseudo_weight=1.0,
                notes="original_ground_truth",
            )
        )

    for source_case_id, target_case_id in source_to_target_pseudo.items():
        link_case_images(images_un, target_images_tr, source_case_id, target_case_id, channels, args.link_mode)
        pseudo_label = fused_labels / f"{source_case_id}{FILE_ENDING}"
        validate_label(pseudo_label, valid_labels)
        copy_or_link(pseudo_label, target_labels_tr / f"{target_case_id}{FILE_ENDING}", args.link_mode)
        role = "pseudo_labeled_ml_strict" if source_case_id in strict_cases else "pseudo_labeled_ml_relaxed"
        qc_row = qc.get(source_case_id, {})
        weight = qc_row.get("recommended_pseudo_weight") or (1.0 if role.endswith("strict") else 0.7)
        rows.append(
            build_manifest_row(
                case_id=target_case_id,
                role=role,
                source_image=str(images_un / f"{source_case_id}_*{FILE_ENDING}"),
                source_label=str(pseudo_label),
                selected=True,
                recommended_pseudo_weight=weight,
                qc_row=qc_row,
            )
        )

    selected_set = set(selected_source_cases)
    remainder_cases = sorted(set(unlabeled_cases) - selected_set)
    if args.copy_unselected_imagesUn:
        for case_id in remainder_cases:
            link_case_images(images_un, target_images_un, case_id, case_id, channels, args.link_mode)
            role = "excluded" if case_id in exclude_cases else "ml_unlabeled_remainder"
            rows.append(
                build_manifest_row(
                    case_id=case_id,
                    role=role,
                    source_image=str(images_un / f"{case_id}_*{FILE_ENDING}"),
                    source_label="",
                    selected=False,
                    recommended_pseudo_weight=0.0,
                    qc_row=qc.get(case_id, {}),
                    notes="not_selected_for_pseudo_training",
                )
            )

    dataset007_json = dict(dataset_json)
    dataset007_json["name"] = "Dataset007_Brats26_Goat_MLConsensusPseudo"
    dataset007_json["numTraining"] = len(original_cases) + len(selected_source_cases)
    write_json(target_root / "dataset.json", dataset007_json)
    write_csv(target_root / "dataset007_case_manifest.csv", rows)

    metadata = {
        "timestamp": utc_timestamp(),
        "dataset_name": "Dataset007_Brats26_Goat_MLConsensusPseudo",
        "selection": args.selection,
        "link_mode": args.link_mode,
        "source_dataset_root": str(source_root),
        "pseudolabel_root": str(pseudolabel_root),
        "target_dataset_root": str(target_root),
        "original_labeled_cases": original_cases,
        "selected_source_cases": selected_source_cases,
        "selected_target_cases": [source_to_target_pseudo[c] for c in selected_source_cases],
        "source_to_target_pseudo": source_to_target_pseudo,
        "unlabeled_remainder_cases": remainder_cases,
        "exclude_recommended_cases": sorted(exclude_cases),
        "counts": {
            "original_labeled_cases": len(original_cases),
            "selected_pseudo_cases": len(selected_source_cases),
            "unlabeled_remainder_cases": len(remainder_cases),
            "numTraining": len(original_cases) + len(selected_source_cases),
        },
    }
    write_json(target_root / "dataset007_metadata.json", metadata)

    print(f"Dataset007 created at: {target_root}")
    print(f"selection: {args.selection}")
    print(f"original labeled cases: {len(original_cases)}")
    print(f"selected pseudo cases: {len(selected_source_cases)}")
    print(f"imagesUn remainder linked: {len(remainder_cases) if args.copy_unselected_imagesUn else 0}")
    print(f"manifest: {target_root / 'dataset007_case_manifest.csv'}")


if __name__ == "__main__":
    main()
