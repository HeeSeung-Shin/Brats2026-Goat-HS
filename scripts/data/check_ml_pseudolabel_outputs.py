#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from mlconsensus_common import (
    EXPECTED_UNLABELED_CASES,
    FILE_ENDING,
    IMAGES_UN,
    OUT_ROOT,
    align_spatial_array_to_ref,
    case_ids_from_images,
    ensure_dir,
    labels_to_regions,
    load_label_aligned,
    load_nifti,
    load_probabilities_npz,
    read_case_ids,
    read_csv_rows,
    ref_modality_path,
    summary_stats,
    utc_timestamp,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ResEnc-M/L consensus pseudo-label outputs.")
    parser.add_argument("--imagesUn", default=str(IMAGES_UN))
    parser.add_argument("--m_pred_dir", default=str(OUT_ROOT / "raw_resencM_5fold"))
    parser.add_argument("--l_pred_dir", default=str(OUT_ROOT / "raw_resencL_5fold"))
    parser.add_argument("--out_root", default=str(OUT_ROOT))
    parser.add_argument("--expected_cases", type=int, default=EXPECTED_UNLABELED_CASES)
    parser.add_argument("--prob_spatial_permutation", default="auto", choices=["auto", "none"])
    parser.add_argument("--max_cases", type=int, default=None)
    return parser.parse_args()


def add_error(errors: list[str], message: str) -> None:
    errors.append(f"ERROR: {message}")


def count_files(folder: Path, suffix: str) -> int:
    return len(list(folder.glob(f"*{suffix}")))


def region_volume_row(prefix: str, label: np.ndarray, voxel_volume_mm3: float) -> dict[str, float | int]:
    row: dict[str, float | int] = {}
    for region, mask in labels_to_regions(label).items():
        voxels = int(mask.sum())
        row[f"{prefix}_{region}_voxels"] = voxels
        row[f"{prefix}_{region}_mm3"] = float(voxels * voxel_volume_mm3)
    return row


def finite_npz(path: Path) -> bool:
    with np.load(path) as data:
        for key in data.keys():
            arr = np.asarray(data[key])
            if not np.isfinite(arr).all():
                return False
    return True


def main() -> None:
    args = parse_args()
    images_un = Path(args.imagesUn)
    m_dir = Path(args.m_pred_dir)
    l_dir = Path(args.l_pred_dir)
    out_root = Path(args.out_root)
    fused_dir = out_root / "fused_labels"
    fused_prob_dir = out_root / "fused_probabilities"
    manifests_dir = out_root / "manifests"
    reports_dir = ensure_dir(out_root / "reports")
    errors: list[str] = []

    cases = case_ids_from_images(images_un)
    if len(cases) != args.expected_cases:
        add_error(errors, f"imagesUn case count is {len(cases)}, expected {args.expected_cases}")
    cases_to_check = cases[: args.max_cases] if args.max_cases is not None else cases

    counts = {
        "m_labels": count_files(m_dir, FILE_ENDING),
        "m_npz": count_files(m_dir, ".npz"),
        "l_labels": count_files(l_dir, FILE_ENDING),
        "l_npz": count_files(l_dir, ".npz"),
        "fused_labels": count_files(fused_dir, FILE_ENDING),
        "fused_probabilities": count_files(fused_prob_dir, ".npz") if fused_prob_dir.is_dir() else 0,
        "strict_selected": len(read_case_ids(manifests_dir / "pseudo_labeled_ml_strict_cases.txt")),
        "relaxed_selected": len(read_case_ids(manifests_dir / "pseudo_labeled_ml_relaxed_cases.txt")),
        "ml_disagreement": len(read_case_ids(manifests_dir / "ml_disagreement_cases.txt")),
        "exclude_recommended": len(read_case_ids(manifests_dir / "exclude_recommended_cases.txt")),
    }
    for key in ("m_labels", "m_npz", "l_labels", "l_npz", "fused_labels"):
        if counts[key] != args.expected_cases:
            add_error(errors, f"{key} count is {counts[key]}, expected {args.expected_cases}")

    qc_rows = read_csv_rows(out_root / "qc" / "pseudolabel_qc_ml.csv")
    if qc_rows and len(qc_rows) != counts["fused_labels"]:
        add_error(errors, f"QC CSV row count is {len(qc_rows)}, fused label count is {counts['fused_labels']}")

    volume_rows: list[dict[str, Any]] = []
    for index, case_id in enumerate(cases_to_check, start=1):
        ref_path = ref_modality_path(images_un, case_id)
        if not ref_path.is_file():
            add_error(errors, f"{case_id}: missing reference modality {ref_path}")
            continue
        ref_img, ref_data = load_nifti(ref_path)
        ref_shape = tuple(int(i) for i in ref_data.shape)
        voxel_volume_mm3 = float(np.prod(ref_img.header.get_zooms()[:3]))

        fused_path = fused_dir / f"{case_id}{FILE_ENDING}"
        m_label_path = m_dir / f"{case_id}{FILE_ENDING}"
        l_label_path = l_dir / f"{case_id}{FILE_ENDING}"
        try:
            fused_img, fused_data = load_nifti(fused_path)
            fused = np.asarray(fused_data)
            if fused.ndim == 4 and fused.shape[-1] == 1:
                fused = fused[..., 0]
            fused, _ = align_spatial_array_to_ref(fused, ref_shape, mode=args.prob_spatial_permutation)
            invalid = sorted(set(int(v) for v in np.unique(fused)) - {0, 1, 2, 3})
            if invalid:
                add_error(errors, f"{case_id}: fused label has invalid labels {invalid}")
            if not np.allclose(fused_img.affine, ref_img.affine):
                add_error(errors, f"{case_id}: fused label affine mismatch")
            m_label, _ = load_label_aligned(m_label_path, ref_shape, mode=args.prob_spatial_permutation)
            l_label, _ = load_label_aligned(l_label_path, ref_shape, mode=args.prob_spatial_permutation)
            m_prob, _, _ = load_probabilities_npz(m_dir / f"{case_id}.npz", ref_shape, mode=args.prob_spatial_permutation)
            l_prob, _, _ = load_probabilities_npz(l_dir / f"{case_id}.npz", ref_shape, mode=args.prob_spatial_permutation)
            if not np.isfinite(m_prob).all():
                add_error(errors, f"{case_id}: M probability has NaN/Inf")
            if not np.isfinite(l_prob).all():
                add_error(errors, f"{case_id}: L probability has NaN/Inf")
            fused_prob_path = fused_prob_dir / f"{case_id}.npz"
            if fused_prob_path.is_file() and not finite_npz(fused_prob_path):
                add_error(errors, f"{case_id}: fused probability has NaN/Inf")

            row: dict[str, Any] = {"case_id": case_id}
            row.update(region_volume_row("M", m_label, voxel_volume_mm3))
            row.update(region_volume_row("L", l_label, voxel_volume_mm3))
            row.update(region_volume_row("fused", fused, voxel_volume_mm3))
            for region in ("ET", "TC", "WT"):
                row[f"delta_M_minus_L_{region}_mm3"] = float(row[f"M_{region}_mm3"] - row[f"L_{region}_mm3"])
                row[f"delta_fused_minus_M_{region}_mm3"] = float(row[f"fused_{region}_mm3"] - row[f"M_{region}_mm3"])
                row[f"delta_fused_minus_L_{region}_mm3"] = float(row[f"fused_{region}_mm3"] - row[f"L_{region}_mm3"])
            volume_rows.append(row)
        except Exception as exc:
            add_error(errors, f"{case_id}: validation failed: {exc}")
        if index % 50 == 0:
            print(f"checked {index}/{len(cases_to_check)} cases", flush=True)

    volume_summary: dict[str, Any] = {}
    for region in ("ET", "TC", "WT"):
        for source in ("M", "L", "fused"):
            volume_summary[f"{source}_{region}_mm3"] = summary_stats([float(row[f"{source}_{region}_mm3"]) for row in volume_rows])
        for delta in ("delta_M_minus_L", "delta_fused_minus_M", "delta_fused_minus_L"):
            volume_summary[f"{delta}_{region}_mm3"] = summary_stats([float(row[f"{delta}_{region}_mm3"]) for row in volume_rows])

    summary = {
        "timestamp": utc_timestamp(),
        "imagesUn": str(images_un),
        "out_root": str(out_root),
        "expected_cases": args.expected_cases,
        "checked_cases": len(cases_to_check),
        "counts": counts,
        "volume_summary": volume_summary,
        "num_errors": len(errors),
        "errors": errors[:200],
    }
    write_json(reports_dir / "check_summary_ml.json", summary)

    print("ResEnc-M/L pseudo-label output check")
    print("=" * 48)
    for key, value in counts.items():
        print(f"{key}: {value}")
    print(f"checked cases: {len(cases_to_check)}")
    print(f"summary: {reports_dir / 'check_summary_ml.json'}")
    if errors:
        for error in errors[:50]:
            print(error)
        raise SystemExit(1)
    print("CHECK_ML_OUTPUTS_PASSED")


if __name__ == "__main__":
    main()
