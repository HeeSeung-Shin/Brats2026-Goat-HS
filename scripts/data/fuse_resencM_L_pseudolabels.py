#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from mlconsensus_common import (
    FILE_ENDING,
    IMAGES_UN,
    OUT_ROOT,
    align_spatial_array_to_ref,
    dice,
    ensure_dir,
    labels_to_regions,
    load_label_aligned,
    load_nifti,
    load_probabilities_npz,
    normalize_probabilities,
    probability_quality,
    read_case_ids,
    ref_modality_path,
    remove_small_components,
    save_nifti_like,
    summary_stats,
    utc_timestamp,
    write_case_ids,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse ResEnc-M and ResEnc-L pseudo labels with region-wise consensus QC.")
    parser.add_argument("--imagesUn", default=str(IMAGES_UN))
    parser.add_argument("--m_pred_dir", default=str(OUT_ROOT / "raw_resencM_5fold"))
    parser.add_argument("--l_pred_dir", default=str(OUT_ROOT / "raw_resencL_5fold"))
    parser.add_argument("--out_root", default=str(OUT_ROOT))
    parser.add_argument("--mode", default="regionwise", choices=["regionwise"])
    parser.add_argument("--prob_spatial_permutation", default="auto", choices=["auto", "none"])
    parser.add_argument("--et_threshold", type=float, default=0.50)
    parser.add_argument("--tc_threshold", type=float, default=0.50)
    parser.add_argument("--wt_threshold", type=float, default=0.50)
    parser.add_argument("--w_et_m", type=float, default=0.70)
    parser.add_argument("--w_tc_m", type=float, default=0.50)
    parser.add_argument("--w_wt_m", type=float, default=0.40)
    parser.add_argument("--highconf_threshold", type=float, default=0.75)
    parser.add_argument("--min_mean_fg_confidence", type=float, default=0.70)
    parser.add_argument("--max_mean_fg_normalized_entropy", type=float, default=0.35)
    parser.add_argument("--min_mean_fg_margin", type=float, default=0.20)
    parser.add_argument("--min_highconf_fg_fraction", type=float, default=0.60)
    parser.add_argument("--min_foreground_ratio", type=float, default=0.00001)
    parser.add_argument("--max_foreground_ratio", type=float, default=0.30)
    parser.add_argument("--min_dice_ml_wt", type=float, default=0.85)
    parser.add_argument("--min_dice_ml_tc", type=float, default=0.70)
    parser.add_argument("--min_dice_ml_et", type=float, default=0.50)
    parser.add_argument("--et_absent_agreement_ok", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--et_one_sided_small_volume_mm3", type=float, default=5.0)
    parser.add_argument("--relaxed_tc_slack", type=float, default=0.10)
    parser.add_argument("--relaxed_et_slack", type=float, default=0.15)
    parser.add_argument("--connectivity", type=int, default=26, choices=[6, 18, 26])
    parser.add_argument("--min_wt_component_volume_mm3", type=float, default=5.0)
    parser.add_argument("--min_tc_component_volume_mm3", type=float, default=3.0)
    parser.add_argument("--min_et_component_volume_mm3", type=float, default=1.0)
    parser.add_argument("--component_min_mean_probability", type=float, default=0.15)
    parser.add_argument("--component_min_max_probability", type=float, default=0.30)
    parser.add_argument("--et_suppression_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--et_suppression_volume_mm3", type=float, default=1.0)
    parser.add_argument("--et_suppression_max_probability", type=float, default=0.35)
    parser.add_argument("--save_fused_probabilities", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--probability_dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--case-list", default=None, help="Optional text file with case IDs to process.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_generated_outputs(out_root: Path, overwrite: bool) -> dict[str, Path]:
    dirs = {
        "fused_labels": out_root / "fused_labels",
        "fused_probabilities": out_root / "fused_probabilities",
        "qc": out_root / "qc",
        "manifests": out_root / "manifests",
        "reports": out_root / "reports",
    }
    generated_dirs = list(dirs.values())
    nonempty = [path for path in generated_dirs if path.is_dir() and any(path.iterdir())]
    if nonempty and not overwrite:
        joined = "\n  ".join(str(p) for p in nonempty)
        raise RuntimeError(f"Generated output folders already contain files. Re-run with --overwrite to replace them:\n  {joined}")
    if overwrite:
        for path in generated_dirs:
            if path.exists():
                shutil.rmtree(path)
    for path in generated_dirs:
        ensure_dir(path)
    return dirs


def make_label_from_regions(mask_et: np.ndarray, mask_tc: np.ndarray, mask_wt: np.ndarray) -> np.ndarray:
    label = np.zeros(mask_wt.shape, dtype=np.uint8)
    label[np.logical_and(mask_wt, ~mask_tc)] = 2
    label[np.logical_and(mask_tc, ~mask_et)] = 1
    label[mask_et] = 3
    return label


def region_probabilities(p_m: np.ndarray, p_l: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    p_m_bg, p_m_ncr, p_m_ed, p_m_et = p_m[0], p_m[1], p_m[2], p_m[3]
    p_l_bg, p_l_ncr, p_l_ed, p_l_et = p_l[0], p_l[1], p_l[2], p_l[3]
    _ = p_m_bg, p_l_bg
    p_m_regions = {
        "ET": p_m_et,
        "TC": p_m_ncr + p_m_et,
        "WT": p_m_ncr + p_m_ed + p_m_et,
    }
    p_l_regions = {
        "ET": p_l_et,
        "TC": p_l_ncr + p_l_et,
        "WT": p_l_ncr + p_l_ed + p_l_et,
    }
    return {
        "ET": args.w_et_m * p_m_regions["ET"] + (1.0 - args.w_et_m) * p_l_regions["ET"],
        "TC": args.w_tc_m * p_m_regions["TC"] + (1.0 - args.w_tc_m) * p_l_regions["TC"],
        "WT": args.w_wt_m * p_m_regions["WT"] + (1.0 - args.w_wt_m) * p_l_regions["WT"],
    }


def initial_label_from_probs(region_probs: dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    mask_et = region_probs["ET"] >= args.et_threshold
    mask_tc = region_probs["TC"] >= args.tc_threshold
    mask_wt = region_probs["WT"] >= args.wt_threshold
    mask_tc = np.logical_or(mask_tc, mask_et)
    mask_wt = np.logical_or(mask_wt, mask_tc)
    return make_label_from_regions(mask_et, mask_tc, mask_wt)


def apply_component_filtering(label: np.ndarray, region_probs: dict[str, np.ndarray], voxel_volume_mm3: float, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    pre_regions = labels_to_regions(label)
    filtered_wt, wt_stats = remove_small_components(
        pre_regions["WT"],
        region_probs["WT"],
        voxel_volume_mm3,
        args.min_wt_component_volume_mm3,
        args.component_min_mean_probability,
        args.component_min_max_probability,
        args.connectivity,
    )
    filtered_tc, tc_stats = remove_small_components(
        pre_regions["TC"],
        region_probs["TC"],
        voxel_volume_mm3,
        args.min_tc_component_volume_mm3,
        args.component_min_mean_probability,
        args.component_min_max_probability,
        args.connectivity,
    )
    filtered_et, et_stats = remove_small_components(
        pre_regions["ET"],
        region_probs["ET"],
        voxel_volume_mm3,
        args.min_et_component_volume_mm3,
        args.component_min_mean_probability,
        args.component_min_max_probability,
        args.connectivity,
    )

    if args.et_suppression_enabled:
        et_volume_mm3 = float(filtered_et.sum() * voxel_volume_mm3)
        max_et_prob = float(region_probs["ET"][filtered_et].max()) if filtered_et.any() else 0.0
        if et_volume_mm3 < args.et_suppression_volume_mm3 and max_et_prob < args.et_suppression_max_probability:
            filtered_et[:] = False
            et_stats["et_suppressed"] = True
            et_stats["et_suppression_volume_mm3"] = et_volume_mm3
            et_stats["et_suppression_max_probability"] = max_et_prob
        else:
            et_stats["et_suppressed"] = False

    filtered_tc = np.logical_or(filtered_tc, filtered_et)
    filtered_wt = np.logical_or(filtered_wt, filtered_tc)
    filtered_label = make_label_from_regions(filtered_et, filtered_tc, filtered_wt)

    stats: dict[str, Any] = {}
    for prefix, item in (("WT", wt_stats), ("TC", tc_stats), ("ET", et_stats)):
        for key, value in item.items():
            stats[f"component_{prefix}_{key}"] = value
    for region, mask in pre_regions.items():
        stats[f"volume_pre_filter_{region}_voxels"] = int(mask.sum())
        stats[f"volume_pre_filter_{region}_mm3"] = float(mask.sum() * voxel_volume_mm3)
    post_regions = labels_to_regions(filtered_label)
    for region, mask in post_regions.items():
        stats[f"volume_post_filter_{region}_voxels"] = int(mask.sum())
        stats[f"volume_post_filter_{region}_mm3"] = float(mask.sum() * voxel_volume_mm3)
        stats[f"volume_filter_delta_{region}_voxels"] = int(mask.sum()) - int(pre_regions[region].sum())
        stats[f"volume_filter_delta_{region}_mm3"] = float((int(mask.sum()) - int(pre_regions[region].sum())) * voxel_volume_mm3)
    return filtered_label, stats


def region_agreement_metrics(label_m: np.ndarray, label_l: np.ndarray, label_fused: np.ndarray, voxel_volume_mm3: float) -> dict[str, Any]:
    regions_m = labels_to_regions(label_m)
    regions_l = labels_to_regions(label_l)
    regions_fused = labels_to_regions(label_fused)
    metrics: dict[str, Any] = {}
    for region in ("ET", "TC", "WT"):
        metrics[f"dice_ML_{region}"] = dice(regions_m[region], regions_l[region])
        for source, regions in (("M", regions_m), ("L", regions_l), ("fused", regions_fused)):
            voxels = int(regions[region].sum())
            metrics[f"volume_{source}_{region}"] = voxels
            metrics[f"volume_{source}_{region}_mm3"] = float(voxels * voxel_volume_mm3)
    metrics["et_absent_M"] = metrics["volume_M_ET"] == 0
    metrics["et_absent_L"] = metrics["volume_L_ET"] == 0
    metrics["et_absent_fused"] = metrics["volume_fused_ET"] == 0
    return metrics


def confidence_pass(row: dict[str, Any], args: argparse.Namespace) -> bool:
    values = {
        "mean_fg_confidence": row.get("mean_fg_confidence"),
        "mean_fg_normalized_entropy": row.get("mean_fg_normalized_entropy"),
        "mean_fg_margin": row.get("mean_fg_margin"),
        "highconf_fg_fraction": row.get("highconf_fg_fraction"),
        "foreground_ratio_fused": row.get("foreground_ratio_fused"),
    }
    if any(not math.isfinite(float(v)) for v in values.values()):
        return False
    return (
        float(values["mean_fg_confidence"]) >= args.min_mean_fg_confidence
        and float(values["mean_fg_normalized_entropy"]) <= args.max_mean_fg_normalized_entropy
        and float(values["mean_fg_margin"]) >= args.min_mean_fg_margin
        and float(values["highconf_fg_fraction"]) >= args.min_highconf_fg_fraction
        and args.min_foreground_ratio <= float(values["foreground_ratio_fused"]) <= args.max_foreground_ratio
    )


def et_agreement_pass(row: dict[str, Any], args: argparse.Namespace, notes: list[str]) -> bool:
    absent_m = bool(row["et_absent_M"])
    absent_l = bool(row["et_absent_L"])
    if absent_m and absent_l:
        return bool(args.et_absent_agreement_ok)
    if absent_m != absent_l:
        present_volume = float(row["volume_L_ET_mm3"] if absent_m else row["volume_M_ET_mm3"])
        if present_volume < args.et_one_sided_small_volume_mm3:
            notes.append("one_sided_ET_present_small_volume")
            return True
        notes.append("one_sided_ET_present_large_volume")
        return False
    return float(row["dice_ML_ET"]) >= args.min_dice_ml_et


def assign_selection(row: dict[str, Any], args: argparse.Namespace, notes: list[str]) -> tuple[str, dict[str, bool]]:
    conf_pass = confidence_pass(row, args)
    wt_pass = float(row["dice_ML_WT"]) >= args.min_dice_ml_wt
    tc_pass = float(row["dice_ML_TC"]) >= args.min_dice_ml_tc
    et_pass = et_agreement_pass(row, args, notes)
    strict_agree = wt_pass and tc_pass and et_pass

    tc_relaxed = float(row["dice_ML_TC"]) >= max(0.0, args.min_dice_ml_tc - args.relaxed_tc_slack)
    et_relaxed = et_pass or float(row["dice_ML_ET"]) >= max(0.0, args.min_dice_ml_et - args.relaxed_et_slack)
    relaxed_agree = wt_pass and ((tc_pass and et_relaxed) or (et_pass and tc_relaxed))

    if conf_pass and strict_agree:
        status = "strict"
    elif conf_pass and relaxed_agree:
        status = "relaxed"
    else:
        status = "exclude"
    passes = {
        "confidence_pass": conf_pass,
        "ml_wt_pass": wt_pass,
        "ml_tc_pass": tc_pass,
        "ml_et_pass": et_pass,
        "ml_agreement_pass_strict": strict_agree,
        "ml_agreement_pass_relaxed": relaxed_agree,
    }
    return status, passes


def float_or_none(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: float_or_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [float_or_none(v) for v in value]
    return value


def failure_row(case_id: str, reason: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "selection_status": "failure",
        "failure_reason": reason,
        "selected_strict": False,
        "selected_relaxed": False,
        "recommended_pseudo_weight": 0.0,
    }


def process_case(case_id: str, args: argparse.Namespace, dirs: dict[str, Path]) -> dict[str, Any]:
    images_un = Path(args.imagesUn)
    m_dir = Path(args.m_pred_dir)
    l_dir = Path(args.l_pred_dir)
    ref_path = ref_modality_path(images_un, case_id)
    if not ref_path.is_file():
        return failure_row(case_id, f"missing reference modality: {ref_path}")
    ref_img, ref_data = load_nifti(ref_path)
    ref_shape = tuple(int(i) for i in ref_data.shape)
    voxel_volume_mm3 = float(np.prod(ref_img.header.get_zooms()[:3]))
    spatial_mode = args.prob_spatial_permutation

    paths = {
        "m_prob": m_dir / f"{case_id}.npz",
        "l_prob": l_dir / f"{case_id}.npz",
        "m_label": m_dir / f"{case_id}{FILE_ENDING}",
        "l_label": l_dir / f"{case_id}{FILE_ENDING}",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return failure_row(case_id, "missing inputs: " + ", ".join(f"{name}={paths[name]}" for name in missing))

    try:
        p_m, key_m, transform_m = load_probabilities_npz(paths["m_prob"], ref_shape, mode=spatial_mode)
        p_l, key_l, transform_l = load_probabilities_npz(paths["l_prob"], ref_shape, mode=spatial_mode)
        if p_m.shape != p_l.shape:
            return failure_row(case_id, f"M/L probability shape mismatch: {p_m.shape} vs {p_l.shape}")
        label_m, label_transform_m = load_label_aligned(paths["m_label"], ref_shape, mode=spatial_mode)
        label_l, label_transform_l = load_label_aligned(paths["l_label"], ref_shape, mode=spatial_mode)
    except Exception as exc:
        return failure_row(case_id, str(exc))

    notes: list[str] = []
    sum_error_m = float(np.max(np.abs(p_m.sum(axis=0) - 1.0)))
    sum_error_l = float(np.max(np.abs(p_l.sum(axis=0) - 1.0)))
    if sum_error_m > 0.05:
        notes.append(f"M_probability_sum_warning_{sum_error_m:.4f}")
    if sum_error_l > 0.05:
        notes.append(f"L_probability_sum_warning_{sum_error_l:.4f}")

    region_probs = region_probabilities(p_m, p_l, args)
    pre_filter_label = initial_label_from_probs(region_probs, args)
    fused_label, filter_stats = apply_component_filtering(pre_filter_label, region_probs, voxel_volume_mm3, args)

    try:
        saved_label_path = dirs["fused_labels"] / f"{case_id}{FILE_ENDING}"
        save_nifti_like(fused_label, ref_img, saved_label_path, dtype=np.uint8)
        saved_img, saved_data = load_nifti(saved_label_path)
        saved_arr = np.asarray(saved_data)
        if saved_arr.shape != ref_shape:
            return failure_row(case_id, f"saved fused label shape mismatch: {saved_arr.shape} vs {ref_shape}")
        if not np.allclose(saved_img.affine, ref_img.affine):
            return failure_row(case_id, "saved fused label affine mismatch")
        invalid = sorted(set(int(v) for v in np.unique(saved_arr)) - {0, 1, 2, 3})
        if invalid:
            return failure_row(case_id, f"saved fused label has invalid labels: {invalid}")
    except Exception as exc:
        return failure_row(case_id, f"failed to save/validate fused label: {exc}")

    if args.save_fused_probabilities:
        dtype = np.float16 if args.probability_dtype == "float16" else np.float32
        np.savez_compressed(
            dirs["fused_probabilities"] / f"{case_id}.npz",
            p_ET=region_probs["ET"].astype(dtype, copy=False),
            p_TC=region_probs["TC"].astype(dtype, copy=False),
            p_WT=region_probs["WT"].astype(dtype, copy=False),
        )

    p4avg = normalize_probabilities(0.5 * p_m + 0.5 * p_l)
    foreground = fused_label > 0
    quality = probability_quality(p4avg, foreground, highconf_threshold=args.highconf_threshold)
    agreement = region_agreement_metrics(label_m, label_l, fused_label, voxel_volume_mm3)
    foreground_ratio = float(foreground.mean())
    status_row: dict[str, Any] = {
        "case_id": case_id,
        "selection_status": "pending",
        "failure_reason": "",
        "prob_key_M": key_m,
        "prob_key_L": key_l,
        "prob_transform_M": transform_m,
        "prob_transform_L": transform_l,
        "label_transform_M": label_transform_m,
        "label_transform_L": label_transform_l,
        "probability_sum_max_abs_error_M": sum_error_m,
        "probability_sum_max_abs_error_L": sum_error_l,
        "foreground_ratio_fused": foreground_ratio,
        "voxel_volume_mm3": voxel_volume_mm3,
    }
    status_row.update(filter_stats)
    status_row.update(quality)
    status_row.update(agreement)
    status, passes = assign_selection(status_row, args, notes)
    status_row.update(passes)
    status_row["selection_status"] = status
    status_row["selected_strict"] = status == "strict"
    status_row["selected_relaxed"] = status in {"strict", "relaxed"}
    status_row["recommended_pseudo_weight"] = 1.0 if status == "strict" else (0.7 if status == "relaxed" else 0.0)
    status_row["notes"] = ";".join(notes)
    return status_row


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    dirs = clean_generated_outputs(out_root, overwrite=args.overwrite)
    if args.case_list:
        cases = read_case_ids(args.case_list)
    else:
        from mlconsensus_common import case_ids_from_images

        cases = case_ids_from_images(args.imagesUn)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    rows: list[dict[str, Any]] = []
    for index, case_id in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] fuse+qc {case_id}", flush=True)
        rows.append(process_case(case_id, args, dirs))

    strict_cases = sorted(row["case_id"] for row in rows if row.get("selection_status") == "strict")
    relaxed_cases = sorted(row["case_id"] for row in rows if row.get("selection_status") in {"strict", "relaxed"})
    disagreement_cases = sorted(
        row["case_id"]
        for row in rows
        if row.get("selection_status") not in {"failure"}
        and (not row.get("ml_agreement_pass_strict", False) or row.get("notes"))
    )
    exclude_cases = sorted(row["case_id"] for row in rows if row.get("selection_status") in {"exclude", "failure"})

    write_csv(dirs["qc"] / "pseudolabel_qc_ml.csv", rows)
    write_case_ids(dirs["manifests"] / "pseudo_labeled_ml_strict_cases.txt", strict_cases)
    write_case_ids(dirs["manifests"] / "pseudo_labeled_ml_relaxed_cases.txt", relaxed_cases)
    write_case_ids(dirs["manifests"] / "ml_disagreement_cases.txt", disagreement_cases)
    write_case_ids(dirs["manifests"] / "exclude_recommended_cases.txt", exclude_cases)

    summary = {
        "timestamp": utc_timestamp(),
        "out_root": str(out_root),
        "m_pred_dir": str(Path(args.m_pred_dir)),
        "l_pred_dir": str(Path(args.l_pred_dir)),
        "num_cases_requested": len(cases),
        "num_rows": len(rows),
        "counts": {
            "strict": len(strict_cases),
            "relaxed_including_strict": len(relaxed_cases),
            "relaxed_extra": len(set(relaxed_cases) - set(strict_cases)),
            "ml_disagreement_or_warning": len(disagreement_cases),
            "exclude_recommended": len(exclude_cases),
            "failure": sum(1 for row in rows if row.get("selection_status") == "failure"),
        },
        "thresholds": {
            "et_threshold": args.et_threshold,
            "tc_threshold": args.tc_threshold,
            "wt_threshold": args.wt_threshold,
            "w_et_m": args.w_et_m,
            "w_tc_m": args.w_tc_m,
            "w_wt_m": args.w_wt_m,
            "min_mean_fg_confidence": args.min_mean_fg_confidence,
            "max_mean_fg_normalized_entropy": args.max_mean_fg_normalized_entropy,
            "min_mean_fg_margin": args.min_mean_fg_margin,
            "min_highconf_fg_fraction": args.min_highconf_fg_fraction,
            "min_foreground_ratio": args.min_foreground_ratio,
            "max_foreground_ratio": args.max_foreground_ratio,
            "min_dice_ml_wt": args.min_dice_ml_wt,
            "min_dice_ml_tc": args.min_dice_ml_tc,
            "min_dice_ml_et": args.min_dice_ml_et,
            "component_filtering": {
                "connectivity": args.connectivity,
                "min_wt_component_volume_mm3": args.min_wt_component_volume_mm3,
                "min_tc_component_volume_mm3": args.min_tc_component_volume_mm3,
                "min_et_component_volume_mm3": args.min_et_component_volume_mm3,
                "component_min_mean_probability": args.component_min_mean_probability,
                "component_min_max_probability": args.component_min_max_probability,
                "et_suppression_enabled": args.et_suppression_enabled,
            },
        },
        "metric_summaries": {
            "dice_ML_WT": summary_stats([float(row.get("dice_ML_WT", float("nan"))) for row in rows]),
            "dice_ML_TC": summary_stats([float(row.get("dice_ML_TC", float("nan"))) for row in rows]),
            "dice_ML_ET": summary_stats([float(row.get("dice_ML_ET", float("nan"))) for row in rows]),
            "mean_fg_confidence": summary_stats([float(row.get("mean_fg_confidence", float("nan"))) for row in rows]),
            "mean_fg_normalized_entropy": summary_stats([float(row.get("mean_fg_normalized_entropy", float("nan"))) for row in rows]),
            "mean_fg_margin": summary_stats([float(row.get("mean_fg_margin", float("nan"))) for row in rows]),
            "foreground_ratio_fused": summary_stats([float(row.get("foreground_ratio_fused", float("nan"))) for row in rows]),
        },
    }
    write_json(dirs["reports"] / "selection_summary_ml.json", float_or_none(summary))
    write_json(dirs["reports"] / "fusion_args_ml.json", vars(args))

    print("FUSION_QC_DONE")
    print(f"strict cases: {len(strict_cases)}")
    print(f"relaxed cases including strict: {len(relaxed_cases)}")
    print(f"exclude recommended: {len(exclude_cases)}")
    print(f"QC CSV: {dirs['qc'] / 'pseudolabel_qc_ml.csv'}")
    print(f"Summary: {dirs['reports'] / 'selection_summary_ml.json'}")


if __name__ == "__main__":
    main()
