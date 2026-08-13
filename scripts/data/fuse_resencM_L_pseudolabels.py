#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from brats_regions import (
    class_probabilities_to_region_masks,
    fuse_teacher_region_probabilities,
    region_masks_to_labels,
)
from mlconsensus_common import (
    FILE_ENDING,
    IMAGES_UN,
    OUT_ROOT,
    dice,
    ensure_dir,
    labels_to_regions,
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
    parser.add_argument("--prob_spatial_permutation", default="auto", choices=["auto", "none"])
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--case-list", default=None, help="Optional text file with case IDs to process.")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(
        et_threshold=0.50,
        tc_threshold=0.50,
        wt_threshold=0.50,
        agreement_region_threshold=0.50,
        w_et_m=0.70,
        w_tc_m=0.50,
        w_wt_m=0.40,
        highconf_threshold=0.75,
        min_mean_fg_confidence=0.70,
        max_mean_fg_normalized_entropy=0.35,
        min_mean_fg_margin=0.20,
        min_highconf_fg_fraction=0.60,
        min_foreground_ratio=0.00001,
        max_foreground_ratio=0.30,
        min_dice_ml_wt=0.85,
        min_dice_ml_tc=0.70,
        min_dice_ml_et=0.50,
        et_absent_agreement_ok=True,
        et_one_sided_small_volume_mm3=5.0,
        connectivity=26,
        min_wt_component_volume_mm3=5.0,
        min_tc_component_volume_mm3=3.0,
        min_et_component_volume_mm3=1.0,
        component_min_mean_probability=0.15,
        component_min_max_probability=0.30,
        et_suppression_enabled=True,
        et_suppression_volume_mm3=1.0,
        et_suppression_max_probability=0.35,
    )
    return parser.parse_args()


def clean_generated_outputs(out_root: Path, overwrite: bool) -> dict[str, Path]:
    dirs = {
        "fused_labels": out_root / "fused_labels",
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
    return region_masks_to_labels({"ET": mask_et, "TC": mask_tc, "WT": mask_wt})


def region_probabilities(p_m: np.ndarray, p_l: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    return fuse_teacher_region_probabilities(
        p_m,
        p_l,
        weights_m={"ET": args.w_et_m, "TC": args.w_tc_m, "WT": args.w_wt_m},
    )


def initial_label_from_probs(region_probs: dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    thresholds = {"ET": args.et_threshold, "TC": args.tc_threshold, "WT": args.wt_threshold}
    masks = {region: region_probs[region] >= threshold for region, threshold in thresholds.items()}
    return region_masks_to_labels(masks)


def teacher_region_masks(probabilities: np.ndarray, threshold: float = 0.50) -> dict[str, np.ndarray]:
    """Build nested teacher masks from regional probabilities, not argmax labels."""
    return class_probabilities_to_region_masks(probabilities, threshold=threshold)


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
        probability_threshold = float(
            np.asarray(args.et_suppression_max_probability, dtype=region_probs["ET"].dtype)
        )
        if et_volume_mm3 < args.et_suppression_volume_mm3 and max_et_prob < probability_threshold:
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


def region_agreement_metrics(regions_m: dict[str, np.ndarray], regions_l: dict[str, np.ndarray], label_fused: np.ndarray, voxel_volume_mm3: float) -> dict[str, Any]:
    """Compute teacher agreement; ET clinical exceptions remain explicit below."""
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

    status = "strict" if conf_pass and strict_agree else "exclude"
    passes = {
        "confidence_pass": conf_pass,
        "ml_wt_pass": wt_pass,
        "ml_tc_pass": tc_pass,
        "ml_et_pass": et_pass,
        "ml_agreement_pass_strict": strict_agree,
        "I_conf": conf_pass,
        "I_agree": strict_agree,
        "I_strict": conf_pass and strict_agree,
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
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return failure_row(case_id, "missing inputs: " + ", ".join(f"{name}={paths[name]}" for name in missing))

    try:
        p_m, key_m, transform_m = load_probabilities_npz(paths["m_prob"], ref_shape, mode=spatial_mode)
        p_l, key_l, transform_l = load_probabilities_npz(paths["l_prob"], ref_shape, mode=spatial_mode)
        if p_m.shape != p_l.shape:
            return failure_row(case_id, f"M/L probability shape mismatch: {p_m.shape} vs {p_l.shape}")
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

    p4avg = normalize_probabilities(0.5 * p_m + 0.5 * p_l)
    foreground = fused_label > 0
    quality = probability_quality(p4avg, foreground, highconf_threshold=args.highconf_threshold)
    teacher_masks_m = teacher_region_masks(p_m, threshold=args.agreement_region_threshold)
    teacher_masks_l = teacher_region_masks(p_l, threshold=args.agreement_region_threshold)
    agreement = region_agreement_metrics(
        teacher_masks_m, teacher_masks_l, fused_label, voxel_volume_mm3)
    foreground_ratio = float(foreground.mean())
    status_row: dict[str, Any] = {
        "case_id": case_id,
        "selection_status": "pending",
        "failure_reason": "",
        "prob_key_M": key_m,
        "prob_key_L": key_l,
        "prob_transform_M": transform_m,
        "prob_transform_L": transform_l,
        "agreement_source": "thresholded_teacher_region_probabilities",
        "agreement_region_threshold": args.agreement_region_threshold,
        "agreement_hierarchy_correction": True,
        "teacher_argmax_used_for_qc": False,
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
    exclude_cases = sorted(row["case_id"] for row in rows if row.get("selection_status") in {"exclude", "failure"})

    write_csv(dirs["qc"] / "pseudolabel_qc_ml.csv", rows)
    write_case_ids(dirs["manifests"] / "pseudo_labeled_ml_strict_cases.txt", strict_cases)
    write_case_ids(dirs["manifests"] / "excluded_cases.txt", exclude_cases)

    summary = {
        "timestamp": utc_timestamp(),
        "out_root": str(out_root),
        "m_pred_dir": str(Path(args.m_pred_dir)),
        "l_pred_dir": str(Path(args.l_pred_dir)),
        "num_cases_requested": len(cases),
        "num_rows": len(rows),
        "counts": {
            "strict": len(strict_cases),
            "excluded": len(exclude_cases),
            "failure": sum(1 for row in rows if row.get("selection_status") == "failure"),
        },
        "thresholds": {
            "et_threshold": args.et_threshold,
            "tc_threshold": args.tc_threshold,
            "wt_threshold": args.wt_threshold,
            "agreement": {
                "source": "thresholded_teacher_region_probabilities",
                "region_threshold": args.agreement_region_threshold,
                "hierarchy_correction": True,
                "teacher_argmax_used_for_qc": False,
                "min_dice_wt": args.min_dice_ml_wt,
                "min_dice_tc": args.min_dice_ml_tc,
                "min_dice_et": args.min_dice_ml_et,
                "one_sided_et_volume_mm3_strictly_less_than": args.et_one_sided_small_volume_mm3,
            },
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
    print(f"excluded cases: {len(exclude_cases)}")
    print(f"QC CSV: {dirs['qc'] / 'pseudolabel_qc_ml.csv'}")
    print(f"Summary: {dirs['reports'] / 'selection_summary_ml.json'}")


if __name__ == "__main__":
    main()
