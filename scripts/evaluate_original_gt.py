#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, distance_transform_edt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BraTS ET/TC/WT metrics on original GT validation cases only."
    )
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--dataset005-labels", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--fold", default="all", help="Fold index 0-4 or 'all'.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--compute-hd95", action="store_true")
    return parser.parse_args()


def read_seg(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.uint8, copy=False)
    spacing_xyz = tuple(float(x) for x in img.GetSpacing())
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    return arr, spacing_zyx


def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    pred = pred.astype(bool, copy=False)
    ref = ref.astype(bool, copy=False)
    pred_sum = int(pred.sum())
    ref_sum = int(ref.sum())
    if pred_sum == 0 and ref_sum == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, ref).sum() / max(1, pred_sum + ref_sum))


def surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool, copy=False)
    if not mask.any():
        return mask
    eroded = binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def hd95(pred: np.ndarray, ref: np.ndarray, spacing: tuple[float, float, float]) -> float:
    pred = pred.astype(bool, copy=False)
    ref = ref.astype(bool, copy=False)
    if not pred.any() and not ref.any():
        return 0.0
    if not pred.any() or not ref.any():
        return float("nan")
    pred_surface = surface(pred)
    ref_surface = surface(ref)
    dt_ref = distance_transform_edt(~ref_surface, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([dt_ref[pred_surface], dt_pred[ref_surface]])
    return float(np.percentile(distances, 95)) if distances.size else 0.0


def region_masks(seg: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "ET": seg == 3,
        "TC": np.logical_or(seg == 1, seg == 3),
        "WT": seg > 0,
    }


def selected_validation_cases(splits_path: Path, fold_arg: str) -> dict[str, int]:
    with splits_path.open() as f:
        splits = json.load(f)
    if fold_arg.lower() == "all":
        folds = range(len(splits))
    else:
        folds = [int(fold_arg)]
    cases: dict[str, int] = {}
    for fold in folds:
        for case_id in splits[fold]["val"]:
            if case_id in cases:
                raise RuntimeError(f"Case appears in multiple selected validation folds: {case_id}")
            cases[case_id] = fold
    return cases


def main() -> None:
    args = parse_args()
    val_cases = selected_validation_cases(args.splits_json, args.fold)
    gt_cases = {p.name[:-7] for p in args.dataset005_labels.glob("*.nii.gz")}
    pseudo_in_val = sorted(set(val_cases) - gt_cases)
    if pseudo_in_val:
        raise RuntimeError(
            "Validation split contains cases without Dataset005 GT labels. "
            f"This would violate original-GT-only evaluation. Examples: {pseudo_in_val[:5]}"
        )

    rows = []
    for case_id in sorted(val_cases):
        pred_path = args.prediction_dir / f"{case_id}.nii.gz"
        ref_path = args.dataset005_labels / f"{case_id}.nii.gz"
        if not pred_path.is_file():
            raise FileNotFoundError(f"Missing prediction for validation case: {pred_path}")
        pred, spacing = read_seg(pred_path)
        ref, ref_spacing = read_seg(ref_path)
        if pred.shape != ref.shape:
            raise RuntimeError(f"Shape mismatch for {case_id}: pred={pred.shape}, ref={ref.shape}")
        pred_masks = region_masks(pred)
        ref_masks = region_masks(ref)
        row = {
            "case_id": case_id,
            "fold": val_cases[case_id],
            "prediction_file": str(pred_path),
            "reference_file": str(ref_path),
        }
        for region in ("ET", "TC", "WT"):
            row[f"Dice_{region}"] = dice(pred_masks[region], ref_masks[region])
            if args.compute_hd95:
                row[f"HD95_{region}"] = hd95(pred_masks[region], ref_masks[region], ref_spacing or spacing)
        row["mean_dice"] = float(np.mean([row["Dice_ET"], row["Dice_TC"], row["Dice_WT"]]))
        rows.append(row)

    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    summary = {
        "prediction_dir": str(args.prediction_dir),
        "n_cases": int(len(df)),
        "fold": args.fold,
    }
    for col in [c for c in df.columns if c.startswith("Dice_") or c.startswith("HD95_") or c == "mean_dice"]:
        summary[f"{col}_mean"] = float(pd.to_numeric(df[col], errors="coerce").mean())
        summary[f"{col}_median"] = float(pd.to_numeric(df[col], errors="coerce").median())
    summary_path = args.summary_json or args.output_csv.with_suffix(".summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {args.output_csv} rows={len(df)}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
