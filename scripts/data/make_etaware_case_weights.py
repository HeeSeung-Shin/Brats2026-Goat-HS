#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRIVATE_ROOT = Path(__file__).resolve().parents[2] / "private_assets"
DEFAULT_META = PRIVATE_ROOT / "case_weights" / "d007_case_region_cluster_meta.csv"
DEFAULT_SPLITS = PRIVATE_ROOT / "splits_final.json"
DEFAULT_OUTPUT_DIR = PRIVATE_ROOT / "case_weights"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make fold-specific ET/TC-aware mild case sampling weights.")
    parser.add_argument("--meta-csv", type=Path, default=DEFAULT_META)
    parser.add_argument("--splits-json", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--worst-clusters", nargs="+", type=int, default=[4, 1, 5])
    return parser.parse_args()


def bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"true", "1", "yes"})


def q30(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.quantile(0.30))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.meta_csv)
    with args.splits_json.open() as f:
        splits = json.load(f)

    meta["is_gt_bool"] = bool_series(meta["is_gt"])
    meta["is_pseudo_bool"] = bool_series(meta["is_pseudo"])
    meta["ET_present_bool"] = bool_series(meta["ET_present"])
    meta["TC_present_bool"] = bool_series(meta["TC_present"])
    if "majority_cluster" in meta:
        meta["majority_cluster_int"] = pd.to_numeric(meta["majority_cluster"], errors="coerce").astype("Int64")
    else:
        meta["majority_cluster_int"] = pd.Series([pd.NA] * len(meta), dtype="Int64")

    summary_rows = []
    all_weight_rows = []
    meta_by_case = {row.case_id: row for row in meta.itertuples(index=False)}

    for fold, split in enumerate(splits):
        train_cases = set(split["train"])
        val_cases = set(split["val"])
        train_df = meta[meta["case_id"].isin(train_cases)].copy()
        gt_train = train_df[train_df["is_gt_bool"]].copy()
        if gt_train.empty:
            raise RuntimeError(f"Fold {fold}: no GT train cases for threshold computation")

        et_pos_gt = gt_train[gt_train["ET_present_bool"]]
        tc_pos_gt = gt_train[gt_train["TC_present_bool"]]
        small_et_thr = q30(et_pos_gt["ET_volume_mm3"])
        small_tc_thr = q30(tc_pos_gt["TC_volume_mm3"])
        low_fg_thr = q30(gt_train["foreground_ratio"])

        case_weights: dict[str, float] = {}
        case_details: dict[str, dict] = {}
        for row in train_df.itertuples(index=False):
            weight = 1.0
            is_gt = bool(row.is_gt_bool)
            is_pseudo = bool(row.is_pseudo_bool)
            et_present = bool(row.ET_present_bool)
            small_et = bool(et_present and np.isfinite(small_et_thr) and row.ET_volume_mm3 <= small_et_thr)
            small_tc = bool(row.TC_present_bool and np.isfinite(small_tc_thr) and row.TC_volume_mm3 <= small_tc_thr)
            low_fg = bool(np.isfinite(low_fg_thr) and row.foreground_ratio <= low_fg_thr)
            cluster_boost = False
            if not pd.isna(row.majority_cluster_int):
                cluster_boost = int(row.majority_cluster_int) in set(args.worst_clusters)

            if is_gt:
                weight *= 1.10
            if et_present:
                weight *= 1.15
            if small_et:
                weight *= 1.30
            if small_tc:
                weight *= 1.15
            if low_fg:
                weight *= 1.15
            if cluster_boost:
                weight *= 1.10
            if is_pseudo:
                weight = min(weight, 1.4)
            if is_gt:
                weight = min(weight, 2.0)
            weight = max(weight, 0.5)

            case_weights[row.case_id] = float(weight)
            detail = {
                "case_id": row.case_id,
                "fold": fold,
                "weight": float(weight),
                "is_gt": is_gt,
                "is_pseudo": is_pseudo,
                "ET_present": et_present,
                "small_ET_fold_threshold": small_et,
                "small_TC_fold_threshold": small_tc,
                "low_foreground_ratio_fold_threshold": low_fg,
                "proxy_cluster_boost": cluster_boost,
                "majority_cluster": "" if pd.isna(row.majority_cluster_int) else int(row.majority_cluster_int),
            }
            case_details[row.case_id] = detail
            all_weight_rows.append(detail)

        leakage = sorted(train_cases & val_cases)
        if leakage:
            raise RuntimeError(f"Fold {fold}: train/val overlap detected: {leakage[:5]}")
        if any(c in case_weights for c in val_cases):
            raise RuntimeError(f"Fold {fold}: validation case has training weight")

        out = {
            "fold": fold,
            "threshold_source": "GT train cases only",
            "small_ET_volume_mm3_q30": small_et_thr,
            "small_TC_volume_mm3_q30": small_tc_thr,
            "low_foreground_ratio_q30": low_fg_thr,
            "worst_clusters": args.worst_clusters,
            "case_weights": case_weights,
            "case_details": case_details,
        }
        out_path = args.output_dir / f"case_weights_fold{fold}.json"
        with out_path.open("w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
            f.write("\n")
        weights = np.asarray(list(case_weights.values()), dtype=float)
        summary_rows.append(
            {
                "fold": fold,
                "n_train": len(case_weights),
                "n_gt_train": int(train_df["is_gt_bool"].sum()),
                "n_pseudo_train": int(train_df["is_pseudo_bool"].sum()),
                "small_ET_volume_mm3_q30": small_et_thr,
                "small_TC_volume_mm3_q30": small_tc_thr,
                "low_foreground_ratio_q30": low_fg_thr,
                "weight_min": float(weights.min()),
                "weight_mean": float(weights.mean()),
                "weight_max": float(weights.max()),
                "n_weight_gt_1": int((weights > 1.0).sum()),
            }
        )
        print(f"Wrote {out_path}")

    pd.DataFrame(summary_rows).to_csv(args.output_dir / "case_weights_summary.csv", index=False)
    pd.DataFrame(all_weight_rows).to_csv(args.output_dir / "case_weights_all_folds_long.csv", index=False)
    with (args.output_dir / "case_weights_summary.json").open("w") as f:
        json.dump({"folds": summary_rows}, f, indent=2)
        f.write("\n")
    print(f"Wrote {args.output_dir / 'case_weights_summary.csv'}")
    print(f"Wrote {args.output_dir / 'case_weights_all_folds_long.csv'}")


if __name__ == "__main__":
    main()
