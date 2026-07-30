#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import label as cc_label


PRIVATE_ROOT = Path(__file__).resolve().parents[2] / "private_assets"
DEFAULT_D007_RAW = PRIVATE_ROOT / "nnUNet_data" / "nnUNet_raw" / "Dataset007_Brats26_Goat_MLConsensusPseudo"
DEFAULT_D005_LABELS = PRIVATE_ROOT / "nnUNet_data" / "nnUNet_raw" / "Dataset005_Brats26_Goat_With_GroundTruth" / "labelsTr"
DEFAULT_CLUSTER_CSV = PRIVATE_ROOT / "case_weights" / "latent_clusters_with_fold_agreement.csv"
DEFAULT_OUTPUT = PRIVATE_ROOT / "case_weights" / "d007_case_region_cluster_meta.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build D007 ET/TC/WT morphology + proxy cluster case metadata.")
    parser.add_argument("--dataset007-raw", type=Path, default=DEFAULT_D007_RAW)
    parser.add_argument("--dataset005-labels", type=Path, default=DEFAULT_D005_LABELS)
    parser.add_argument("--cluster-csv", type=Path, default=DEFAULT_CLUSTER_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-components",
        action="store_true",
        help="Skip ET/TC/WT connected-component counts. By default these are computed for the region auxiliary-head analysis.",
    )
    return parser.parse_args()


def read_seg(path: Path) -> tuple[np.ndarray, float]:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.uint8, copy=False)
    spacing = img.GetSpacing()
    voxel_volume = float(spacing[0] * spacing[1] * spacing[2])
    return arr, voxel_volume


def component_stats(mask: np.ndarray) -> tuple[int, float]:
    if not mask.any():
        return 0, 0.0
    struct = np.ones((3, 3, 3), dtype=np.uint8)
    cc, n = cc_label(mask, structure=struct)
    if n == 0:
        return 0, 0.0
    sizes = np.bincount(cc.ravel())[1:]
    return int(n), float(sizes.max() / max(1, mask.sum()))


def main() -> None:
    args = parse_args()
    labels_dir = args.dataset007_raw / "labelsTr"
    d005_cases = {p.name[:-7] for p in args.dataset005_labels.glob("*.nii.gz")}
    clusters = pd.read_csv(args.cluster_csv) if args.cluster_csv.is_file() else pd.DataFrame()
    cluster_cols = [
        c
        for c in [
            "case_id",
            "majority_cluster",
            "fold_cluster_agreement",
            "low_fold_agreement",
            "fold_cluster_histogram",
            "role_group",
        ]
        if c in clusters.columns
    ]
    clusters = clusters[cluster_cols].drop_duplicates("case_id") if len(cluster_cols) else pd.DataFrame()

    rows = []
    for idx, label_path in enumerate(sorted(labels_dir.glob("*.nii.gz")), start=1):
        case_id = label_path.name[:-7]
        seg, voxel_volume = read_seg(label_path)
        et = seg == 3
        tc = (seg == 1) | (seg == 3)
        wt = seg > 0
        if not args.skip_components:
            et_cc, et_largest = component_stats(et)
            tc_cc, tc_largest = component_stats(tc)
            wt_cc, wt_largest = component_stats(wt)
        else:
            et_cc, et_largest = 0, float("nan")
            tc_cc, tc_largest = 0, float("nan")
            wt_cc, wt_largest = 0, float("nan")
        spatial_voxels = int(np.prod(seg.shape))
        row = {
            "case_id": case_id,
            "label_file": str(label_path),
            "is_gt": case_id in d005_cases,
            "is_pseudo": case_id not in d005_cases,
            "ET_voxels": int(et.sum()),
            "TC_voxels": int(tc.sum()),
            "WT_voxels": int(wt.sum()),
            "ET_volume_mm3": float(et.sum() * voxel_volume),
            "TC_volume_mm3": float(tc.sum() * voxel_volume),
            "WT_volume_mm3": float(wt.sum() * voxel_volume),
            "ET_present": bool(et.any()),
            "TC_present": bool(tc.any()),
            "WT_present": bool(wt.any()),
            "foreground_ratio": float(wt.sum() / max(1, spatial_voxels)),
            "ET_component_count": et_cc,
            "TC_component_count": tc_cc,
            "WT_component_count": wt_cc,
            "ET_largest_component_ratio": et_largest,
            "TC_largest_component_ratio": tc_largest,
            "WT_largest_component_ratio": wt_largest,
            "cluster_note": "latent_embedding_proxy_subgroup_not_domain_label",
        }
        rows.append(row)
        if idx % 250 == 0:
            print(f"{idx} labels processed", flush=True)

    df = pd.DataFrame(rows)
    if not clusters.empty:
        df = df.merge(clusters, on="case_id", how="left")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv} rows={len(df)}", flush=True)


if __name__ == "__main__":
    main()
