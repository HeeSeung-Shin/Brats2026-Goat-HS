#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mlconsensus_common import (
    DATASET005_SPLITS,
    DATASET007_PREPROCESSED,
    DATASET007_ROOT,
    ensure_dir,
    load_json,
    read_csv_rows,
    utc_timestamp,
    write_csv,
    write_json,
)


DEFAULT_SPLIT_DIR = Path(__file__).resolve().parents[2] / "private_assets" / "splits_dataset007"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Dataset007 splits with Dataset005 GT-only validation folds.")
    parser.add_argument("--dataset007-root", default=str(DATASET007_ROOT))
    parser.add_argument("--dataset005-splits", default=str(DATASET005_SPLITS))
    parser.add_argument("--output-json", default=str(DEFAULT_SPLIT_DIR / "splits_final.json"))
    parser.add_argument("--out-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--copy-to-preprocessed", action="store_true")
    parser.add_argument("--preprocessed-dataset007", default=str(DATASET007_PREPROCESSED))
    parser.add_argument("--preview-only", action="store_true")
    return parser.parse_args()


def write_validation_report(path: Path, rows: list[dict[str, Any]], split_source: str) -> None:
    lines = [
        "Dataset007 GT-only validation split report",
        "=" * 48,
        f"timestamp: {utc_timestamp()}",
        f"source: {split_source}",
        "",
    ]
    for row in rows:
        lines.append(
            "fold {fold}: train={train_total} "
            "(original={train_original_labeled}, pseudo={train_pseudo_labeled_ml}), "
            "val={val_total} original-only, overlap={train_val_overlap}".format(**row)
        )
    lines.extend(
        [
            "",
            "Assertions enforced:",
            "- validation contains Dataset005 original GT cases only",
            "- pseudo labeled cases are in train for every fold",
            "- pseudo labeled cases are never in validation",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    dataset007_root = Path(args.dataset007_root)
    manifest_path = dataset007_root / "dataset007_case_manifest.csv"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing Dataset007 manifest: {manifest_path}")
    rows = read_csv_rows(manifest_path)
    original_cases = sorted(row["case_id"] for row in rows if row.get("role") == "original_labeled")
    pseudo_cases = sorted(
        row["case_id"]
        for row in rows
        if row.get("role") == "pseudo_labeled_qc_strict"
        and row.get("selected") in {"True", "true", "1", "yes", True}
    )
    original_set = set(original_cases)
    pseudo_set = set(pseudo_cases)
    if not original_cases:
        raise RuntimeError("No original_labeled cases in Dataset007 manifest")
    if not pseudo_cases:
        raise RuntimeError("No selected pseudo-labeled cases in non-labeled-only manifest")

    dataset005_splits_path = Path(args.dataset005_splits)
    dataset005_splits = load_json(dataset005_splits_path)
    splits: list[dict[str, list[str]]] = []
    summary_rows: list[dict[str, Any]] = []
    for fold_idx, split in enumerate(dataset005_splits):
        val_original = sorted(c for c in split["val"] if c in original_set)
        train_original = sorted(c for c in split["train"] if c in original_set)
        if set(val_original) != set(split["val"]):
            missing = sorted(set(split["val"]) - set(val_original))
            raise RuntimeError(f"Fold {fold_idx}: Dataset005 validation cases missing from Dataset007 originals: {missing[:5]}")
        train = sorted(set(train_original) | pseudo_set)
        val = val_original
        train_set = set(train)
        val_set = set(val)
        if train_set & val_set:
            raise RuntimeError(f"Fold {fold_idx}: train/val overlap: {sorted(train_set & val_set)[:5]}")
        if not val_set <= original_set:
            raise RuntimeError(f"Fold {fold_idx}: validation contains non-original cases")
        if pseudo_set & val_set:
            raise RuntimeError(f"Fold {fold_idx}: pseudo case entered validation")
        if pseudo_set - train_set:
            raise RuntimeError(f"Fold {fold_idx}: pseudo cases missing from train")
        splits.append({"train": train, "val": val})
        summary_rows.append(
            {
                "fold": fold_idx,
                "train_total": len(train_set),
                "train_original_labeled": len(train_set & original_set),
                "train_pseudo_labeled_ml": len(train_set & pseudo_set),
                "val_total": len(val_set),
                "val_original_labeled": len(val_set & original_set),
                "val_pseudo_labeled_ml": len(val_set & pseudo_set),
                "train_val_overlap": len(train_set & val_set),
            }
        )

    out_dir = ensure_dir(args.out_dir)
    output_json = Path(args.output_json)
    ensure_dir(output_json.parent)
    write_json(output_json, splits)
    write_csv(out_dir / "splits_dataset007_summary.csv", summary_rows)
    write_json(
        out_dir / "splits_dataset007_summary.json",
        {
            "timestamp": utc_timestamp(),
            "student_architecture": rows[0].get("student_architecture", ""),
            "softmoe_k": rows[0].get("softmoe_k", ""),
            "auxiliary_head": rows[0].get("auxiliary_head", ""),
            "pseudo_source": rows[0].get("pseudo_source", ""),
            "qc_status": rows[0].get("qc_status", ""),
            "dataset007_root": str(dataset007_root),
            "dataset005_splits": str(dataset005_splits_path),
            "output_json": str(output_json),
            "counts": {
                "original_labeled": len(original_cases),
                "pseudo_labeled_ml": len(pseudo_cases),
            },
            "folds": summary_rows,
        },
    )
    write_validation_report(out_dir / "split_validation_report.txt", summary_rows, str(dataset005_splits_path))

    copied_to = None
    if args.copy_to_preprocessed and not args.preview_only:
        preprocessed = Path(args.preprocessed_dataset007)
        if not preprocessed.is_dir():
            raise RuntimeError(f"Cannot copy splits because preprocessed folder does not exist: {preprocessed}")
        copied_to = preprocessed / "splits_final.json"
        write_json(copied_to, splits)

    print(f"Wrote split json: {output_json}")
    print(f"Wrote summary CSV: {out_dir / 'splits_dataset007_summary.csv'}")
    print(f"Wrote validation report: {out_dir / 'split_validation_report.txt'}")
    if copied_to:
        print(f"Copied splits_final.json to: {copied_to}")
    for row in summary_rows:
        print(
            f"fold {row['fold']}: train={row['train_total']} "
            f"(original={row['train_original_labeled']}, pseudo={row['train_pseudo_labeled_ml']}), "
            f"val={row['val_total']} original-only"
        )


if __name__ == "__main__":
    main()
