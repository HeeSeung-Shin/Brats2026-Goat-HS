#!/usr/bin/env python3
"""Verify private assets required to reproduce the audited Dataset007 run.

The default checks are structural and fast. Pass ``--hash`` to also compute the
audited SHA-256 values, including the five approximately 779 MiB D005
initialization checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DATASET_NAME = "Dataset007_Brats26_Goat_MLConsensusPseudo"
D005_EXPERIMENT = "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"

EXPECTED_DATASET_JSON = {
    "channel_names": {"0": "t1n", "1": "t1c", "2": "t2w", "3": "t2f"},
    "file_ending": ".nii.gz",
    "labels": {"background": 0, "NCR": 1, "ED": 2, "ET": 3},
    "name": DATASET_NAME,
    "numTraining": 2334,
}
EXPECTED_ROLE_COUNTS = {
    "original_labeled": 1351,
    "pseudo_labeled_ml_strict": 983,
    "excluded": 154,
    "ml_unlabeled_remainder": 1,
}
EXPECTED_FOLDS = {
    0: {"train": 2063, "train_gt": 1080, "train_pseudo": 983, "val": 271},
    1: {"train": 2064, "train_gt": 1081, "train_pseudo": 983, "val": 270},
    2: {"train": 2064, "train_gt": 1081, "train_pseudo": 983, "val": 270},
    3: {"train": 2064, "train_gt": 1081, "train_pseudo": 983, "val": 270},
    4: {"train": 2064, "train_gt": 1081, "train_pseudo": 983, "val": 270},
}
EXPECTED_HASHES = {
    "dataset.json": "089ca9b7e4f06eaaad40b7c95ebe64ba713e1265383cf9e44a3921e6664e9ce7",
    "dataset007_case_manifest.csv": "76feef1b15e1bd8eac31443e84875ab13329b9d77b30fd139f50f8738ca4f665",
    "splits_final.json": "fba61c6486e48a3893b5ce9147271e44e8aacf48718db86f552fbacdfea6957e",
    "case_weights_fold0.json": "d594e55e7073a8b0d347e216d7cdd43dbdd17f0a3de019ef70bc74c63688c249",
    "case_weights_fold1.json": "cf49e9681dd6b16a494945f6890ef56e3951e35ced7fcfe94d2783ec584ae29f",
    "case_weights_fold2.json": "2cf4f8e788e24ab965638b2dac4c7d170d8d425b2c84219662cf5d7905594bff",
    "case_weights_fold3.json": "7e0f574e030df161d62bb4a85d1b8c3d68b0818aa80d842c3ce28c56748fec41",
    "case_weights_fold4.json": "3d15d6c39a7e300c4bb61a1d766823f2246000e3dcf5c633e67f7909f764eef6",
    "d007_case_region_cluster_meta.csv": "852d03abb62c6251e2a414bf106d36df542ebf4ace27cdf2ade5341dad63870c",
}
EXPECTED_D005 = {
    0: (816315151, "02ef1d4310e61e90a90cbf452ab9ff4a5ce62bd090759da6d1b03dbf1d5f2ad7"),
    1: (816346639, "42cf485aadad58b7d7b47d5715b2d2a282da53534ad3ff619dfc7dc88e961a39"),
    2: (816293135, "a2a3992ad3de73684913713c2deff418fab33008b8248074a41e62b91187a1fb"),
    3: (816224911, "24372da3da6f65c4972b41158c0b954edc49ce45eb76bd608e94410b840ed9fc"),
    4: (816333647, "d28917c3cbd4220f59a73eb0e48258a4e5c827723e6bff7aac3c4712050f7fe1"),
}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.passes: list[str] = []

    def check(self, condition: bool, message: str) -> bool:
        if condition:
            self.passes.append(message)
            return True
        self.errors.append(message)
        return False

    def error(self, message: str) -> None:
        self.errors.append(message)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Verify Dataset007, five folds, case weights, and D005 initialization assets."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--config", type=Path, default=repo_root / "config" / "private_assets.json")
    parser.add_argument("--dataset007-raw", "--dataset-root", dest="dataset007_raw", type=Path)
    parser.add_argument("--dataset007-preprocessed", "--preprocessed-root", dest="dataset007_preprocessed", type=Path)
    parser.add_argument("--case-weights-dir", type=Path)
    parser.add_argument(
        "--d005-pretrained-root",
        "--d005-root",
        dest="d005_pretrained_root",
        type=Path,
        help="Exact D005 ResEnc-M experiment directory, or its Dataset005 results parent.",
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Compute and compare audited SHA-256 values (slow for five checkpoints).",
    )
    return parser.parse_args()


def load_json(path: Path, report: Report) -> Any | None:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        report.error(f"cannot read JSON {path}: {exc}")
        return None


def load_config(path: Path, report: Report) -> dict[str, Any]:
    value = load_json(path, report)
    if not isinstance(value, dict):
        report.error(f"asset config must be a JSON object: {path}")
        return {}
    return value


def resolve_path(
    cli_value: Path | None,
    env_name: str,
    config_value: Any,
    repo_root: Path,
    fallback: Path | None = None,
) -> Path | None:
    raw: str | Path | None = cli_value or os.environ.get(env_name) or config_value or fallback
    if raw is None or str(raw).strip() == "":
        return None
    path = Path(os.path.expandvars(str(raw))).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve(strict=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def check_hash(path: Path, expected: str, report: Report) -> None:
    print(f"HASH {path}", flush=True)
    actual = sha256(path)
    report.check(actual == expected, f"SHA-256 {path.name}: expected {expected}, got {actual}")


def nifti_image_cases(directory: Path, report: Report) -> dict[str, set[str]]:
    cases: dict[str, set[str]] = defaultdict(set)
    for path in directory.glob("*.nii.gz"):
        base = path.name[:-7]
        case_id, separator, channel = base.rpartition("_")
        if not separator or len(channel) != 4 or not channel.isdigit():
            report.error(f"unexpected nnU-Net image filename: {path}")
            continue
        cases[case_id].add(channel)
    return dict(cases)


def nifti_label_cases(directory: Path) -> set[str]:
    return {path.name[:-7] for path in directory.glob("*.nii.gz")}


def selected_value(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def verify_dataset(dataset_root: Path | None, hash_assets: bool, report: Report) -> dict[str, set[str]] | None:
    if dataset_root is None:
        report.error("Dataset007 raw path is unset; use --dataset007-raw or BRATS_D007_RAW")
        return None
    required = {
        "dataset_json": dataset_root / "dataset.json",
        "manifest": dataset_root / "dataset007_case_manifest.csv",
        "imagesTr": dataset_root / "imagesTr",
        "labelsTr": dataset_root / "labelsTr",
        "imagesUn": dataset_root / "imagesUn",
    }
    for key, path in required.items():
        report.check(path.is_file() if key in {"dataset_json", "manifest"} else path.is_dir(), f"exists: {path}")
    if not all(
        path.is_file() if key in {"dataset_json", "manifest"} else path.is_dir()
        for key, path in required.items()
    ):
        return None

    dataset_json = load_json(required["dataset_json"], report)
    if isinstance(dataset_json, dict):
        for key, expected in EXPECTED_DATASET_JSON.items():
            report.check(dataset_json.get(key) == expected, f"dataset.json {key} == {expected!r}")

    train_images = nifti_image_cases(required["imagesTr"], report)
    unlabeled_images = nifti_image_cases(required["imagesUn"], report)
    label_cases = nifti_label_cases(required["labelsTr"])
    channels = {"0000", "0001", "0002", "0003"}
    bad_train = sorted(case for case, found in train_images.items() if found != channels)
    bad_unlabeled = sorted(case for case, found in unlabeled_images.items() if found != channels)
    report.check(len(train_images) == 2334, f"imagesTr contains 2334 cases (found {len(train_images)})")
    report.check(len(label_cases) == 2334, f"labelsTr contains 2334 cases (found {len(label_cases)})")
    report.check(len(unlabeled_images) == 155, f"imagesUn contains 155 cases (found {len(unlabeled_images)})")
    report.check(not bad_train, f"all imagesTr cases have channels 0000..0003; bad={bad_train[:5]}")
    report.check(not bad_unlabeled, f"all imagesUn cases have channels 0000..0003; bad={bad_unlabeled[:5]}")
    report.check(set(train_images) == label_cases, "imagesTr and labelsTr case IDs match exactly")

    rows: list[dict[str, str]] = []
    try:
        with required["manifest"].open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        report.error(f"cannot read manifest {required['manifest']}: {exc}")
        return None
    required_columns = {"case_id", "role", "selected"}
    report.check(bool(rows) and required_columns <= set(rows[0]), f"manifest has columns {sorted(required_columns)}")
    if not rows or not required_columns <= set(rows[0]):
        return None

    role_counts = Counter(row["role"] for row in rows)
    all_ids = [row["case_id"] for row in rows]
    selected_rows = [row for row in rows if selected_value(row["selected"])]
    unselected_rows = [row for row in rows if not selected_value(row["selected"])]
    selected_ids = {row["case_id"] for row in selected_rows}
    unselected_ids = {row["case_id"] for row in unselected_rows}
    gt_ids = {row["case_id"] for row in selected_rows if row["role"] == "original_labeled"}
    pseudo_ids = {
        row["case_id"] for row in selected_rows if row["role"] == "pseudo_labeled_ml_strict"
    }
    report.check(len(rows) == 2489, f"manifest contains 2489 rows (found {len(rows)})")
    report.check(len(set(all_ids)) == len(all_ids), "manifest case_id values are unique")
    report.check(dict(role_counts) == EXPECTED_ROLE_COUNTS, f"manifest roles equal {EXPECTED_ROLE_COUNTS}; found {dict(role_counts)}")
    report.check(len(selected_ids) == 2334, f"manifest has 2334 selected cases (found {len(selected_ids)})")
    report.check(len(unselected_ids) == 155, f"manifest has 155 unselected cases (found {len(unselected_ids)})")
    report.check(selected_ids == set(train_images), "selected manifest IDs equal imagesTr IDs")
    report.check(unselected_ids == set(unlabeled_images), "unselected manifest IDs equal imagesUn IDs")
    report.check(len(gt_ids) == 1351, f"selected original GT count is 1351 (found {len(gt_ids)})")
    report.check(len(pseudo_ids) == 983, f"selected strict pseudo count is 983 (found {len(pseudo_ids)})")
    report.check(gt_ids.isdisjoint(pseudo_ids), "GT and pseudo case sets are disjoint")

    if hash_assets:
        check_hash(required["dataset_json"], EXPECTED_HASHES["dataset.json"], report)
        check_hash(required["manifest"], EXPECTED_HASHES["dataset007_case_manifest.csv"], report)
    return {"selected": selected_ids, "gt": gt_ids, "pseudo": pseudo_ids}


def verify_splits(
    preprocessed_root: Path | None,
    cases: dict[str, set[str]] | None,
    hash_assets: bool,
    report: Report,
) -> list[dict[str, list[str]]] | None:
    if preprocessed_root is None:
        report.error("Dataset007 preprocessed path is unset; use --dataset007-preprocessed or BRATS_D007_PREPROCESSED")
        return None
    path = preprocessed_root / "splits_final.json"
    if not report.check(path.is_file(), f"exists: {path}"):
        return None
    splits = load_json(path, report)
    if not isinstance(splits, list):
        report.error(f"splits_final.json must contain a list: {path}")
        return None
    report.check(len(splits) == 5, f"splits_final.json has 5 folds (found {len(splits)})")
    if len(splits) != 5 or cases is None:
        return None
    for fold, split in enumerate(splits):
        if not isinstance(split, dict) or not isinstance(split.get("train"), list) or not isinstance(split.get("val"), list):
            report.error(f"fold {fold} must contain list-valued train and val")
            continue
        train_list = split["train"]
        val_list = split["val"]
        train = set(train_list)
        val = set(val_list)
        expected = EXPECTED_FOLDS[fold]
        report.check(len(train_list) == len(train), f"fold {fold} train IDs are unique")
        report.check(len(val_list) == len(val), f"fold {fold} val IDs are unique")
        report.check(not train & val, f"fold {fold} train/val do not overlap")
        report.check(train | val == cases["selected"], f"fold {fold} train+val equals all 2334 supervised cases")
        report.check(val <= cases["gt"], f"fold {fold} validation is original-GT only")
        report.check(cases["pseudo"] <= train, f"fold {fold} contains every pseudo case in train")
        report.check(len(train) == expected["train"], f"fold {fold} train count is {expected['train']} (found {len(train)})")
        report.check(len(train & cases["gt"]) == expected["train_gt"], f"fold {fold} GT-train count is {expected['train_gt']}")
        report.check(len(train & cases["pseudo"]) == expected["train_pseudo"], f"fold {fold} pseudo-train count is 983")
        report.check(len(val) == expected["val"], f"fold {fold} validation count is {expected['val']} (found {len(val)})")
    if hash_assets:
        check_hash(path, EXPECTED_HASHES["splits_final.json"], report)
    return splits


def verify_case_weights(
    directory: Path | None,
    splits: list[dict[str, list[str]]] | None,
    hash_assets: bool,
    report: Report,
) -> None:
    if directory is None:
        report.error("case-weight path is unset; use --case-weights-dir or BRATS_CASE_WEIGHTS_DIR")
        return
    if not report.check(directory.is_dir(), f"exists: {directory}"):
        return
    case_meta = directory / "d007_case_region_cluster_meta.csv"
    if report.check(case_meta.is_file(), f"exists: {case_meta}") and hash_assets:
        check_hash(case_meta, EXPECTED_HASHES[case_meta.name], report)
    if splits is None:
        report.error("cannot cross-check case weights because valid splits were not available")
        return
    for fold in range(5):
        path = directory / f"case_weights_fold{fold}.json"
        if not report.check(path.is_file(), f"exists: {path}"):
            continue
        payload = load_json(path, report)
        if not isinstance(payload, dict):
            continue
        weights = payload.get("case_weights")
        details = payload.get("case_details")
        if not isinstance(weights, dict):
            report.error(f"fold {fold} case_weights must be an object")
            continue
        train = set(splits[fold]["train"])
        val = set(splits[fold]["val"])
        weight_ids = set(weights)
        report.check(payload.get("fold") == fold, f"case-weight payload fold equals {fold}")
        report.check(payload.get("threshold_source") == "GT train cases only", f"fold {fold} thresholds use GT train only")
        report.check(weight_ids == train, f"fold {fold} weights exactly cover its train cases")
        report.check(weight_ids.isdisjoint(val), f"fold {fold} weights exclude validation cases")
        report.check(len(weights) == EXPECTED_FOLDS[fold]["train"], f"fold {fold} weight count is {EXPECTED_FOLDS[fold]['train']}")
        invalid_values = []
        for case_id, value in weights.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                invalid_values.append(case_id)
                continue
            if not math.isfinite(number) or not 0.5 <= number <= 2.0:
                invalid_values.append(case_id)
        report.check(not invalid_values, f"fold {fold} weights are finite and within [0.5, 2.0]; bad={invalid_values[:5]}")
        if isinstance(details, dict):
            report.check(set(details) == train, f"fold {fold} case_details exactly cover its train cases")
        else:
            report.error(f"fold {fold} case_details must be an object")
        if hash_assets:
            check_hash(path, EXPECTED_HASHES[path.name], report)


def normalize_d005_root(path: Path | None) -> Path | None:
    if path is None:
        return None
    if (path / "fold_0" / "checkpoint_best.pth").is_file():
        return path
    child = path / D005_EXPERIMENT
    if child.is_dir():
        return child
    return path


def verify_d005(root: Path | None, hash_assets: bool, report: Report) -> None:
    root = normalize_d005_root(root)
    if root is None:
        report.error("D005 pretrained path is unset; use --d005-pretrained-root or BRATS_D005_PRETRAINED_ROOT")
        return
    if not report.check(root.is_dir(), f"exists: {root}"):
        return
    for fold, (expected_size, expected_hash) in EXPECTED_D005.items():
        path = root / f"fold_{fold}" / "checkpoint_best.pth"
        if not report.check(path.is_file(), f"exists: D005 fold {fold} checkpoint_best.pth"):
            continue
        actual_size = path.stat().st_size
        report.check(actual_size == expected_size, f"D005 fold {fold} size is {expected_size} bytes (found {actual_size})")
        if hash_assets:
            check_hash(path, expected_hash, report)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve(strict=False)
    report = Report()
    config = load_config(args.config.resolve(strict=False), report)
    paths = config.get("paths", {}) if isinstance(config.get("paths", {}), dict) else {}

    dataset_raw_fallback = None
    if os.environ.get("nnUNet_raw"):
        dataset_raw_fallback = Path(os.environ["nnUNet_raw"]) / DATASET_NAME
    preprocessed_fallback = None
    if os.environ.get("nnUNet_preprocessed"):
        preprocessed_fallback = Path(os.environ["nnUNet_preprocessed"]) / DATASET_NAME
    d005_fallback = None
    if os.environ.get("nnUNet_results"):
        d005_fallback = (
            Path(os.environ["nnUNet_results"])
            / "Dataset005_Brats26_Goat_With_GroundTruth"
            / D005_EXPERIMENT
        )

    dataset_root = resolve_path(
        args.dataset007_raw, "BRATS_D007_RAW", paths.get("dataset007_raw"), repo_root, dataset_raw_fallback
    )
    case_weights_override = args.case_weights_dir
    if case_weights_override is None and os.environ.get("D007_ETAWARE_CASE_WEIGHT_ROOT"):
        case_weights_override = Path(os.environ["D007_ETAWARE_CASE_WEIGHT_ROOT"])
    d005_override = args.d005_pretrained_root
    if d005_override is None and os.environ.get("D005_PRETRAINED_ROOT"):
        d005_override = Path(os.environ["D005_PRETRAINED_ROOT"])
    preprocessed_root = resolve_path(
        args.dataset007_preprocessed,
        "BRATS_D007_PREPROCESSED",
        paths.get("dataset007_preprocessed"),
        repo_root,
        preprocessed_fallback,
    )
    case_weights_dir = resolve_path(
        case_weights_override,
        "BRATS_CASE_WEIGHTS_DIR",
        paths.get("case_weights_dir"),
        repo_root,
        repo_root / "private_assets" / "case_weights",
    )
    d005_root = resolve_path(
        d005_override,
        "BRATS_D005_PRETRAINED_ROOT",
        paths.get("d005_pretrained_root"),
        repo_root,
        d005_fallback or repo_root / "private_assets" / "pretrained_d005",
    )

    print("Resolved private assets:")
    print(f"  Dataset007 raw:          {dataset_root}")
    print(f"  Dataset007 preprocessed: {preprocessed_root}")
    print(f"  case weights:            {case_weights_dir}")
    print(f"  D005 initialization:     {d005_root}")
    print(f"  SHA-256 mode:            {args.hash}")

    cases = verify_dataset(dataset_root, args.hash, report)
    splits = verify_splits(preprocessed_root, cases, args.hash, report)
    verify_case_weights(case_weights_dir, splits, args.hash, report)
    verify_d005(d005_root, args.hash, report)

    if report.errors:
        print(f"\nFAIL: {len(report.errors)} check(s) failed; {len(report.passes)} passed.", file=sys.stderr)
        for message in report.errors:
            print(f"  - {message}", file=sys.stderr)
        return 1
    print(f"\nPASS: all {len(report.passes)} private-asset checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
