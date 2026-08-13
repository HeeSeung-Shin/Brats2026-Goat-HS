#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate externally supplied fold-specific ET-aware sampling weights."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


MIN_WEIGHT = 1.1
MAX_WEIGHT = 2.0
PAPER_ANNOTATED_MASS = 0.555
DEFAULT_MASS_TOLERANCE = 0.01


def read_case_ids(path: Path) -> list[str]:
    case_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not case_ids:
        raise ValueError(f"No case IDs in {path}")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"Duplicate case IDs in {path}")
    return case_ids


def read_fold_training_ids(path: Path, fold: int) -> list[str]:
    splits = json.loads(path.read_text())
    if not isinstance(splits, list) or not 0 <= fold < len(splits):
        raise ValueError(f"Fold {fold} is not available in {path}")
    split = splits[fold]
    if not isinstance(split, dict):
        raise ValueError(f"Fold {fold} must be a JSON object in {path}")
    training = split.get("train")
    if not isinstance(training, list) or not training:
        raise ValueError(f"Fold {fold} has no nonempty train list in {path}")
    case_ids = [str(case_id) for case_id in training]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"Duplicate training case IDs in fold {fold} of {path}")
    return case_ids


def read_annotated_manifest_ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "case_id" not in (reader.fieldnames or ()) or "role" not in (reader.fieldnames or ()):
            raise ValueError(f"Manifest must contain case_id and role columns: {path}")
        annotated = {
            row["case_id"]
            for row in reader
            if row.get("role") == "original_labeled"
        }
    if not annotated:
        raise ValueError(f"Manifest contains no original_labeled cases: {path}")
    return annotated


def load_case_weights(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Weight JSON must contain an object")
    raw = payload.get("case_weights", payload)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Weight JSON must contain a nonempty case_weights object")
    weights: dict[str, float] = {}
    for case_id, value in raw.items():
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Every weight key must be a nonempty case ID string")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Weight for {case_id} must be a JSON number: {value!r}")
        weights[case_id] = float(value)
    return weights


def normalized_replacement_probabilities(
    case_weights: dict[str, float], expected_case_ids: list[str]
) -> list[float]:
    expected = set(expected_case_ids)
    actual = set(case_weights)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Weight case IDs differ: missing={missing[:5]}, extra={extra[:5]}")
    values = [case_weights[case_id] for case_id in expected_case_ids]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Weights contain NaN or Inf")
    if min(values) < MIN_WEIGHT or max(values) > MAX_WEIGHT:
        raise ValueError(
            f"Weights must be within [{MIN_WEIGHT}, {MAX_WEIGHT}], "
            f"got [{min(values)}, {max(values)}]"
        )
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Weight sum must be finite and positive")
    return [value / total for value in values]


def annotated_sampling_mass(
    probabilities: list[float], case_ids: list[str], annotated_case_ids: set[str]
) -> float:
    if not set(case_ids) & annotated_case_ids:
        raise ValueError("No annotated IDs occur in the training fold")
    return math.fsum(
        probability
        for case_id, probability in zip(case_ids, probabilities)
        if case_id in annotated_case_ids
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="One fold's private weight JSON.")
    training = parser.add_mutually_exclusive_group(required=True)
    training.add_argument(
        "--training-cases", type=Path, help="One training case ID per line, in sampling order."
    )
    training.add_argument("--splits-json", type=Path, help="Read training IDs from this splits_final.json.")
    parser.add_argument("--fold", type=int, choices=range(5), help="Required with --splits-json.")
    annotated = parser.add_mutually_exclusive_group()
    annotated.add_argument(
        "--annotated-cases",
        type=Path,
        help="Optional annotated training IDs; when given, check sampling mass near 55.5%%.",
    )
    annotated.add_argument(
        "--manifest", type=Path, help="Read annotated IDs from a Dataset007 manifest."
    )
    parser.add_argument("--mass-tolerance", type=float, default=DEFAULT_MASS_TOLERANCE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.splits_json is not None:
        if args.fold is None:
            raise ValueError("--fold is required with --splits-json")
        case_ids = read_fold_training_ids(args.splits_json, args.fold)
    else:
        if args.fold is not None:
            raise ValueError("--fold is only valid with --splits-json")
        case_ids = read_case_ids(args.training_cases)
    weights = load_case_weights(args.weights)
    probabilities = normalized_replacement_probabilities(weights, case_ids)
    print(
        f"PASS: {len(case_ids)} weights; range=[{min(weights.values()):.6g}, "
        f"{max(weights.values()):.6g}]; normalized_sum={math.fsum(probabilities):.12f}"
    )
    annotated_ids: set[str] | None = None
    if args.annotated_cases:
        annotated_ids = set(read_case_ids(args.annotated_cases))
    elif args.manifest:
        annotated_ids = read_annotated_manifest_ids(args.manifest)
    if annotated_ids is not None:
        mass = annotated_sampling_mass(probabilities, case_ids, annotated_ids)
        if abs(mass - PAPER_ANNOTATED_MASS) > args.mass_tolerance:
            raise ValueError(
                f"Annotated sampling mass {mass:.6f} is not within "
                f"{PAPER_ANNOTATED_MASS:.3f}±{args.mass_tolerance:.3f}"
            )
        print(f"PASS: annotated sampling mass={mass:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
