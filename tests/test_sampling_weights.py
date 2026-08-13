from __future__ import annotations

import json
import math

import pytest

from scripts.data.validate_case_weights import (
    annotated_sampling_mass,
    load_case_weights,
    normalized_replacement_probabilities,
    read_annotated_manifest_ids,
    read_fold_training_ids,
)


def test_weights_are_normalized_for_replacement_sampling() -> None:
    case_ids = ["annotated", "pseudo"]
    probabilities = normalized_replacement_probabilities(
        {"annotated": 1.665, "pseudo": 1.335}, case_ids
    )
    assert math.fsum(probabilities) == pytest.approx(1.0)
    assert probabilities == pytest.approx([0.555, 0.445])
    assert annotated_sampling_mass(probabilities, case_ids, {"annotated"}) == pytest.approx(0.555)


def test_weight_validation_rejects_range_and_case_id_errors() -> None:
    with pytest.raises(ValueError, match="within"):
        normalized_replacement_probabilities({"case": 1.0}, ["case"])
    with pytest.raises(ValueError, match="case IDs differ"):
        normalized_replacement_probabilities({"wrong": 1.1}, ["case"])
    with pytest.raises(ValueError, match="NaN or Inf"):
        normalized_replacement_probabilities({"case": float("nan")}, ["case"])


def test_weight_json_requires_numeric_values(tmp_path) -> None:
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({"case_weights": {"case": "1.1"}}))
    with pytest.raises(ValueError, match="JSON number"):
        load_case_weights(weights)


def test_split_and_manifest_ids_are_read_without_a_weight_formula(tmp_path) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps([{"train": ["annotated", "pseudo"], "val": []}]))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "case_id,role\nannotated,original_labeled\npseudo,pseudo_labeled_qc_strict\n"
    )
    assert read_fold_training_ids(splits, 0) == ["annotated", "pseudo"]
    assert read_annotated_manifest_ids(manifest) == {"annotated"}
