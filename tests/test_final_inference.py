from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.final_inference import (
    MIRROR_AXES,
    PATCH_SIZE,
    STEP_SIZE,
    average_fold_logits_and_softmax,
    build_parser,
    configure_k4_environment,
    final_region_mapping,
    validate_final_plans,
)


ROOT = Path(__file__).resolve().parents[1]


def test_final_inference_forces_k4_environment(monkeypatch) -> None:
    monkeypatch.setenv("SOFTMOE_NUM_EXPERTS", "2")
    configure_k4_environment()
    assert os.environ["SOFTMOE_NUM_EXPERTS"] == "4"
    assert os.environ["SOFTMOE_TEMPERATURE"] == "1.0"
    assert os.environ["SOFTMOE_REDUCTION"] == "4"
    assert PATCH_SIZE == (128, 160, 112)
    assert STEP_SIZE == 0.5
    assert MIRROR_AXES == (0, 1, 2)


def test_final_inference_region_mapping_is_not_argmax() -> None:
    probabilities = np.asarray([0.40, 0.30, 0.20, 0.10], dtype=np.float32)[:, None]
    assert int(np.argmax(probabilities[:, 0])) == 0
    assert final_region_mapping(probabilities).tolist() == [2]


def test_final_inference_hierarchy_and_mutually_exclusive_mapping() -> None:
    probabilities = np.asarray(
        [
            [0.40, 0.40, 0.40, 1.00],
            [0.10, 0.60, 0.00, 0.00],
            [0.00, 0.00, 0.60, 0.00],
            [0.50, 0.00, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    assert final_region_mapping(probabilities).tolist() == [3, 1, 2, 0]


def test_five_fold_logits_are_equally_averaged_before_softmax() -> None:
    folds = [np.asarray([[float(index)], [0.0]], dtype=np.float32) for index in range(5)]
    probabilities = average_fold_logits_and_softmax(folds)
    expected = np.exp(2.0) / (np.exp(2.0) + 1.0)
    assert probabilities.shape == (2, 1)
    np.testing.assert_allclose(probabilities[0, 0], expected, rtol=1e-6)


def test_final_plans_fix_resencm_architecture_and_training_shape() -> None:
    plans = json.loads(
        (ROOT / "config" / "nnUNetResEncUNetMPlans_D005Compat.json").read_text()
    )
    assert list(plans["configurations"]) == ["3d_fullres"]
    validate_final_plans(plans, "3d_fullres")
    architecture = plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]
    assert architecture["features_per_stage"] == [32, 64, 128, 256, 320, 320]
    assert architecture["n_blocks_per_stage"] == [1, 3, 4, 6, 6, 6]

    architecture["features_per_stage"][-1] = 321
    with pytest.raises(ValueError, match="features_per_stage"):
        validate_final_plans(plans, "3d_fullres")


def test_final_inference_cli_requires_five_explicit_checkpoints() -> None:
    parser = build_parser()
    common = [
        "--input-dir", "inputs", "--output-dir", "outputs",
        "--plans-json", "plans.json", "--dataset-json", "dataset.json",
        "--configuration", "3d_fullres",
    ]
    checkpoints = [
        item
        for fold in range(5)
        for item in (f"--checkpoint-fold{fold}", f"fold_{fold}/checkpoint_best.pth")
    ]
    args = parser.parse_args(common + checkpoints)
    assert [getattr(args, f"checkpoint_fold{fold}").name for fold in range(5)] == [
        "checkpoint_best.pth"
    ] * 5
