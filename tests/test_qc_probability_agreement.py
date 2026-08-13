from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from brats_regions import (
    FUSION_WEIGHTS_M,
    class_probabilities_to_region_masks,
    fuse_teacher_region_probabilities,
    region_masks_to_labels,
)
from fuse_resencM_L_pseudolabels import (
    apply_component_filtering,
    assign_selection,
    et_agreement_pass,
    region_agreement_metrics,
    teacher_region_masks,
)
from mlconsensus_common import probability_quality, remove_small_components


def qc_args(**overrides: object) -> SimpleNamespace:
    values = {
        "min_mean_fg_confidence": 0.70,
        "max_mean_fg_normalized_entropy": 0.35,
        "min_mean_fg_margin": 0.20,
        "min_highconf_fg_fraction": 0.60,
        "min_foreground_ratio": 0.00001,
        "max_foreground_ratio": 0.30,
        "min_dice_ml_wt": 0.85,
        "min_dice_ml_tc": 0.70,
        "min_dice_ml_et": 0.50,
        "et_absent_agreement_ok": True,
        "et_one_sided_small_volume_mm3": 5.0,
        "min_wt_component_volume_mm3": 5.0,
        "min_tc_component_volume_mm3": 3.0,
        "min_et_component_volume_mm3": 1.0,
        "component_min_mean_probability": 0.15,
        "component_min_max_probability": 0.30,
        "connectivity": 26,
        "et_suppression_enabled": True,
        "et_suppression_volume_mm3": 1.0,
        "et_suppression_max_probability": 0.35,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def passing_row() -> dict[str, object]:
    return {
        "mean_fg_confidence": 0.70,
        "mean_fg_normalized_entropy": 0.35,
        "mean_fg_margin": 0.20,
        "highconf_fg_fraction": 0.60,
        "foreground_ratio_fused": 0.10,
        "dice_ML_WT": 0.85,
        "dice_ML_TC": 0.70,
        "dice_ML_ET": 0.50,
        "et_absent_M": False,
        "et_absent_L": False,
        "volume_M_ET_mm3": 10.0,
        "volume_L_ET_mm3": 10.0,
    }


def test_probability_regions_can_differ_from_four_class_argmax() -> None:
    probabilities = np.asarray([0.40, 0.30, 0.20, 0.10], dtype=np.float32)[:, None, None, None]
    assert int(np.argmax(probabilities[:, 0, 0, 0])) == 0
    masks = teacher_region_masks(probabilities, threshold=0.50)
    assert not masks["ET"][0, 0, 0]
    assert not masks["TC"][0, 0, 0]
    assert masks["WT"][0, 0, 0]


def test_teacher_region_masks_are_nested_and_include_threshold_boundary() -> None:
    probabilities = np.asarray([0.50, 0.00, 0.00, 0.50], dtype=np.float32)[:, None, None, None]
    masks = class_probabilities_to_region_masks(probabilities, threshold=0.50)
    assert masks["ET"][0, 0, 0]
    assert np.all(masks["ET"] <= masks["TC"])
    assert np.all(masks["TC"] <= masks["WT"])


def test_region_fusion_weights_are_070_050_040() -> None:
    probabilities_m = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)[:, None]
    probabilities_l = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)[:, None]
    fused = fuse_teacher_region_probabilities(probabilities_m, probabilities_l)
    assert FUSION_WEIGHTS_M == {"ET": 0.70, "TC": 0.50, "WT": 0.40}
    assert float(fused["ET"][0]) == pytest.approx(0.70)
    assert float(fused["TC"][0]) == pytest.approx(0.50)
    assert float(fused["WT"][0]) == pytest.approx(0.40)


def test_high_confidence_probability_threshold_includes_075() -> None:
    foreground = np.ones((1,), dtype=bool)
    at_boundary = np.asarray([0.75, 0.10, 0.10, 0.05], dtype=np.float32)[:, None]
    below_boundary = np.asarray(
        [np.nextafter(np.float32(0.75), np.float32(0.0)), 0.10, 0.10, 0.05],
        dtype=np.float32,
    )[:, None]
    assert probability_quality(at_boundary, foreground)["highconf_fg_fraction"] == 1.0
    assert probability_quality(below_boundary, foreground)["highconf_fg_fraction"] == 0.0


@pytest.mark.parametrize("minimum", [5.0, 3.0, 1.0])
def test_component_volume_thresholds_are_strict(minimum: float) -> None:
    mask = np.ones((1, 1, 1), dtype=bool)
    probability = np.full(mask.shape, 0.10, dtype=np.float32)
    removed, _ = remove_small_components(
        mask,
        probability,
        voxel_volume_mm3=minimum / 2.0,
        min_volume_mm3=minimum,
        min_mean_probability=0.15,
        min_max_probability=0.30,
        connectivity=26,
    )
    boundary, _ = remove_small_components(
        mask,
        probability,
        voxel_volume_mm3=minimum,
        min_volume_mm3=minimum,
        min_mean_probability=0.15,
        min_max_probability=0.30,
        connectivity=26,
    )
    assert not removed.any()
    assert boundary.any()


def test_component_probability_threshold_boundaries_are_kept() -> None:
    mask = np.ones((1, 1, 2), dtype=bool)
    for probability in (
        np.full(mask.shape, 0.15, dtype=np.float32),
        np.asarray([[[0.0, 0.30]]], dtype=np.float32),
    ):
        filtered, _ = remove_small_components(
            mask,
            probability,
            voxel_volume_mm3=1.0,
            min_volume_mm3=5.0,
            min_mean_probability=0.15,
            min_max_probability=0.30,
            connectivity=26,
        )
        assert filtered.all()


def test_et_suppression_uses_1_mm3_and_035_strict_boundaries() -> None:
    label = np.asarray([[[3]]], dtype=np.uint8)
    low = {name: np.asarray([[[value]]], dtype=np.float32) for name, value in {"ET": 0.34, "TC": 0.60, "WT": 0.70}.items()}
    filtered, stats = apply_component_filtering(label, low, voxel_volume_mm3=0.5, args=qc_args())
    assert not (filtered == 3).any()
    assert stats["component_ET_et_suppressed"] is True

    at_probability_boundary = dict(low)
    at_probability_boundary["ET"] = np.asarray([[[0.35]]], dtype=np.float32)
    filtered, _ = apply_component_filtering(
        label, at_probability_boundary, voxel_volume_mm3=0.5, args=qc_args()
    )
    assert (filtered == 3).any()

    filtered, _ = apply_component_filtering(label, low, voxel_volume_mm3=1.0, args=qc_args())
    assert (filtered == 3).any()


def test_both_empty_teacher_masks_are_complete_agreement_and_et_passes() -> None:
    empty = np.zeros((1, 1, 2), dtype=bool)
    regions = {"ET": empty, "TC": empty, "WT": empty}
    fused = np.zeros_like(empty, dtype=np.uint8)
    row = region_agreement_metrics(regions, regions, fused, voxel_volume_mm3=1.0)
    assert row["dice_ML_ET"] == 1.0
    assert row["dice_ML_TC"] == 1.0
    assert et_agreement_pass(row, qc_args(), []) is True


def test_one_sided_et_exception_is_strictly_less_than_5_mm3() -> None:
    args = qc_args()
    row = passing_row()
    row.update({"et_absent_M": True, "et_absent_L": False, "volume_L_ET_mm3": 4.999})
    assert et_agreement_pass(row, args, []) is True
    row["volume_L_ET_mm3"] = 5.0
    assert et_agreement_pass(row, args, []) is False


def test_agreement_and_confidence_threshold_boundaries_are_inclusive() -> None:
    row = passing_row()
    status, passes = assign_selection(row, qc_args(), [])
    assert status == "strict"
    assert passes["ml_wt_pass"] and passes["ml_tc_pass"] and passes["ml_et_pass"]
    assert passes["I_strict"] == (passes["I_conf"] and passes["I_agree"])


@pytest.mark.parametrize(
    ("key", "toward"),
    [
        ("mean_fg_confidence", 0.0),
        ("mean_fg_normalized_entropy", 1.0),
        ("mean_fg_margin", 0.0),
        ("highconf_fg_fraction", 0.0),
        ("dice_ML_WT", 0.0),
        ("dice_ML_TC", 0.0),
        ("dice_ML_ET", 0.0),
    ],
)
def test_qc_threshold_fails_immediately_outside_boundary(key: str, toward: float) -> None:
    row = passing_row()
    row[key] = float(np.nextafter(float(row[key]), toward))
    status, _ = assign_selection(row, qc_args(), [])
    assert status == "exclude"


def test_foreground_minimum_boundary_is_inclusive() -> None:
    row = passing_row()
    row["foreground_ratio_fused"] = 0.00001
    assert assign_selection(row, qc_args(), [])[0] == "strict"
    row["foreground_ratio_fused"] = float(np.nextafter(0.00001, 0.0))
    assert assign_selection(row, qc_args(), [])[0] == "exclude"


def test_foreground_maximum_boundary_is_inclusive() -> None:
    row = passing_row()
    row["foreground_ratio_fused"] = 0.30
    assert assign_selection(row, qc_args(), [])[0] == "strict"
    row["foreground_ratio_fused"] = float(np.nextafter(0.30, 1.0))
    assert assign_selection(row, qc_args(), [])[0] == "exclude"


def test_mutually_exclusive_label_reconstruction_enforces_hierarchy() -> None:
    masks = {
        "ET": np.asarray([1, 0, 0, 0], dtype=bool),
        "TC": np.asarray([0, 1, 0, 0], dtype=bool),
        "WT": np.asarray([0, 0, 1, 0], dtype=bool),
    }
    labels = region_masks_to_labels(masks)
    assert labels.tolist() == [3, 1, 2, 0]
    reconstructed = {
        "ET": labels == 3,
        "TC": np.logical_or(labels == 1, labels == 3),
        "WT": labels > 0,
    }
    assert np.all(reconstructed["ET"] <= reconstructed["TC"])
    assert np.all(reconstructed["TC"] <= reconstructed["WT"])
