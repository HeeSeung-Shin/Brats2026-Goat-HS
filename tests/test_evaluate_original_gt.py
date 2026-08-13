from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.evaluate_original_gt import dice, dice_summary, nanmean


def test_dice_empty_policy() -> None:
    empty = np.zeros((2, 2, 2), dtype=np.uint8)
    one = empty.copy()
    one[0, 0, 0] = 1

    assert np.isnan(dice(empty, empty))
    assert dice(empty, one) == 0.0
    assert dice(one, empty) == 0.0
    assert dice(one, one) == 1.0


def test_dice_known_partial_overlap() -> None:
    a = np.asarray([1, 1, 0, 0], dtype=np.uint8)
    b = np.asarray([0, 1, 1, 0], dtype=np.uint8)
    assert dice(a, b) == 0.5


def test_regional_summary_excludes_nan_before_publication_mean() -> None:
    frame = pd.DataFrame(
        {
            "Dice_ET": [1.0, np.nan, 0.5],
            "Dice_TC": [0.9, 0.7, 0.8],
            "Dice_WT": [0.8, 0.6, 0.7],
            "casewise_mean_dice": [0.9, np.nan, 2.0 / 3.0],
        }
    )
    summary, rows = dice_summary(frame)

    assert summary["regions"]["ET"]["valid_n"] == 2
    assert summary["regions"]["ET"]["excluded_both_empty_n"] == 1
    assert summary["regions"]["ET"]["mean"] == 0.75
    assert summary["regions"]["TC"]["valid_n"] == 3
    assert summary["regions"]["WT"]["valid_n"] == 3
    assert summary["publication_mean_dsc"] == np.mean([0.75, 0.8, 0.7])
    assert rows[-1]["statistic"] == "publication_mean_dsc_from_three_regional_means"


def test_region_with_no_valid_values_keeps_nan_mean() -> None:
    frame = pd.DataFrame(
        {
            "Dice_ET": [np.nan, np.nan],
            "Dice_TC": [1.0, 1.0],
            "Dice_WT": [1.0, 1.0],
        }
    )
    summary, _ = dice_summary(frame)
    assert summary["regions"]["ET"]["valid_n"] == 0
    assert np.isnan(summary["regions"]["ET"]["mean"])
    assert np.isnan(summary["publication_mean_dsc"])


def test_casewise_mean_excludes_both_empty_region() -> None:
    assert nanmean([np.nan, 0.5, 1.0]) == 0.75
    assert np.isnan(nanmean([np.nan, np.nan, np.nan]))
