# SPDX-License-Identifier: Apache-2.0
"""Pure NumPy region operations shared by pseudo-labeling and final inference."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


REGIONS = ("ET", "TC", "WT")
FUSION_WEIGHTS_M = {"ET": 0.70, "TC": 0.50, "WT": 0.40}


def class_probabilities_to_region_probabilities(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    """Convert class-first ``[BG, NCR, ED, ET]`` probabilities to nested regions."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim < 2 or probabilities.shape[0] != 4:
        raise ValueError(f"Expected class-first four-class probabilities, got {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain NaN or Inf")
    return {
        "ET": probabilities[3],
        "TC": probabilities[1] + probabilities[3],
        "WT": probabilities[1] + probabilities[2] + probabilities[3],
    }


def enforce_region_hierarchy(
    mask_et: np.ndarray,
    mask_tc: np.ndarray,
    mask_wt: np.ndarray,
) -> dict[str, np.ndarray]:
    """Enforce ET ⊆ TC ⊆ WT using Boolean unions."""
    et = np.asarray(mask_et, dtype=bool)
    tc = np.logical_or(np.asarray(mask_tc, dtype=bool), et)
    wt = np.logical_or(np.asarray(mask_wt, dtype=bool), tc)
    if et.shape != tc.shape or tc.shape != wt.shape:
        raise ValueError(f"Region shapes differ: ET={et.shape}, TC={tc.shape}, WT={wt.shape}")
    return {"ET": et, "TC": tc, "WT": wt}


def region_probabilities_to_masks(
    region_probabilities: Mapping[str, np.ndarray],
    threshold: float = 0.50,
) -> dict[str, np.ndarray]:
    """Threshold ET/TC/WT independently, then enforce their hierarchy."""
    missing = set(REGIONS) - set(region_probabilities)
    if missing:
        raise ValueError(f"Missing region probabilities: {sorted(missing)}")
    return enforce_region_hierarchy(
        np.asarray(region_probabilities["ET"]) >= threshold,
        np.asarray(region_probabilities["TC"]) >= threshold,
        np.asarray(region_probabilities["WT"]) >= threshold,
    )


def class_probabilities_to_region_masks(
    probabilities: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, np.ndarray]:
    return region_probabilities_to_masks(
        class_probabilities_to_region_probabilities(probabilities),
        threshold=threshold,
    )


def fuse_teacher_region_probabilities(
    probabilities_m: np.ndarray,
    probabilities_l: np.ndarray,
    weights_m: Mapping[str, float] = FUSION_WEIGHTS_M,
) -> dict[str, np.ndarray]:
    """Fuse ResEnc-M/L region probabilities with the paper's region weights."""
    regions_m = class_probabilities_to_region_probabilities(probabilities_m)
    regions_l = class_probabilities_to_region_probabilities(probabilities_l)
    fused: dict[str, np.ndarray] = {}
    for region in REGIONS:
        weight_m = float(weights_m[region])
        if not 0.0 <= weight_m <= 1.0:
            raise ValueError(f"Invalid {region} ResEnc-M fusion weight: {weight_m}")
        fused[region] = weight_m * regions_m[region] + (1.0 - weight_m) * regions_l[region]
    return fused


def region_masks_to_labels(region_masks: Mapping[str, np.ndarray]) -> np.ndarray:
    """Map nested masks to mutually exclusive BG=0, NCR=1, ED=2, ET=3 labels."""
    nested = enforce_region_hierarchy(
        region_masks["ET"], region_masks["TC"], region_masks["WT"]
    )
    label = np.zeros(nested["WT"].shape, dtype=np.uint8)
    label[np.logical_and(nested["WT"], ~nested["TC"])] = 2
    label[np.logical_and(nested["TC"], ~nested["ET"])] = 1
    label[nested["ET"]] = 3
    return label


def class_probabilities_to_labels(probabilities: np.ndarray, threshold: float = 0.50) -> np.ndarray:
    return region_masks_to_labels(
        class_probabilities_to_region_masks(probabilities, threshold=threshold)
    )
