# SPDX-License-Identifier: Apache-2.0
"""Dependency-free scalar semantics used by the final trainer and unit tests."""

from __future__ import annotations


def pseudo_label_weight(epoch: int) -> float:
    """Paper Eq. 11: 0.3 before 50, linear to 0.7 at 150, then 0.7."""
    epoch = int(epoch)
    if epoch < 50:
        return 0.3
    if epoch < 150:
        return 0.3 + 0.4 * (epoch - 50) / 100.0
    return 0.7
