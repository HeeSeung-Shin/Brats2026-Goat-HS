from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SEMANTICS = (
    ROOT
    / "src/nnunet_overlays/nnunetv2/training/nnUNetTrainer/variants/method_semantics.py"
)


def load_pseudo_label_weight():
    spec = importlib.util.spec_from_file_location("brats_method_semantics", SEMANTICS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pseudo_label_weight


def test_pseudo_label_curriculum() -> None:
    weight = load_pseudo_label_weight()
    assert weight(0) == pytest.approx(0.3)
    assert weight(49) == pytest.approx(0.3)
    assert weight(50) == pytest.approx(0.3)
    assert weight(100) == pytest.approx(0.5)
    assert weight(149) == pytest.approx(0.696)
    assert weight(150) == pytest.approx(0.7)
    assert weight(500) == pytest.approx(0.7)
