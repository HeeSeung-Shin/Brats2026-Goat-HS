"""Optional CPU PyTorch test; skipped by lightweight CI when torch is absent."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
TRAINER_FILE = (
    ROOT
    / "src/nnunet_overlays/nnunetv2/training/nnUNetTrainer/variants/nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot.py"
)


def load_adapter_class():
    spec = importlib.util.spec_from_file_location("brats_softmoe_trainer", TRAINER_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BottleneckSoftMoEAdapter


def test_adapter_shape_gate_sum_and_balance_loss_definition() -> None:
    adapter_class = load_adapter_class()
    adapter = adapter_class(
        channels=320,
        num_experts=4,
        temperature=1.0,
        reduction=4,
        adapter_scale_init=0.1,
    )
    calls = [0, 0, 0, 0]
    handles = []
    for index, expert in enumerate(adapter.experts):
        def count_call(_module, _inputs, _output, *, expert_index=index):
            calls[expert_index] += 1

        handles.append(expert.register_forward_hook(count_call))
    inputs = torch.randn(2, 320, 2, 2, 2)
    outputs = adapter(inputs)
    for handle in handles:
        handle.remove()
    assert outputs.shape == inputs.shape
    assert calls == [1, 1, 1, 1]
    assert adapter.num_experts == 4
    assert adapter.temperature == 1.0
    assert adapter.alpha.item() == pytest.approx(0.1)
    assert adapter.experts[0].net[1].out_channels == 80
    assert adapter.experts[0].net[3].in_channels == 80
    assert adapter.experts[0].net[3].out_channels == 80
    assert adapter.experts[0].net[5].out_channels == 320
    assert adapter.last_gate is not None
    torch.testing.assert_close(adapter.last_gate.sum(dim=1), torch.ones(2))
    mean_gate = adapter.last_gate.mean(dim=0)
    expected_balance = torch.sum((mean_gate - 0.25) ** 2)
    torch.testing.assert_close(adapter.balance_loss().detach(), expected_balance)
