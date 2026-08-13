from __future__ import annotations

import os
import types
from pathlib import Path

import numpy as np
import torch
from torch import autocast, nn

from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainer_D007ETTCRegionAuxHead500 import (
    nnUNetTrainer_D007ETTCRegionAuxHead500,
)
from nnunetv2.utilities.helpers import dummy_context


FINAL_NUM_EXPERTS = 4
FINAL_TEMPERATURE = 1.0
FINAL_ADAPTER_SCALE_INIT = 0.1
FINAL_REDUCTION = 4
FINAL_BALANCE_WEIGHT = 0.001
FINAL_EPOCHS = 500
FINAL_TRAIN_ITERATIONS = 250
FINAL_VALIDATION_ITERATIONS = 50


def _group_count(channels: int, preferred: int = 8) -> int:
    for groups in (preferred, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class BottleneckSoftMoEExpert(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // int(reduction), 8)
        self.net = nn.Sequential(
            nn.GroupNorm(_group_count(channels), channels),
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BottleneckSoftMoEAdapter(nn.Module):
    """Patch/tile-level dense soft routing over residual bottleneck experts."""

    def __init__(
        self,
        channels: int,
        num_experts: int,
        temperature: float = 1.0,
        reduction: int = 4,
        adapter_scale_init: float = 0.1,
    ):
        super().__init__()
        if num_experts < 2:
            raise RuntimeError(f"SOFTMOE_NUM_EXPERTS must be >=2, got {num_experts}")
        self.channels = int(channels)
        self.num_experts = int(num_experts)
        self.temperature = float(temperature)
        hidden = max(self.channels // 4, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(self.channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.num_experts),
        )
        self.experts = nn.ModuleList(
            [BottleneckSoftMoEExpert(self.channels, reduction=reduction) for _ in range(self.num_experts)]
        )
        self.alpha = nn.Parameter(torch.tensor(float(adapter_scale_init), dtype=torch.float32))
        self.last_gate: torch.Tensor | None = None
        self.last_balance_loss: torch.Tensor | None = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        temperature = max(float(self.temperature), 1e-6)
        logits = self.gate(z)
        gate = torch.softmax(logits / temperature, dim=1)
        correction = torch.zeros_like(z)
        for idx, expert in enumerate(self.experts):
            weight = gate[:, idx].view(-1, 1, 1, 1, 1)
            correction = correction + weight * expert(z)
        mean_gate = gate.mean(dim=0)
        target = torch.full_like(mean_gate, 1.0 / float(self.num_experts))
        self.last_gate = gate.detach()
        self.last_balance_loss = torch.sum((mean_gate - target) ** 2)
        return z + self.alpha.to(dtype=z.dtype) * correction

    def balance_loss(self) -> torch.Tensor:
        if self.last_balance_loss is None:
            return self.alpha.new_tensor(0.0)
        return self.last_balance_loss


def attach_softmoe_to_network(
    network: nn.Module,
    num_experts: int,
    temperature: float,
    adapter_scale_init: float,
    reduction: int = 4,
) -> nn.Module:
    actual = (int(num_experts), float(temperature), float(adapter_scale_init), int(reduction))
    expected = (
        FINAL_NUM_EXPERTS,
        FINAL_TEMPERATURE,
        FINAL_ADAPTER_SCALE_INIT,
        FINAL_REDUCTION,
    )
    if actual != expected:
        raise RuntimeError(
            "Final ResEnc-M dense adapter requires "
            f"(K, temperature, scale, reduction)={expected}, got {actual}"
        )
    if not hasattr(network, "encoder") or not hasattr(network, "decoder"):
        raise RuntimeError("The dense adapter expects a nnU-Net-like network with encoder and decoder")
    if not hasattr(network.encoder, "output_channels"):
        raise RuntimeError("Unable to infer bottleneck channels: network.encoder.output_channels missing")
    channels = int(network.encoder.output_channels[-1])
    if channels != 320:
        raise RuntimeError(f"Final ResEnc-M adapter requires a 320-channel encoder skip, got {channels}")
    network.softmoe_adapter = BottleneckSoftMoEAdapter(
        channels=channels,
        num_experts=num_experts,
        temperature=temperature,
        reduction=reduction,
        adapter_scale_init=adapter_scale_init,
    )
    network.softmoe_num_experts = int(num_experts)
    network.softmoe_temperature = float(temperature)

    def forward_with_softmoe(self, x):
        skips = list(self.encoder(x))
        skips[-1] = self.softmoe_adapter(skips[-1])
        return self.decoder(skips)

    network.forward = types.MethodType(forward_with_softmoe, network)
    return network


class nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot(nnUNetTrainer_D007ETTCRegionAuxHead500):
    """Final ResEnc-M trainer with the fixed K=4 dense residual adapter.

    The adapter is patch/tile feature-routed, dense, soft, and residual-only.
    """

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: tuple[str, ...] | list[str],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        network = nnUNetTrainer_D007ETTCRegionAuxHead500.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        return attach_softmoe_to_network(
            network,
            num_experts=FINAL_NUM_EXPERTS,
            temperature=FINAL_TEMPERATURE,
            adapter_scale_init=FINAL_ADAPTER_SCALE_INIT,
            reduction=FINAL_REDUCTION,
        )

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.softmoe_num_experts = FINAL_NUM_EXPERTS
        self.softmoe_temperature = FINAL_TEMPERATURE
        self.softmoe_balance_weight = FINAL_BALANCE_WEIGHT
        self.softmoe_adapter_scale_init = FINAL_ADAPTER_SCALE_INIT
        self.num_epochs = FINAL_EPOCHS
        self.num_iterations_per_epoch = FINAL_TRAIN_ITERATIONS
        self.num_val_iterations_per_epoch = FINAL_VALIDATION_ITERATIONS
        self.softmoe_init_checkpoint = os.environ.get("SOFTMOE_INIT_CHECKPOINT", "").strip()
        self._softmoe_init_loaded = False
        self._ensure_logger_keys()
        self.print_to_log_file("D007 final ResEnc-M K=4 dense-adapter trainer initialized")
        self.print_to_log_file(f"  SOFTMOE_NUM_EXPERTS: {self.softmoe_num_experts}")
        self.print_to_log_file(f"  SOFTMOE_TEMPERATURE: {self.softmoe_temperature}")
        self.print_to_log_file(f"  SOFTMOE_BALANCE_WEIGHT: {self.softmoe_balance_weight}")
        self.print_to_log_file(f"  SOFTMOE_ADAPTER_SCALE_INIT: {self.softmoe_adapter_scale_init}")
        self.print_to_log_file(f"  SOFTMOE_MAX_EPOCHS: {self.num_epochs}")
        if self.softmoe_init_checkpoint:
            self.print_to_log_file(f"  SOFTMOE_INIT_CHECKPOINT: {self.softmoe_init_checkpoint}")
        self.print_to_log_file("  routing: softmax over bottleneck global-average pooled feature; no hard argmax")

    @staticmethod
    def _clean_state_key(key: str) -> str:
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        return key.replace("._orig_mod.", ".")

    @staticmethod
    def _is_new_final_model_key(key: str) -> bool:
        return key.startswith("softmoe_adapter.") or key.startswith("region_aux_head.")

    def _load_softmoe_compatible_pretrained(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"SOFTMOE_INIT_CHECKPOINT does not exist: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        pretrained_raw = checkpoint.get("network_weights")
        if pretrained_raw is None:
            raise RuntimeError(f"Checkpoint has no network_weights: {path}")
        pretrained = {self._clean_state_key(str(k)): v for k, v in pretrained_raw.items()}

        mod = self._network_module()
        model_state = mod.state_dict()
        loadable = {}
        skipped_new = []
        for key, value in model_state.items():
            if self._is_new_final_model_key(key):
                skipped_new.append(key)
                continue
            if key not in pretrained:
                raise RuntimeError(
                    f"D005 init checkpoint is missing non-new model key {key!r}. "
                    "Only softmoe_adapter.* and region_aux_head.* may be absent."
                )
            if tuple(value.shape) != tuple(pretrained[key].shape):
                raise RuntimeError(
                    f"D005 init checkpoint shape mismatch for {key!r}: "
                    f"checkpoint={tuple(pretrained[key].shape)}, model={tuple(value.shape)}"
                )
            loadable[key] = pretrained[key]

        updated_state = dict(model_state)
        updated_state.update(loadable)
        mod.load_state_dict(updated_state, strict=True)
        self._softmoe_init_loaded = True
        self.print_to_log_file(
            "Final-model D005-compatible initialization loaded: "
            f"{path}; loaded_tensors={len(loadable)}, skipped_new_tensors={len(skipped_new)}"
        )

    def initialize(self):
        was_initialized_before = self.was_initialized
        super().initialize()
        if (
            not was_initialized_before
            and self.softmoe_init_checkpoint
            and not self._softmoe_init_loaded
        ):
            self._load_softmoe_compatible_pretrained(self.softmoe_init_checkpoint)

    def _softmoe_adapter(self) -> BottleneckSoftMoEAdapter:
        mod = self._network_module()
        if not hasattr(mod, "softmoe_adapter"):
            raise RuntimeError("Network is missing softmoe_adapter")
        return mod.softmoe_adapter

    def _balance_loss(self) -> torch.Tensor:
        return self._softmoe_adapter().balance_loss()

    def train_step(self, batch: dict) -> dict:
        if "keys" not in batch:
            raise RuntimeError("Batch dict has no keys; cannot apply pseudo weighting safely.")
        keys = list(batch["keys"])
        roles = self._roles_for_keys(keys)
        if any(role == "dycon_unlabeled" for role in roles):
            bad = [key for key, role in zip(keys, roles) if role == "dycon_unlabeled"]
            raise RuntimeError(f"dycon_unlabeled cases appeared in supervised training batch: {bad}")

        gt_indices = [idx for idx, role in enumerate(roles) if role == "original_labeled"]
        pseudo_indices = [idx for idx, role in enumerate(roles) if role == "pseudo_labeled_highconf"]
        unknown_roles = sorted({role for role in roles if role not in {"original_labeled", "pseudo_labeled_highconf"}})
        if unknown_roles:
            raise RuntimeError(f"Unsupported Dataset007 case roles in supervised batch: {unknown_roles}")

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        lambda_pseudo = self._pseudo_lambda()
        loss_gt_value = np.nan
        loss_pseudo_value = np.nan
        region_gt_value = np.nan
        region_pseudo_value = np.nan

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            losses = []
            if gt_indices:
                idx = torch.as_tensor(gt_indices, dtype=torch.long, device=data.device)
                output_gt = self._index_nested_batch(output, idx)
                target_gt = self._index_nested_batch(target, idx)
                loss_gt, base_gt, region_gt = self._loss_with_region(output_gt, target_gt)
                loss_gt_value = float(base_gt.detach().cpu().numpy())
                region_gt_value = float(region_gt.detach().cpu().numpy())
                losses.append(loss_gt)
            if pseudo_indices:
                idx = torch.as_tensor(pseudo_indices, dtype=torch.long, device=data.device)
                output_pseudo = self._index_nested_batch(output, idx)
                target_pseudo = self._index_nested_batch(target, idx)
                loss_pseudo, base_pseudo, region_pseudo = self._loss_with_region(output_pseudo, target_pseudo)
                loss_pseudo_value = float(base_pseudo.detach().cpu().numpy())
                region_pseudo_value = float(region_pseudo.detach().cpu().numpy())
                losses.append(lambda_pseudo * loss_pseudo)
            if not losses:
                raise RuntimeError(f"No usable supervised samples in batch. keys={keys}, roles={roles}")
            balance_loss = self._balance_loss()
            total_loss = sum(losses) + float(self.softmoe_balance_weight) * balance_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        out = {
            "loss": total_loss.detach().cpu().numpy(),
            "loss_gt": loss_gt_value,
            "loss_pseudo": loss_pseudo_value,
            "region_loss_gt": region_gt_value,
            "region_loss_pseudo": region_pseudo_value,
            "n_gt": len(gt_indices),
            "n_pseudo": len(pseudo_indices),
            "lambda_pseudo": lambda_pseudo,
        }
        return out
