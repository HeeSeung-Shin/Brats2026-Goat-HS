from __future__ import annotations

import csv
import math
import os
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast, nn

from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainer_D007ETTCRegionAuxHead500 import (
    nnUNetTrainer_D007ETTCRegionAuxHead500,
)
from nnunetv2.utilities.helpers import dummy_context


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return default if value == "" else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    return default if value == "" else float(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if value == "":
        return default
    return value in {"1", "true", "yes", "y", "on"}


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
    """Case-level soft routing over lightweight residual bottleneck experts."""

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
        self.last_gate_logits: torch.Tensor | None = None
        self.last_balance_loss: torch.Tensor | None = None
        self.last_gate_entropy: torch.Tensor | None = None

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
        entropy = -(gate * torch.log(gate.clamp_min(1e-8))).sum(dim=1)
        self.last_gate = gate.detach()
        self.last_gate_logits = logits.detach()
        self.last_balance_loss = torch.sum((mean_gate - target) ** 2)
        self.last_gate_entropy = entropy.detach()
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
    if not hasattr(network, "encoder") or not hasattr(network, "decoder"):
        raise RuntimeError("SoftMoE pilot expects a nnU-Net-like network with encoder and decoder")
    if not hasattr(network.encoder, "output_channels"):
        raise RuntimeError("Unable to infer bottleneck channels: network.encoder.output_channels missing")
    channels = int(network.encoder.output_channels[-1])
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
    """Fold0 pilot trainer with configurable bottleneck SoftMoE residual adapter.

    The adapter is feature-routed, soft, and residual-only. It does not use EDA
    cluster assignments, source labels, or hard argmax expert routing.
    """

    custom_logger_keys = nnUNetTrainer_D007ETTCRegionAuxHead500.custom_logger_keys + (
        "moe/balance_loss",
        "moe/gate_entropy",
        "moe/gate_max_mean",
        "moe/adapter_scale",
        "moe/temperature",
        "moe/num_experts",
        "moe/collapse_warning",
        "moe/val_gate_entropy",
        "moe/val_gate_max_mean",
        "moe/val_dominant_expert_frac",
        "moe/val_collapse_warning",
    )

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
            num_experts=_env_int("SOFTMOE_NUM_EXPERTS", 2),
            temperature=_env_float("SOFTMOE_TEMPERATURE", 1.0),
            adapter_scale_init=_env_float("SOFTMOE_ADAPTER_SCALE_INIT", 0.1),
            reduction=_env_int("SOFTMOE_REDUCTION", 4),
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
        self.softmoe_num_experts = _env_int("SOFTMOE_NUM_EXPERTS", 2)
        self.softmoe_temperature = _env_float("SOFTMOE_TEMPERATURE", 1.0)
        self.softmoe_balance_weight = _env_float("SOFTMOE_BALANCE_WEIGHT", 0.001)
        self.softmoe_adapter_scale_init = _env_float("SOFTMOE_ADAPTER_SCALE_INIT", 0.1)
        self.num_epochs = _env_int("SOFTMOE_MAX_EPOCHS", 300)
        self.softmoe_init_checkpoint = os.environ.get("SOFTMOE_INIT_CHECKPOINT", "").strip()
        self._softmoe_init_loaded = False
        if os.environ.get("SOFTMOE_NUM_ITERATIONS_PER_EPOCH", "").strip():
            self.num_iterations_per_epoch = _env_int("SOFTMOE_NUM_ITERATIONS_PER_EPOCH", self.num_iterations_per_epoch)
        if os.environ.get("SOFTMOE_NUM_VAL_ITERATIONS_PER_EPOCH", "").strip():
            self.num_val_iterations_per_epoch = _env_int(
                "SOFTMOE_NUM_VAL_ITERATIONS_PER_EPOCH", self.num_val_iterations_per_epoch
            )
        self.softmoe_output_dir = Path(
            os.environ.get("SOFTMOE_PILOT_OUTPUT_DIR", "outputs/softmoe_k_pilot")
        )
        self._current_val_gate_rows: list[dict[str, Any]] = []
        self._ensure_logger_keys()
        self._ensure_moe_logger_keys()
        self.print_to_log_file("D007 ETTCRegionAuxHead SoftMoEConfigK pilot initialized")
        self.print_to_log_file(f"  SOFTMOE_NUM_EXPERTS: {self.softmoe_num_experts}")
        self.print_to_log_file(f"  SOFTMOE_TEMPERATURE: {self.softmoe_temperature}")
        self.print_to_log_file(f"  SOFTMOE_BALANCE_WEIGHT: {self.softmoe_balance_weight}")
        self.print_to_log_file(f"  SOFTMOE_ADAPTER_SCALE_INIT: {self.softmoe_adapter_scale_init}")
        self.print_to_log_file(f"  SOFTMOE_MAX_EPOCHS: {self.num_epochs}")
        if self.softmoe_init_checkpoint:
            self.print_to_log_file(f"  SOFTMOE_INIT_CHECKPOINT: {self.softmoe_init_checkpoint}")
        self.print_to_log_file("  routing: softmax over bottleneck global-average pooled feature; no hard argmax")
        self.print_to_log_file("  EDA clusters are not read or used as expert labels.")

    @staticmethod
    def _clean_state_key(key: str) -> str:
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        return key.replace("._orig_mod.", ".")

    @staticmethod
    def _is_new_softmoe_pilot_key(key: str) -> bool:
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
            if self._is_new_softmoe_pilot_key(key):
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
            "SoftMoE pilot D005-compatible initialization loaded: "
            f"{path}; loaded_tensors={len(loadable)}, skipped_new_tensors={len(skipped_new)}"
        )
        if self.local_rank == 0:
            try:
                manifest = Path(self.output_folder) / "softmoe_init_checkpoint.txt"
                manifest.parent.mkdir(parents=True, exist_ok=True)
                manifest.write_text(
                    "\n".join(
                        [
                            f"checkpoint={path}",
                            f"loaded_tensors={len(loadable)}",
                            f"skipped_new_tensors={len(skipped_new)}",
                            "skipped_prefixes=softmoe_adapter.,region_aux_head.",
                            "decoder_seg_layers_loaded=true",
                            "",
                        ]
                    )
                )
            except Exception as exc:
                self.print_to_log_file(f"WARNING: failed to write SoftMoE init manifest: {exc}")

    def initialize(self):
        was_initialized_before = self.was_initialized
        super().initialize()
        if (
            not was_initialized_before
            and self.softmoe_init_checkpoint
            and not self._softmoe_init_loaded
            and not _env_bool("SOFTMOE_DISABLE_INIT_CHECKPOINT", False)
        ):
            self._load_softmoe_compatible_pretrained(self.softmoe_init_checkpoint)

    def _ensure_moe_logger_keys(self) -> None:
        local_logger = getattr(self.logger, "local_logger", None)
        logging_dict = getattr(local_logger, "my_fantastic_logging", None)
        if isinstance(logging_dict, dict):
            for key in self.custom_logger_keys:
                logging_dict.setdefault(key, [])
            for idx in range(int(self.softmoe_num_experts)):
                logging_dict.setdefault(f"moe/gate_mean_{idx}", [])
                logging_dict.setdefault(f"moe/gate_std_{idx}", [])

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        super().load_checkpoint(filename_or_checkpoint)
        self._ensure_moe_logger_keys()

    def _softmoe_adapter(self) -> BottleneckSoftMoEAdapter:
        mod = self._network_module()
        if not hasattr(mod, "softmoe_adapter"):
            raise RuntimeError("Network is missing softmoe_adapter")
        return mod.softmoe_adapter

    def _moe_stats(self) -> dict[str, float]:
        adapter = self._softmoe_adapter()
        gate = adapter.last_gate
        if gate is None:
            out = {
                "balance_loss": float("nan"),
                "gate_entropy": float("nan"),
                "gate_max_mean": float("nan"),
                "adapter_scale": float(adapter.alpha.detach().float().cpu().item()),
            }
            for idx in range(self.softmoe_num_experts):
                out[f"gate_mean_{idx}"] = float("nan")
                out[f"gate_std_{idx}"] = float("nan")
            return out
        gate_cpu = gate.detach().float().cpu()
        entropy = -(gate_cpu * torch.log(gate_cpu.clamp_min(1e-8))).sum(dim=1)
        out = {
            "balance_loss": float(adapter.balance_loss().detach().float().cpu().item()),
            "gate_entropy": float(entropy.mean().item()),
            "gate_max_mean": float(gate_cpu.max(dim=1).values.mean().item()),
            "adapter_scale": float(adapter.alpha.detach().float().cpu().item()),
        }
        for idx in range(self.softmoe_num_experts):
            out[f"gate_mean_{idx}"] = float(gate_cpu[:, idx].mean().item())
            out[f"gate_std_{idx}"] = float(gate_cpu[:, idx].std(unbiased=False).item())
        return out

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
        balance_value = np.nan

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
            balance_value = float(balance_loss.detach().float().cpu().item())
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

        meta = []
        for key in keys:
            case_id = str(key)
            merged = dict(self.case_meta_by_case.get(case_id, {}))
            merged.update(self.case_weight_details_by_case.get(case_id, {}))
            meta.append(merged)
        cluster_415 = [str(m.get("majority_cluster", "")) in {"4", "1", "5"} for m in meta]
        et_present = [str(m.get("ET_present", "")).lower() in {"1", "true", "yes"} for m in meta]
        small_et = [str(m.get("small_ET_fold_threshold", "")).lower() in {"1", "true", "yes"} for m in meta]
        moe = self._moe_stats()
        out = {
            "loss": total_loss.detach().cpu().numpy(),
            "loss_gt": loss_gt_value,
            "loss_pseudo": loss_pseudo_value,
            "region_loss_gt": region_gt_value,
            "region_loss_pseudo": region_pseudo_value,
            "n_gt": len(gt_indices),
            "n_pseudo": len(pseudo_indices),
            "lambda_pseudo": lambda_pseudo,
            "cluster_415_frac": float(np.mean(cluster_415)) if cluster_415 else np.nan,
            "et_present_frac": float(np.mean(et_present)) if et_present else np.nan,
            "small_et_frac": float(np.mean(small_et)) if small_et else np.nan,
            "moe_balance_loss": balance_value,
            "moe_gate_entropy": moe["gate_entropy"],
            "moe_gate_max_mean": moe["gate_max_mean"],
            "moe_adapter_scale": moe["adapter_scale"],
        }
        for idx in range(self.softmoe_num_experts):
            out[f"moe_gate_mean_{idx}"] = moe[f"gate_mean_{idx}"]
            out[f"moe_gate_std_{idx}"] = moe[f"gate_std_{idx}"]
        return out

    def on_train_epoch_end(self, train_outputs: list[dict]):
        super().on_train_epoch_end(train_outputs)

        def mean_present(key: str) -> float:
            values = [float(o[key]) for o in train_outputs if key in o and np.isfinite(float(o[key]))]
            return float(np.mean(values)) if values else float("nan")

        self._ensure_moe_logger_keys()
        self.logger.log("moe/balance_loss", mean_present("moe_balance_loss"), self.current_epoch)
        self.logger.log("moe/gate_entropy", mean_present("moe_gate_entropy"), self.current_epoch)
        self.logger.log("moe/gate_max_mean", mean_present("moe_gate_max_mean"), self.current_epoch)
        self.logger.log("moe/adapter_scale", mean_present("moe_adapter_scale"), self.current_epoch)
        self.logger.log("moe/temperature", float(self.softmoe_temperature), self.current_epoch)
        self.logger.log("moe/num_experts", int(self.softmoe_num_experts), self.current_epoch)
        max_mean = 0.0
        for idx in range(self.softmoe_num_experts):
            gate_mean = mean_present(f"moe_gate_mean_{idx}")
            gate_std = mean_present(f"moe_gate_std_{idx}")
            max_mean = max(max_mean, gate_mean if math.isfinite(gate_mean) else 0.0)
            self.logger.log(f"moe/gate_mean_{idx}", gate_mean, self.current_epoch)
            self.logger.log(f"moe/gate_std_{idx}", gate_std, self.current_epoch)
        self.logger.log("moe/collapse_warning", float(max_mean >= 0.85), self.current_epoch)
        self.print_to_log_file(
            "SoftMoE gate stats: "
            f"K={self.softmoe_num_experts}, balance={mean_present('moe_balance_loss'):.6f}, "
            f"entropy={mean_present('moe_gate_entropy'):.4f}, max_gate_mean={max_mean:.4f}, "
            f"alpha={mean_present('moe_adapter_scale'):.4f}"
        )

    def on_validation_epoch_start(self):
        self._current_val_gate_rows = []
        super().on_validation_epoch_start()

    @staticmethod
    def _role_to_label_source(role: str) -> str:
        if role == "original_labeled":
            return "gt"
        if role == "pseudo_labeled_highconf":
            return "pseudo"
        return "unknown"

    def validation_step(self, batch: dict) -> dict:
        out = super().validation_step(batch)
        adapter = self._softmoe_adapter()
        gate = adapter.last_gate
        if gate is None:
            return out
        gate_cpu = gate.detach().float().cpu()
        entropy = -(gate_cpu * torch.log(gate_cpu.clamp_min(1e-8))).sum(dim=1)
        keys = [str(k) for k in batch.get("keys", [f"batch{len(self._current_val_gate_rows)}_{i}" for i in range(gate_cpu.shape[0])])]
        roles = self._roles_for_keys(keys)
        val_loss_batch = float(np.asarray(out.get("loss", np.nan), dtype=np.float64).mean())
        for row_idx, case_id in enumerate(keys):
            row = {
                "case_id": case_id,
                "fold": int(self.fold),
                "epoch": int(self.current_epoch),
                "label_source": self._role_to_label_source(roles[row_idx]),
                "gate_entropy": float(entropy[row_idx].item()),
                "dominant_expert": int(torch.argmax(gate_cpu[row_idx]).item()),
                "gate_max": float(torch.max(gate_cpu[row_idx]).item()),
                "predicted_or_val_metrics_if_available": "",
                "val_loss_batch": val_loss_batch,
            }
            for expert_idx in range(self.softmoe_num_experts):
                row[f"gate_{expert_idx}"] = float(gate_cpu[row_idx, expert_idx].item())
            self._current_val_gate_rows.append(row)
        out["moe_val_gate_entropy"] = float(entropy.mean().item())
        out["moe_val_gate_max_mean"] = float(gate_cpu.max(dim=1).values.mean().item())
        return out

    def _gate_log_dir(self) -> Path:
        path = self.softmoe_output_dir / "gate_logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_val_gate_csv(self) -> None:
        if self.local_rank != 0 or not self._current_val_gate_rows:
            return
        gate_dir = self._gate_log_dir()
        latest = gate_dir / f"fold{int(self.fold)}_K{int(self.softmoe_num_experts)}_val_case_gates.csv"
        epoch_path = gate_dir / f"fold{int(self.fold)}_K{int(self.softmoe_num_experts)}_epoch{int(self.current_epoch):04d}_val_case_gates.csv"
        fieldnames = [
            "case_id",
            "fold",
            "epoch",
            "label_source",
            *[f"gate_{idx}" for idx in range(self.softmoe_num_experts)],
            "gate_entropy",
            "dominant_expert",
            "gate_max",
            "predicted_or_val_metrics_if_available",
            "val_loss_batch",
        ]
        for path in (latest, epoch_path):
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self._current_val_gate_rows)
        self.print_to_log_file(f"Wrote SoftMoE validation gate log: {latest}")

    def on_validation_epoch_end(self, val_outputs: list[dict]):
        super().on_validation_epoch_end(val_outputs)
        if self._current_val_gate_rows:
            ent = np.asarray([float(r["gate_entropy"]) for r in self._current_val_gate_rows], dtype=np.float64)
            gmax = np.asarray([float(r["gate_max"]) for r in self._current_val_gate_rows], dtype=np.float64)
            dominant = [int(r["dominant_expert"]) for r in self._current_val_gate_rows]
            dominant_frac = max(np.bincount(dominant, minlength=self.softmoe_num_experts)) / max(len(dominant), 1)
            self.logger.log("moe/val_gate_entropy", float(np.mean(ent)), self.current_epoch)
            self.logger.log("moe/val_gate_max_mean", float(np.mean(gmax)), self.current_epoch)
            self.logger.log("moe/val_dominant_expert_frac", float(dominant_frac), self.current_epoch)
            self.logger.log("moe/val_collapse_warning", float(dominant_frac >= 0.80), self.current_epoch)
            self._write_val_gate_csv()

    def perform_actual_validation(self, save_probabilities: bool = False):
        if _env_bool("SOFTMOE_SKIP_ACTUAL_VALIDATION", False):
            self.print_to_log_file(
                "Skipping perform_actual_validation because SOFTMOE_SKIP_ACTUAL_VALIDATION=1. "
                "Use this only for smoke tests."
            )
            return None
        return super().perform_actual_validation(save_probabilities)
