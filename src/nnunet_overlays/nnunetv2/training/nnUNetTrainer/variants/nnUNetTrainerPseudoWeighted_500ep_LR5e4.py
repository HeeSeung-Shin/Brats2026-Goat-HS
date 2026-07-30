from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainerPseudoWeighted_500ep_LR5e4(nnUNetTrainer):
    """Dataset006 supervised mixed fine-tuning trainer.

    This trainer keeps the default nnU-Net loss, optimizer, scheduler, and
    augmentation path. It only changes:
    - num_epochs = 500
    - initial_lr = 5e-4
    - pseudo-labeled samples are down-weighted at loss aggregation time.

    Dataset006 roles are read from dataset006_case_manifest.csv. Validation is
    not modified and should contain original_labeled cases only via splits.
    """

    pseudo_weight_mode_env = "DATASET006_PSEUDO_WEIGHT_MODE"
    pseudo_fixed_weight_env = "DATASET006_PSEUDO_FIXED_WEIGHT"
    custom_logger_keys = (
        "pseudo_lambda",
        "train_loss_gt_unweighted",
        "train_loss_pseudo_unweighted",
        "train_batch_n_gt_mean",
        "train_batch_n_pseudo_mean",
    )

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
        self.initial_lr = 5e-4
        self.pseudo_weight_mode = os.environ.get(self.pseudo_weight_mode_env, "ramp").strip().lower()
        self.pseudo_fixed_weight = float(os.environ.get(self.pseudo_fixed_weight_env, "0.5"))
        if self.pseudo_weight_mode not in {"ramp", "fixed"}:
            raise RuntimeError(
                f"{self.pseudo_weight_mode_env} must be 'ramp' or 'fixed', got {self.pseudo_weight_mode!r}"
            )
        self.role_by_case = self._load_case_roles()
        self._missing_role_warnings: set[str] = set()
        self._ensure_logger_keys()
        counts = Counter(self.role_by_case.values())
        self.print_to_log_file("Dataset006 pseudo-weighted trainer initialized")
        self.print_to_log_file(f"  num_epochs: {self.num_epochs}")
        self.print_to_log_file(f"  initial_lr: {self.initial_lr}")
        self.print_to_log_file(f"  pseudo_weight_mode: {self.pseudo_weight_mode}")
        self.print_to_log_file(f"  pseudo_fixed_weight: {self.pseudo_fixed_weight}")
        self.print_to_log_file(f"  manifest role counts: {dict(counts)}")
        self.print_to_log_file("  UnCL/DyCON unlabeled cases are not used by this supervised trainer.")

    def _ensure_logger_keys(self) -> None:
        local_logger = getattr(self.logger, "local_logger", None)
        logging_dict = getattr(local_logger, "my_fantastic_logging", None)
        if isinstance(logging_dict, dict):
            for key in self.custom_logger_keys:
                logging_dict.setdefault(key, [])

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        super().load_checkpoint(filename_or_checkpoint)
        self._ensure_logger_keys()

    def _manifest_path(self) -> Path:
        raw_root = Path(os.environ.get("nnUNet_raw", "nnUNet_data/nnUNet_raw"))
        return raw_root / self.plans_manager.dataset_name / "dataset006_case_manifest.csv"

    def _load_case_roles(self) -> dict[str, str]:
        path = self._manifest_path()
        if not path.is_file():
            raise RuntimeError(
                f"Missing Dataset006 manifest required for pseudo weighting: {path}. "
                "Run validate_dataset006_ready.py and make sure nnUNet_raw is set correctly."
            )
        roles: dict[str, str] = {}
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if "case_id" not in (reader.fieldnames or []):
                raise RuntimeError(f"{path} does not contain a case_id column")
            role_column = "role" if "role" in (reader.fieldnames or []) else "source_type"
            if role_column not in (reader.fieldnames or []):
                raise RuntimeError(f"{path} must contain either role or source_type column")
            for row in reader:
                roles[row["case_id"]] = row[role_column]
        if "pseudo_labeled_highconf" not in set(roles.values()):
            raise RuntimeError(f"{path} contains no pseudo_labeled_highconf cases")
        return roles

    def _pseudo_lambda(self) -> float:
        if self.pseudo_weight_mode == "fixed":
            return self.pseudo_fixed_weight
        epoch = int(self.current_epoch)
        if epoch < 50:
            return 0.3
        if epoch < 150:
            return 0.3 + (epoch - 50) / 100.0 * 0.4
        return 0.7

    @staticmethod
    def _index_nested_batch(value, indices: torch.Tensor):
        if isinstance(value, (list, tuple)):
            return [v.index_select(0, indices) for v in value]
        return value.index_select(0, indices)

    def _roles_for_keys(self, keys: Iterable[str]) -> list[str]:
        roles = []
        for key in keys:
            case_id = str(key)
            role = self.role_by_case.get(case_id)
            if role is None:
                role = "original_labeled"
                if case_id not in self._missing_role_warnings:
                    self._missing_role_warnings.add(case_id)
                    self.print_to_log_file(
                        f"WARNING: case {case_id} is missing from Dataset006 manifest; treating it as original_labeled."
                    )
            roles.append(role)
        return roles

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._ensure_logger_keys()
        self.print_to_log_file(f"Dataset006 pseudo lambda: {self._pseudo_lambda():.4f}")
        self.logger.log("pseudo_lambda", self._pseudo_lambda(), self.current_epoch)

    def train_step(self, batch: dict) -> dict:
        if "keys" not in batch:
            raise RuntimeError(
                "The nnU-Net batch dict does not contain 'keys'. Cannot apply case-level pseudo weighting safely."
            )
        keys = list(batch["keys"])
        roles = self._roles_for_keys(keys)
        if any(role == "dycon_unlabeled" for role in roles):
            bad = [key for key, role in zip(keys, roles) if role == "dycon_unlabeled"]
            raise RuntimeError(f"dycon_unlabeled cases appeared in supervised training batch: {bad}")

        gt_indices = [idx for idx, role in enumerate(roles) if role == "original_labeled"]
        pseudo_indices = [idx for idx, role in enumerate(roles) if role == "pseudo_labeled_highconf"]
        unknown_roles = sorted({role for role in roles if role not in {"original_labeled", "pseudo_labeled_highconf"}})
        if unknown_roles:
            raise RuntimeError(f"Unsupported Dataset006 case roles in supervised batch: {unknown_roles}")

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

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            losses = []
            if gt_indices:
                idx = torch.as_tensor(gt_indices, dtype=torch.long, device=data.device)
                output_gt = self._index_nested_batch(output, idx)
                target_gt = self._index_nested_batch(target, idx)
                loss_gt = self.loss(output_gt, target_gt)
                loss_gt_value = float(loss_gt.detach().cpu().numpy())
                losses.append(loss_gt)
            if pseudo_indices:
                idx = torch.as_tensor(pseudo_indices, dtype=torch.long, device=data.device)
                output_pseudo = self._index_nested_batch(output, idx)
                target_pseudo = self._index_nested_batch(target, idx)
                loss_pseudo = self.loss(output_pseudo, target_pseudo)
                loss_pseudo_value = float(loss_pseudo.detach().cpu().numpy())
                losses.append(lambda_pseudo * loss_pseudo)
            if not losses:
                raise RuntimeError(f"No usable supervised samples in batch. keys={keys}, roles={roles}")
            total_loss = sum(losses)

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

        return {
            "loss": total_loss.detach().cpu().numpy(),
            "loss_gt": loss_gt_value,
            "loss_pseudo": loss_pseudo_value,
            "n_gt": len(gt_indices),
            "n_pseudo": len(pseudo_indices),
            "lambda_pseudo": lambda_pseudo,
        }

    def on_train_epoch_end(self, train_outputs: list[dict]):
        super().on_train_epoch_end(train_outputs)
        self._ensure_logger_keys()

        def mean_present(key: str) -> float:
            values = [float(o[key]) for o in train_outputs if key in o and np.isfinite(float(o[key]))]
            return float(np.mean(values)) if values else float("nan")

        mean_gt = mean_present("loss_gt")
        mean_pseudo = mean_present("loss_pseudo")
        mean_n_gt = mean_present("n_gt")
        mean_n_pseudo = mean_present("n_pseudo")
        self.logger.log("train_loss_gt_unweighted", mean_gt, self.current_epoch)
        self.logger.log("train_loss_pseudo_unweighted", mean_pseudo, self.current_epoch)
        self.logger.log("train_batch_n_gt_mean", mean_n_gt, self.current_epoch)
        self.logger.log("train_batch_n_pseudo_mean", mean_n_pseudo, self.current_epoch)
        self.print_to_log_file(
            "Dataset006 batch roles: "
            f"mean_n_gt={mean_n_gt:.3f}, mean_n_pseudo={mean_n_pseudo:.3f}, "
            f"mean_loss_gt={mean_gt:.6f}, mean_loss_pseudo={mean_pseudo:.6f}"
        )
