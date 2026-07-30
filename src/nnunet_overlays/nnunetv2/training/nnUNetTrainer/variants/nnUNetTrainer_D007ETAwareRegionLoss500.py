from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from torch import autocast

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainerPseudoWeighted_500ep_LR5e4 import (
    nnUNetTrainerPseudoWeighted_500ep_LR5e4,
)
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainer_D007ETAwareRegionLoss500(nnUNetTrainerPseudoWeighted_500ep_LR5e4):
    """Dataset007 ET/TC-aware region-loss fine-tuning trainer.

    This trainer intentionally keeps the D007 standard FT500 optimizer, LR,
    PolyLR scheduler, and pseudo-label curriculum. It adds:
    - oversample_foreground_percent = 0.5
    - ET/TC/WT soft region Dice auxiliary loss
    - optional fold-specific case sampling probabilities from
      analysis/proxy_clusters_mean5fold_robust/case_weights_fold{fold}.json

    Clusters are latent proxy subgroups only. They are not real site/scanner
    labels and are used only as a mild sampling cap.
    """

    lambda_et = 0.2
    lambda_tc = 0.2
    lambda_wt = 0.1
    case_weight_root_env = "D007_ETAWARE_CASE_WEIGHT_ROOT"
    disable_case_weights_env = "D007_ETAWARE_DISABLE_CASE_WEIGHTS"
    custom_logger_keys = nnUNetTrainerPseudoWeighted_500ep_LR5e4.custom_logger_keys + (
        "train_region_loss_gt",
        "train_region_loss_pseudo",
        "train_batch_weighted_cluster_415_frac",
        "train_batch_weighted_et_present_frac",
        "train_batch_weighted_small_et_frac",
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
        self.num_epochs = 500
        self.initial_lr = 5e-4
        self.oversample_foreground_percent = 0.5
        self.region_loss_weights = {"ET": self.lambda_et, "TC": self.lambda_tc, "WT": self.lambda_wt}
        self.case_meta_by_case = self._load_case_meta()
        self.case_weight_details_by_case: dict[str, dict[str, Any]] = {}
        self.case_weights = self._load_case_weights()
        self._ensure_logger_keys()
        self.print_to_log_file("D007 ETAwareRegionLoss500 initialized")
        self.print_to_log_file(f"  oversample_foreground_percent: {self.oversample_foreground_percent}")
        self.print_to_log_file(f"  region_loss_weights: {self.region_loss_weights}")
        self.print_to_log_file(f"  case_weights_loaded: {len(self.case_weights)}")
        self.print_to_log_file("  ignore_label handling: Dataset007 has no ignore label; if present, ignored voxels are masked.")

    def _manifest_path(self) -> Path:
        raw_root = Path(os.environ.get("nnUNet_raw", "nnUNet_data/nnUNet_raw"))
        dataset_root = raw_root / self.plans_manager.dataset_name
        dataset007_manifest = dataset_root / "dataset007_case_manifest.csv"
        if dataset007_manifest.is_file():
            return dataset007_manifest
        return dataset_root / "dataset006_case_manifest.csv"

    def _load_case_roles(self) -> dict[str, str]:
        path = self._manifest_path()
        roles: dict[str, str] = {}
        role_map = {
            "original_labeled": "original_labeled",
            "pseudo_labeled_highconf": "pseudo_labeled_highconf",
            "pseudo_labeled_ml_strict": "pseudo_labeled_highconf",
            "pseudo_labeled_ml_relaxed": "pseudo_labeled_highconf",
            "excluded": "dycon_unlabeled",
            "ml_unlabeled_remainder": "dycon_unlabeled",
        }
        if path.is_file():
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                role_column = "role" if "role" in (reader.fieldnames or []) else "source_type"
                for row in reader:
                    roles[row["case_id"]] = role_map.get(row.get(role_column, ""), row.get(role_column, ""))
        if "pseudo_labeled_highconf" in set(roles.values()):
            return roles

        raw_root = Path(os.environ.get("nnUNet_raw", "nnUNet_data/nnUNet_raw"))
        d005_labels = raw_root / "Dataset005_Brats26_Goat_With_GroundTruth" / "labelsTr"
        d007_labels = raw_root / self.plans_manager.dataset_name / "labelsTr"
        d005_cases = {p.name[:-7] for p in d005_labels.glob("*.nii.gz")}
        for label in d007_labels.glob("*.nii.gz"):
            case_id = label.name[:-7]
            roles[case_id] = "original_labeled" if case_id in d005_cases else "pseudo_labeled_highconf"
        if "pseudo_labeled_highconf" not in set(roles.values()):
            raise RuntimeError("Unable to infer Dataset007 pseudo-labeled cases for pseudo weighting")
        return roles

    def _analysis_root(self) -> Path:
        return Path(os.environ.get(self.case_weight_root_env, "private_assets/case_weights"))

    def _load_case_meta(self) -> dict[str, dict[str, Any]]:
        path = self._analysis_root() / "d007_case_region_cluster_meta.csv"
        if not path.is_file():
            return {}
        with path.open(newline="") as f:
            return {row["case_id"]: row for row in csv.DictReader(f)}

    def _load_case_weights(self) -> dict[str, float]:
        if os.environ.get(self.disable_case_weights_env, "0").strip().lower() in {"1", "true", "yes"}:
            self.print_to_log_file("D007 ETAware case weights disabled by environment variable.")
            return {}
        path = self._analysis_root() / f"case_weights_fold{int(self.fold)}.json"
        if not path.is_file():
            self.print_to_log_file(f"WARNING: missing case weight file; using uniform case sampling: {path}")
            return {}
        with path.open() as f:
            data = json.load(f)
        weights = data.get("case_weights", data)
        self.case_weight_details_by_case = {
            str(k): v for k, v in data.get("case_details", {}).items() if isinstance(v, dict)
        }
        return {str(k): float(v) for k, v in weights.items()}

    @staticmethod
    def _ds_weights(n_outputs: int) -> tuple[float, ...]:
        weights = np.array([1 / (2**i) for i in range(n_outputs)], dtype=np.float32)
        weights[-1] = 0.0
        weights = weights / weights.sum()
        return tuple(float(w) for w in weights)

    @staticmethod
    def _region_dice_loss_single(logits: torch.Tensor, target: torch.Tensor, ignore_label: int | None = None) -> torch.Tensor:
        if target.ndim == logits.ndim:
            target = target[:, 0]
        target = target.long()
        valid_mask = torch.ones_like(target, dtype=torch.bool)
        if ignore_label is not None:
            valid_mask = target != int(ignore_label)
            target = target.clone()
            target[~valid_mask] = 0

        prob = F.softmax(logits.float(), dim=1)
        p_et = prob[:, 3]
        p_tc = prob[:, 1] + prob[:, 3]
        p_wt = prob[:, 1] + prob[:, 2] + prob[:, 3]
        y_et = target == 3
        y_tc = (target == 1) | (target == 3)
        y_wt = target > 0

        losses = []
        smooth = 1e-5
        for p, y, weight in ((p_et, y_et, 0.2), (p_tc, y_tc, 0.2), (p_wt, y_wt, 0.1)):
            p = p * valid_mask
            y_f = y.float() * valid_mask
            reduce_axes = tuple(range(1, p.ndim))
            inter = (p * y_f).sum(dim=reduce_axes)
            denom = p.sum(dim=reduce_axes) + y_f.sum(dim=reduce_axes)
            dice = (2 * inter + smooth) / (denom + smooth)
            losses.append(weight * (1.0 - dice).mean())
        return sum(losses)

    def _region_loss(self, output, target) -> torch.Tensor:
        if isinstance(output, (list, tuple)):
            weights = self._ds_weights(len(output))
            return sum(
                weights[i] * self._region_dice_loss_single(o, t, self.label_manager.ignore_label)
                for i, (o, t) in enumerate(zip(output, target))
                if weights[i] != 0.0
            )
        return self._region_dice_loss_single(output, target, self.label_manager.ignore_label)

    def _loss_with_region(self, output, target) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = self.loss(output, target)
        region = self._region_loss(output, target)
        return base + region, base, region

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

        meta = []
        for key in keys:
            case_id = str(key)
            merged = dict(self.case_meta_by_case.get(case_id, {}))
            merged.update(self.case_weight_details_by_case.get(case_id, {}))
            meta.append(merged)
        cluster_415 = [
            str(m.get("majority_cluster", "")) in {"4", "1", "5"} for m in meta
        ]
        et_present = [str(m.get("ET_present", "")).lower() in {"1", "true", "yes"} for m in meta]
        small_et = [str(m.get("small_ET_fold_threshold", "")).lower() in {"1", "true", "yes"} for m in meta]

        return {
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
        }

    def on_train_epoch_end(self, train_outputs: list[dict]):
        super().on_train_epoch_end(train_outputs)

        def mean_present(key: str) -> float:
            values = [float(o[key]) for o in train_outputs if key in o and np.isfinite(float(o[key]))]
            return float(np.mean(values)) if values else float("nan")

        self.logger.log("train_region_loss_gt", mean_present("region_loss_gt"), self.current_epoch)
        self.logger.log("train_region_loss_pseudo", mean_present("region_loss_pseudo"), self.current_epoch)
        self.logger.log("train_batch_weighted_cluster_415_frac", mean_present("cluster_415_frac"), self.current_epoch)
        self.logger.log("train_batch_weighted_et_present_frac", mean_present("et_present_frac"), self.current_epoch)
        self.logger.log("train_batch_weighted_small_et_frac", mean_present("small_et_frac"), self.current_epoch)
        self.print_to_log_file(
            "D007 ETAware batch proxy stats: "
            f"cluster_415_frac={mean_present('cluster_415_frac'):.3f}, "
            f"et_present_frac={mean_present('et_present_frac'):.3f}, "
            f"small_et_frac={mean_present('small_et_frac'):.3f}, "
            f"region_gt={mean_present('region_loss_gt'):.6f}, "
            f"region_pseudo={mean_present('region_loss_pseudo'):.6f}"
        )

    def _sampling_probabilities_for_dataset(self, dataset) -> np.ndarray | None:
        if not self.case_weights:
            return None
        weights = np.asarray([self.case_weights.get(str(k), 1.0) for k in dataset.identifiers], dtype=np.float64)
        weights = np.clip(weights, 1e-8, None)
        weights = weights / weights.sum()
        selected = [str(k) for k in dataset.identifiers if self.case_weights.get(str(k), 1.0) != 1.0]
        self.print_to_log_file(
            f"D007 ETAware weighted sampling active for fold {self.fold}: "
            f"n_train={len(dataset.identifiers)}, n_nonunit_weights={len(selected)}, "
            f"min_p={weights.min():.6e}, max_p={weights.max():.6e}"
        )
        return weights

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = nnUNetDataLoader(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=self._sampling_probabilities_for_dataset(dataset_tr),
            pad_sides=None,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )
        dl_val = nnUNetDataLoader(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val
