from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainer_D007ETAwareRegionLoss500 import (
    nnUNetTrainer_D007ETAwareRegionLoss500,
)


class ETTCWTRegionAuxHead(nn.Module):
    """1x1x1 trainable ET/TC/WT auxiliary head fed by main segmentation logits.

    The Conv3d is intentionally registered under ``seg_layers`` so nnU-Net's
    official ``-pretrained_weights`` loader skips it, just like the main
    segmentation heads. This allows D005 checkpoints without aux-head weights
    to initialize the backbone safely without modifying nnU-Net core files.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 3):
        super().__init__()
        self.seg_layers = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.seg_layers(logits)


class nnUNetTrainer_D007ETTCRegionAuxHead500(nnUNetTrainer_D007ETAwareRegionLoss500):
    """Dataset007 ResEnc-M FT500 with trainable ET/TC/WT region auxiliary head.

    V0 design:
    - main nnU-Net output remains the only inference output
    - auxiliary head consumes highest-resolution main logits only
    - auxiliary target is multi-label ET/TC/WT membership
    - pseudo-label curriculum, optimizer, LR, PolyLR and D007 GT/pseudo role
      handling are inherited from the FT500 pseudo-weighted trainer path
    """

    lambda_et = 0.2
    lambda_tc = 0.2
    lambda_wt = 0.1
    custom_logger_keys = nnUNetTrainer_D007ETAwareRegionLoss500.custom_logger_keys + (
        "train_aux_region_loss_gt",
        "train_aux_region_loss_pseudo",
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
        network = nnUNetTrainer_D007ETAwareRegionLoss500.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        if num_output_channels != 4:
            raise RuntimeError(
                "D007 ET/TC/WT aux head expects BraTS 4-class main logits "
                f"(background/NCR/ED/ET), got {num_output_channels}."
            )
        network.region_aux_head = ETTCWTRegionAuxHead(in_channels=num_output_channels, out_channels=3)
        return network

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.region_loss_weights = {"ET": self.lambda_et, "TC": self.lambda_tc, "WT": self.lambda_wt}
        self._ensure_logger_keys()
        self.print_to_log_file("D007 ETTCRegionAuxHead500 initialized")
        self.print_to_log_file("  aux_head: network.region_aux_head.seg_layers = Conv3d(4, 3, kernel_size=1)")
        self.print_to_log_file("  aux_target: sigmoid multi-label ET/TC/WT from highest-resolution output only")
        self.print_to_log_file("  inference: main segmentation head only; aux output is training-only")
        self.print_to_log_file(
            "  pretrained loading: aux head Conv is under '.seg_layers.' so nnU-Net -pretrained_weights skips it"
        )

    def _network_module(self) -> nn.Module:
        mod = self.network
        if isinstance(mod, torch.nn.parallel.DistributedDataParallel):
            mod = mod.module
        if hasattr(mod, "_orig_mod"):
            mod = mod._orig_mod
        return mod

    def _region_aux_head(self) -> nn.Module:
        mod = self._network_module()
        if not hasattr(mod, "region_aux_head"):
            raise RuntimeError("Network is missing region_aux_head. Did build_network_architecture run?")
        return mod.region_aux_head

    @staticmethod
    def _highest_resolution(value):
        return value[0] if isinstance(value, (list, tuple)) else value

    @staticmethod
    def _region_targets(target: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        if target.ndim >= 5:
            target = target[:, 0]
        target = target.long()
        if valid_mask is not None:
            target = target.clone()
            target[~valid_mask] = 0
        y_et = target == 3
        y_tc = (target == 1) | (target == 3)
        y_wt = target > 0
        return torch.stack((y_et, y_tc, y_wt), dim=1).float()

    def _aux_region_loss(self, output, target) -> torch.Tensor:
        main_logits = self._highest_resolution(output)
        main_target = self._highest_resolution(target)
        if main_target.ndim >= 5:
            main_target_3d = main_target[:, 0]
        else:
            main_target_3d = main_target

        valid_mask = torch.ones_like(main_target_3d, dtype=torch.bool)
        if self.label_manager.ignore_label is not None:
            valid_mask = main_target_3d != int(self.label_manager.ignore_label)

        aux_logits = self._region_aux_head()(main_logits)
        aux_targets = self._region_targets(main_target, valid_mask)
        valid_f = valid_mask[:, None].float()

        weights = torch.as_tensor(
            [self.lambda_et, self.lambda_tc, self.lambda_wt],
            dtype=torch.float32,
            device=aux_logits.device,
        )

        bce = F.binary_cross_entropy_with_logits(aux_logits.float(), aux_targets, reduction="none")
        valid_voxels = valid_f.sum(dim=tuple(range(2, valid_f.ndim))).clamp_min(1.0)
        bce_per_channel = (bce * valid_f).sum(dim=tuple(range(2, bce.ndim))) / valid_voxels
        bce_per_channel = bce_per_channel.mean(dim=0)

        probs = torch.sigmoid(aux_logits.float()) * valid_f
        targets = aux_targets * valid_f
        reduce_axes = tuple(range(2, probs.ndim))
        intersection = (probs * targets).sum(dim=reduce_axes)
        denominator = probs.sum(dim=reduce_axes) + targets.sum(dim=reduce_axes)
        dice_loss_per_channel = (1.0 - (2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean(dim=0)

        channel_losses = bce_per_channel + dice_loss_per_channel
        return torch.sum(weights * channel_losses)

    def _loss_with_region(self, output, target) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = self.loss(output, target)
        aux = self._aux_region_loss(output, target)
        return base + aux, base, aux

    def on_train_epoch_end(self, train_outputs: list[dict]):
        super().on_train_epoch_end(train_outputs)

        def mean_present(key: str) -> float:
            values = [float(o[key]) for o in train_outputs if key in o and np.isfinite(float(o[key]))]
            return float(np.mean(values)) if values else float("nan")

        self.logger.log("train_aux_region_loss_gt", mean_present("region_loss_gt"), self.current_epoch)
        self.logger.log("train_aux_region_loss_pseudo", mean_present("region_loss_pseudo"), self.current_epoch)
