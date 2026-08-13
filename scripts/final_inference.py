#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Final five-fold ResEnc-M K=4 submission inference from explicit inputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_SCRIPT_DIR = SCRIPT_DIR / "data"
if str(DATA_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_SCRIPT_DIR))

from brats_regions import class_probabilities_to_labels  # noqa: E402


FOLDS = (0, 1, 2, 3, 4)
PATCH_SIZE = (128, 160, 112)
STEP_SIZE = 0.5
MIRROR_AXES = (0, 1, 2)
REGION_THRESHOLD = 0.50
CHECKPOINT_NAME = "checkpoint_best.pth"
TRAINER_NAME = "nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot"
FEATURES_PER_STAGE = (32, 64, 128, 256, 320, 320)
BLOCKS_PER_STAGE = (1, 3, 4, 6, 6, 6)


def configure_k4_environment() -> None:
    """Set final architecture controls before nnU-Net imports the trainer."""
    os.environ.update(
        {
            "SOFTMOE_NUM_EXPERTS": "4",
            "SOFTMOE_TEMPERATURE": "1.0",
            "SOFTMOE_ADAPTER_SCALE_INIT": "0.1",
            "SOFTMOE_REDUCTION": "4",
        }
    )
    os.environ.setdefault("nnUNet_compile", "false")


def stable_softmax(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=axis, keepdims=True)


def average_fold_logits_and_softmax(fold_logits: list[np.ndarray]) -> np.ndarray:
    """CPU-testable equal logit averaging followed by softmax."""
    if len(fold_logits) != len(FOLDS):
        raise ValueError(f"Expected five fold logits, got {len(fold_logits)}")
    shapes = {np.asarray(logits).shape for logits in fold_logits}
    if len(shapes) != 1:
        raise ValueError(f"Fold logit shapes differ: {sorted(shapes)}")
    return stable_softmax(np.mean(np.stack(fold_logits, axis=0), axis=0), axis=0)


def final_region_mapping(probabilities: np.ndarray) -> np.ndarray:
    """Threshold ET/TC/WT at 0.50, enforce hierarchy, and reconstruct labels."""
    return class_probabilities_to_labels(probabilities, threshold=REGION_THRESHOLD)


def validate_final_plans(plans: dict[str, Any], configuration: str) -> None:
    try:
        config = plans["configurations"][configuration]
        architecture = config["architecture"]["arch_kwargs"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Plans do not contain configuration {configuration!r}") from exc
    expected = {
        "n_stages": 6,
        "features_per_stage": list(FEATURES_PER_STAGE),
        "n_blocks_per_stage": list(BLOCKS_PER_STAGE),
    }
    for key, value in expected.items():
        if architecture.get(key) != value:
            raise ValueError(f"Plans {key}={architecture.get(key)!r}, expected {value!r}")
    if tuple(config.get("patch_size", ())) != PATCH_SIZE:
        raise ValueError(f"Plans patch_size={config.get('patch_size')!r}, expected {PATCH_SIZE}")
    if int(config.get("batch_size", -1)) != 2:
        raise ValueError(f"Plans batch_size={config.get('batch_size')!r}, expected 2")


def build_final_network(
    plans: dict[str, Any], dataset_json: dict[str, Any], configuration: str
) -> tuple[Any, Any, Any]:
    """Build the final network without requiring a checkpoint or private data."""
    configure_k4_environment()
    validate_final_plans(plans, configuration)

    from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot import (
        nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot,
    )
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration)
    trainer_class = nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot
    num_input_channels = determine_num_input_channels(
        plans_manager, configuration_manager, dataset_json
    )
    network = trainer_class.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
        enable_deep_supervision=False,
    )
    return network, plans_manager, configuration_manager


def initialize_predictor_from_explicit_inputs(
    predictor: Any,
    plans: dict[str, Any],
    dataset_json: dict[str, Any],
    configuration: str,
    checkpoints: list[Path],
) -> None:
    """Restore five explicit checkpoint_best files without an implicit result path."""
    import torch

    if len(checkpoints) != len(FOLDS):
        raise ValueError(f"Expected five checkpoints, got {len(checkpoints)}")
    if len({path.resolve() for path in checkpoints}) != len(FOLDS):
        raise ValueError("The five checkpoint paths must be distinct")
    parameters = []
    for fold, path in zip(FOLDS, checkpoints):
        if path.name != CHECKPOINT_NAME:
            raise ValueError(f"Fold {fold} must use {CHECKPOINT_NAME}, got {path.name}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing fold {fold} checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("trainer_name") != TRAINER_NAME:
            raise RuntimeError(
                f"Fold {fold} trainer={checkpoint.get('trainer_name')!r}, expected {TRAINER_NAME!r}"
            )
        init_args = checkpoint.get("init_args", {})
        checkpoint_configuration = init_args.get("configuration")
        if checkpoint_configuration != configuration:
            raise RuntimeError(
                f"Fold {fold} configuration={checkpoint_configuration!r}, expected {configuration!r}"
            )
        checkpoint_fold = init_args.get("fold")
        if checkpoint_fold is None or int(checkpoint_fold) != fold:
            raise RuntimeError(
                f"Checkpoint supplied for fold {fold} records fold={checkpoint_fold!r}"
            )
        if "network_weights" not in checkpoint:
            raise RuntimeError(f"Fold {fold} checkpoint has no network_weights")
        parameters.append(checkpoint["network_weights"])

    network, plans_manager, configuration_manager = build_final_network(
        plans, dataset_json, configuration
    )
    network.load_state_dict(parameters[0], strict=True)
    network.to(predictor.device)
    network.eval()
    predictor.plans_manager = plans_manager
    predictor.configuration_manager = configuration_manager
    predictor.list_of_parameters = parameters
    predictor.network = network
    predictor.dataset_json = dataset_json
    predictor.trainer_name = TRAINER_NAME
    predictor.allowed_mirroring_axes = MIRROR_AXES
    predictor.label_manager = plans_manager.get_label_manager(dataset_json)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run final ResEnc-M K=4 folds 0-4 with Gaussian sliding windows, "
            "8-way mirroring, equal logit averaging, and regional decoding."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plans-json", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--configuration", required=True)
    for fold in FOLDS:
        parser.add_argument(
            f"--checkpoint-fold{fold}", type=Path, required=True, metavar="PATH"
        )
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def discover_cases(input_dir: Path, dataset_json: dict[str, Any]) -> list[tuple[str, list[str]]]:
    file_ending = str(dataset_json.get("file_ending", ".nii.gz"))
    channel_names = dataset_json.get("channel_names") or dataset_json.get("modality")
    if not isinstance(channel_names, dict):
        raise ValueError("dataset.json must define channel_names or modality")
    channels = [f"{int(index):04d}" for index in sorted(channel_names, key=lambda value: int(value))]
    first_suffix = f"_{channels[0]}{file_ending}"
    cases: list[tuple[str, list[str]]] = []
    for first_path in sorted(input_dir.glob(f"*{first_suffix}")):
        case_id = first_path.name[: -len(first_suffix)]
        files = [input_dir / f"{case_id}_{channel}{file_ending}" for channel in channels]
        missing = [path for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{case_id} is missing modalities: {missing}")
        cases.append((case_id, [str(path) for path in files]))
    if not cases:
        raise RuntimeError(f"No cases ending in {first_suffix!r} were found in {input_dir}")
    return cases


def restore_probabilities_to_original_space(
    probabilities: Any, properties: dict[str, Any], predictor: Any
) -> np.ndarray:
    """Use the pinned nnU-Net resampler, then undo crop and transpose."""
    import torch

    plans = predictor.plans_manager
    configuration = predictor.configuration_manager
    spacing_transposed = [properties["spacing"][index] for index in plans.transpose_forward]
    current_spacing = (
        configuration.spacing
        if len(configuration.spacing) == len(properties["shape_after_cropping_and_before_resampling"])
        else [spacing_transposed[0], *configuration.spacing]
    )
    restored = configuration.resampling_fn_probabilities(
        probabilities,
        properties["shape_after_cropping_and_before_resampling"],
        current_spacing,
        spacing_transposed,
    )
    if isinstance(restored, torch.Tensor):
        restored = restored.cpu().numpy()
    restored = np.asarray(restored, dtype=np.float32)
    full = np.zeros((4, *properties["shape_before_cropping"]), dtype=np.float32)
    crop = tuple(
        slice(int(bounds[0]), int(bounds[1])) for bounds in properties["bbox_used_for_cropping"]
    )
    full[(slice(None), *crop)] = restored
    return full.transpose([0] + [index + 1 for index in plans.transpose_backward])


def main() -> int:
    args = build_parser().parse_args()
    configure_k4_environment()
    plans = json.loads(args.plans_json.read_text())
    dataset_json = json.loads(args.dataset_json.read_text())
    checkpoints = [getattr(args, f"checkpoint_fold{fold}") for fold in FOLDS]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=STEP_SIZE,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=args.device == "cuda",
        device=torch.device(args.device),
        verbose=args.verbose,
        verbose_preprocessing=args.verbose,
        allow_tqdm=True,
    )
    initialize_predictor_from_explicit_inputs(
        predictor, plans, dataset_json, args.configuration, checkpoints
    )
    if tuple(int(value) for value in predictor.configuration_manager.patch_size) != PATCH_SIZE:
        raise RuntimeError(
            f"Expected patch size {PATCH_SIZE}, got {tuple(predictor.configuration_manager.patch_size)}"
        )

    cases = discover_cases(args.input_dir, dataset_json)
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=args.verbose)
    writer = predictor.plans_manager.image_reader_writer_class()
    file_ending = str(dataset_json["file_ending"])

    for index, (case_id, image_files) in enumerate(cases, start=1):
        output_path = args.output_dir / f"{case_id}{file_ending}"
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(cases)}] skip existing {output_path.name}")
            continue
        print(f"[{index}/{len(cases)}] predict {case_id}", flush=True)
        data, _, properties = preprocessor.run_case(
            image_files,
            None,
            predictor.plans_manager,
            predictor.configuration_manager,
            dataset_json,
        )
        averaged_logits = predictor.predict_logits_from_preprocessed_data(
            torch.from_numpy(data)
        ).cpu().float()
        probabilities = torch.softmax(averaged_logits, dim=0)
        original_probabilities = restore_probabilities_to_original_space(
            probabilities, properties, predictor
        )
        segmentation = final_region_mapping(original_probabilities)
        writer.write_seg(segmentation, str(output_path), properties)

    print(f"Wrote final regional predictions for {len(cases)} case(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
