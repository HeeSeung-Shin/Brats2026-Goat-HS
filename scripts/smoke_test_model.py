#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Construct the final model and optionally load/forward a synthetic patch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from final_inference import CHECKPOINT_NAME, TRAINER_NAME, build_final_network


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plans-json",
        type=Path,
        default=ROOT / "config" / "nnUNetResEncUNetMPlans_D005Compat.json",
    )
    parser.add_argument(
        "--dataset-json", type=Path, default=ROOT / "config" / "dataset.json"
    )
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--checkpoint", type=Path, help="Optional final checkpoint_best.pth.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--shape", type=int, nargs=3, default=(32, 32, 32), metavar=("D", "H", "W"))
    parser.add_argument("--construct-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    plans = json.loads(args.plans_json.read_text())
    dataset_json = json.loads(args.dataset_json.read_text())
    network, _, _ = build_final_network(plans, dataset_json, args.configuration)
    adapter = network.softmoe_adapter
    if adapter.num_experts != 4 or adapter.channels != 320:
        raise RuntimeError(
            f"Unexpected adapter: K={adapter.num_experts}, channels={adapter.channels}"
        )

    if args.checkpoint is not None:
        if args.checkpoint.name != CHECKPOINT_NAME:
            raise ValueError(f"Checkpoint must be named {CHECKPOINT_NAME}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("trainer_name") != TRAINER_NAME:
            raise RuntimeError(f"Unexpected trainer_name: {checkpoint.get('trainer_name')!r}")
        network.load_state_dict(checkpoint["network_weights"], strict=True)

    cuda_available = torch.cuda.is_available()
    device_name = "cuda" if args.device == "auto" and cuda_available else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    network.to(device).eval()
    print(
        f"PASS: constructed final model on {device}; "
        f"K={adapter.num_experts}, adapter_channels={adapter.channels}"
    )
    if args.construct_only:
        return 0

    shape = tuple(args.shape)
    if any(value < 32 or value % 16 for value in shape):
        raise ValueError("Synthetic shape dimensions must be >=32 and divisible by 16")
    inputs = torch.randn((1, 4, *shape), device=device)
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = network(inputs)
        else:
            output = network(inputs)
    output = output[0] if isinstance(output, (tuple, list)) else output
    expected_shape = (1, 4, *shape)
    if tuple(output.shape) != expected_shape or not torch.isfinite(output).all().item():
        raise RuntimeError(f"Invalid forward output: shape={tuple(output.shape)}")
    hardware = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"PASS: synthetic forward on {hardware}; output={tuple(output.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
