#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average nnU-Net checkpoint network_weights tensors.")
    parser.add_argument("--input", nargs="+", required=True, help="Input checkpoint .pth files.")
    parser.add_argument("--output", required=True, help="Output checkpoint .pth path.")
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional JSON report describing the averaged checkpoint.",
    )
    parser.add_argument(
        "--strict-extra-keys",
        action="store_true",
        help="Fail if non-network metadata differs. By default only network key/shape/dtype equality is required.",
    )
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    return str(value)


def main() -> None:
    args = parse_args()
    input_paths = [Path(p) for p in args.input]
    output_path = Path(args.output)
    if len(input_paths) < 2:
        raise SystemExit("Need at least two input checkpoints to average.")
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in input_paths]
    weights_list = [ckpt.get("network_weights") for ckpt in checkpoints]
    if any(weights is None for weights in weights_list):
        missing = [str(path) for path, weights in zip(input_paths, weights_list) if weights is None]
        raise RuntimeError(f"Missing network_weights in: {missing}")

    ref_weights = weights_list[0]
    ref_keys = list(ref_weights.keys())
    for path, weights in zip(input_paths[1:], weights_list[1:]):
        keys = list(weights.keys())
        if keys != ref_keys:
            missing = sorted(set(ref_keys) - set(keys))
            extra = sorted(set(keys) - set(ref_keys))
            raise RuntimeError(f"network_weights key mismatch for {path}: missing={missing[:10]}, extra={extra[:10]}")

    averaged: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in ref_keys:
        tensors = [weights[key] for weights in weights_list]
        ref_tensor = tensors[0]
        for path, tensor in zip(input_paths[1:], tensors[1:]):
            if tuple(tensor.shape) != tuple(ref_tensor.shape):
                raise RuntimeError(f"Shape mismatch for {key} in {path}: {tuple(tensor.shape)} != {tuple(ref_tensor.shape)}")
            if tensor.dtype != ref_tensor.dtype:
                raise RuntimeError(f"Dtype mismatch for {key} in {path}: {tensor.dtype} != {ref_tensor.dtype}")

        if torch.is_floating_point(ref_tensor):
            acc = torch.zeros_like(ref_tensor, dtype=torch.float32)
            for tensor in tensors:
                acc.add_(tensor.to(dtype=torch.float32))
            averaged[key] = (acc / len(tensors)).to(dtype=ref_tensor.dtype)
        else:
            first = ref_tensor
            if not all(torch.equal(first, tensor) for tensor in tensors[1:]):
                raise RuntimeError(f"Non-floating tensor differs across checkpoints: {key}")
            averaged[key] = first.clone()

    if args.strict_extra_keys:
        ref_meta = {k: v for k, v in checkpoints[0].items() if k != "network_weights"}
        for path, ckpt in zip(input_paths[1:], checkpoints[1:]):
            meta = {k: v for k, v in ckpt.items() if k != "network_weights"}
            if meta.keys() != ref_meta.keys():
                raise RuntimeError(f"Metadata key mismatch for {path}")

    out_ckpt = checkpoints[0].copy()
    out_ckpt["network_weights"] = averaged
    out_ckpt["current_epoch"] = max((ckpt.get("current_epoch", -1) for ckpt in checkpoints), default=-1)
    out_ckpt["optimizer_state"] = None
    out_ckpt["grad_scaler_state"] = None
    out_ckpt["_averaged_checkpoint"] = {
        "method": "arithmetic_mean_network_weights",
        "num_checkpoints": len(input_paths),
        "input_checkpoints": [str(path) for path in input_paths],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, output_path)

    report = {
        "output": str(output_path),
        "input_checkpoints": [str(path) for path in input_paths],
        "num_checkpoints": len(input_paths),
        "num_tensors": len(averaged),
        "floating_tensors": sum(1 for value in averaged.values() if torch.is_floating_point(value)),
        "nonfloating_tensors": sum(1 for value in averaged.values() if not torch.is_floating_point(value)),
        "output_size_bytes": output_path.stat().st_size,
    }
    if args.metadata_json:
        metadata_path = Path(args.metadata_json)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=json_default) + "\n")

    print(json.dumps(report, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
