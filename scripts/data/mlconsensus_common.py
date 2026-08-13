#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

try:
    from scipy import ndimage
except Exception:  # pragma: no cover - surfaced by scripts that need filtering
    ndimage = None


FILE_ENDING = ".nii.gz"

REPO_ROOT = Path(__file__).resolve().parents[2]
NNUNET_RAW_ROOT = Path(os.environ.get("nnUNet_raw", str(REPO_ROOT / "private_assets" / "nnUNet_data" / "nnUNet_raw")))
NNUNET_PREPROCESSED_ROOT = Path(os.environ.get("nnUNet_preprocessed", str(REPO_ROOT / "private_assets" / "nnUNet_data" / "nnUNet_preprocessed")))
NNUNET_RESULTS_ROOT = Path(os.environ.get("nnUNet_results", str(REPO_ROOT / "private_assets" / "nnUNet_data" / "nnUNet_results")))

DATASET005_ROOT = Path(os.environ.get("DATASET005_ROOT", str(NNUNET_RAW_ROOT / "Dataset005_Brats26_Goat_With_GroundTruth")))
IMAGES_UN = DATASET005_ROOT / "imagesUn"
IMAGES_TR = DATASET005_ROOT / "imagesTr"
LABELS_TR = DATASET005_ROOT / "labelsTr"
DATASET005_SPLITS = Path(os.environ.get("DATASET005_SPLITS", str(NNUNET_PREPROCESSED_ROOT / "Dataset005_Brats26_Goat_With_GroundTruth" / "splits_final.json")))

RESENC_M_RESULTS = Path(os.environ.get("RESENC_M_RESULTS", str(NNUNET_RESULTS_ROOT / "Dataset005_Brats26_Goat_With_GroundTruth" / "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres")))
RESENC_L_RESULTS = Path(os.environ.get("RESENC_L_RESULTS", str(NNUNET_RESULTS_ROOT / "Dataset005_Brats26_Goat_With_GroundTruth" / "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres")))

OUT_ROOT = Path(os.environ.get("PSEUDOLABEL_ROOT", str(REPO_ROOT / "private_assets" / "pseudolabels_resencML_5fold_best")))
DATASET007_ROOT = Path(os.environ.get("DATASET007_ROOT", str(NNUNET_RAW_ROOT / "Dataset007_Brats26_Goat_MLConsensusPseudo")))
DATASET007_PREPROCESSED = Path(os.environ.get("DATASET007_PREPROCESSED", str(NNUNET_PREPROCESSED_ROOT / "Dataset007_Brats26_Goat_MLConsensusPseudo")))

EXPECTED_UNLABELED_CASES = 1138
EXPECTED_LABELED_CASES = 1351
EXPECTED_CHANNELS = ["0000", "0001", "0002", "0003"]
VALID_LABELS = {0, 1, 2, 3}


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r") as f:
        return json.load(f)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def read_case_ids(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_case_ids(path: str | Path, case_ids: list[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text("\n".join(case_ids) + ("\n" if case_ids else ""))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def case_id_from_image_name(name: str) -> str:
    if not name.endswith(FILE_ENDING):
        raise ValueError(f"Expected {FILE_ENDING}: {name}")
    stem = name[: -len(FILE_ENDING)]
    if "_" not in stem:
        raise ValueError(f"Expected nnU-Net channel suffix: {name}")
    return stem.rsplit("_", 1)[0]


def channel_from_image_name(name: str) -> str:
    return name[: -len(FILE_ENDING)].rsplit("_", 1)[1]


def case_ids_from_images(folder: str | Path) -> list[str]:
    folder = Path(folder)
    return sorted({case_id_from_image_name(p.name) for p in folder.glob(f"*{FILE_ENDING}")})


def label_case_ids(folder: str | Path) -> list[str]:
    folder = Path(folder)
    return sorted(p.name[: -len(FILE_ENDING)] for p in folder.glob(f"*{FILE_ENDING}") if "_" not in p.name[: -len(FILE_ENDING)])


def modalities_for_case(folder: str | Path, case_id: str) -> dict[str, Path]:
    folder = Path(folder)
    result: dict[str, Path] = {}
    for path in folder.glob(f"{case_id}_*{FILE_ENDING}"):
        result[channel_from_image_name(path.name)] = path
    return result


def expected_channels_from_dataset_json(dataset_json: dict[str, Any]) -> list[str]:
    channel_names = dataset_json.get("channel_names") or dataset_json.get("modality")
    if not isinstance(channel_names, dict):
        return EXPECTED_CHANNELS.copy()
    return [f"{int(k):04d}" for k in sorted(channel_names, key=lambda x: int(x))]


def valid_label_ids_from_dataset_json(dataset_json: dict[str, Any]) -> set[int]:
    labels = dataset_json.get("labels")
    if not isinstance(labels, dict):
        return VALID_LABELS.copy()
    values: set[int] = set()
    for value in labels.values():
        if isinstance(value, list):
            values.update(int(v) for v in value)
        else:
            values.add(int(value))
    return values


def load_nifti(path: str | Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    return img, np.asanyarray(img.dataobj)


def save_nifti_like(data: np.ndarray, reference_img: nib.Nifti1Image, path: str | Path, dtype: str | np.dtype = np.uint8) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    header = reference_img.header.copy()
    header.set_data_dtype(dtype)
    out = nib.Nifti1Image(np.asarray(data, dtype=dtype), reference_img.affine, header)
    nib.save(out, str(path))


def ref_modality_path(images_un: str | Path, case_id: str) -> Path:
    return Path(images_un) / f"{case_id}_0000{FILE_ENDING}"


def align_spatial_array_to_ref(array: np.ndarray, ref_shape: tuple[int, int, int], mode: str = "auto") -> tuple[np.ndarray, str]:
    spatial = tuple(int(i) for i in array.shape)
    ref_shape = tuple(int(i) for i in ref_shape)
    if spatial == ref_shape:
        return array, "identity"
    if mode == "auto" and spatial == ref_shape[::-1]:
        return np.transpose(array, (2, 1, 0)), "transpose_210"
    raise ValueError(f"Spatial shape {spatial} does not match reference shape {ref_shape}")


def align_probability_to_ref(probabilities: np.ndarray, ref_shape: tuple[int, int, int], mode: str = "auto") -> tuple[np.ndarray, str]:
    arr = np.asarray(probabilities, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D probability array, got shape {arr.shape}")
    if arr.shape[0] == 4:
        class_first = arr
    elif arr.shape[-1] == 4:
        class_first = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"Cannot infer class axis for probability shape {arr.shape}")
    spatial = tuple(int(i) for i in class_first.shape[1:])
    ref_shape = tuple(int(i) for i in ref_shape)
    if spatial == ref_shape:
        return class_first.astype(np.float32, copy=False), "identity"
    if mode == "auto" and spatial == ref_shape[::-1]:
        return np.transpose(class_first, (0, 3, 2, 1)).astype(np.float32, copy=False), "transpose_210"
    raise ValueError(f"Probability spatial shape {spatial} does not match reference shape {ref_shape}")


def load_probabilities_npz(path: str | Path, ref_shape: tuple[int, int, int], mode: str = "auto") -> tuple[np.ndarray, str, str]:
    path = Path(path)
    with np.load(path) as data:
        chosen_key = None
        selected = None
        for key in ("probabilities", "softmax", "arr_0"):
            if key in data:
                chosen_key = key
                selected = np.asarray(data[key])
                break
        if selected is None:
            keys = list(data.keys())
            for key in keys:
                candidate = np.asarray(data[key])
                if candidate.ndim == 4 and (candidate.shape[0] == 4 or candidate.shape[-1] == 4):
                    chosen_key = key
                    selected = candidate
                    break
            if selected is None:
                raise ValueError(f"{path} has no probability-like key. keys={keys}")
    aligned, transform = align_probability_to_ref(selected, ref_shape, mode=mode)
    if aligned.shape[0] != 4:
        raise ValueError(f"Expected 4 probability channels, got shape {aligned.shape}")
    if not np.isfinite(aligned).all():
        raise ValueError(f"{path} contains NaN or Inf")
    return aligned, str(chosen_key), transform


def load_label_aligned(path: str | Path, ref_shape: tuple[int, int, int], mode: str = "auto") -> tuple[np.ndarray, str]:
    _, data = load_nifti(path)
    arr = np.asarray(data)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"{path} is not a 3D label map. shape={arr.shape}")
    aligned, transform = align_spatial_array_to_ref(arr, ref_shape, mode=mode)
    return np.rint(aligned).astype(np.uint8, copy=False), transform


def labels_to_regions(label: np.ndarray) -> dict[str, np.ndarray]:
    label = np.asarray(label)
    return {
        "ET": label == 3,
        "TC": np.logical_or(label == 1, label == 3),
        "WT": label > 0,
    }


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    vol_a = int(a.sum())
    vol_b = int(b.sum())
    if vol_a == 0 and vol_b == 0:
        # Agreement between two empty teacher masks is complete. This is
        # intentionally distinct from evaluate_original_gt.py, where
        # both-empty case/region pairs are NaN and excluded from means.
        return 1.0
    if vol_a == 0 or vol_b == 0:
        return 0.0
    return float(2.0 * np.logical_and(a, b).sum() / (vol_a + vol_b))


def region_volumes(label: np.ndarray, voxel_volume_mm3: float = 1.0) -> dict[str, Any]:
    regions = labels_to_regions(label)
    result: dict[str, Any] = {}
    for region, mask in regions.items():
        voxels = int(mask.sum())
        result[f"volume_{region}_voxels"] = voxels
        result[f"volume_{region}_mm3"] = float(voxels * voxel_volume_mm3)
    return result


def finite_mean(values: np.ndarray) -> float:
    arr = np.asarray(values)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def finite_fraction(mask: np.ndarray) -> float:
    arr = np.asarray(mask)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr.astype(np.float32)))


def probability_quality(probabilities: np.ndarray, foreground: np.ndarray, highconf_threshold: float = 0.75) -> dict[str, float]:
    probs = np.asarray(probabilities, dtype=np.float32)
    sums = probs.sum(axis=0, keepdims=True)
    probs = probs / np.clip(sums, 1e-8, None)
    clipped = np.clip(probs, 1e-8, 1.0)
    sorted_probs = np.sort(probs, axis=0)
    top1 = sorted_probs[-1]
    top2 = sorted_probs[-2]
    margin = top1 - top2
    entropy = -(clipped * np.log(clipped)).sum(axis=0) / math.log(4)
    fg = np.asarray(foreground, dtype=bool)
    return {
        "probability_sum_max_abs_error": float(np.max(np.abs(np.asarray(probabilities).sum(axis=0) - 1.0))),
        "mean_fg_confidence": finite_mean(top1[fg]),
        "mean_fg_normalized_entropy": finite_mean(entropy[fg]),
        "mean_fg_margin": finite_mean(margin[fg]),
        "highconf_fg_fraction": finite_fraction(top1[fg] >= highconf_threshold),
        "mean_all_confidence": finite_mean(top1),
        "mean_all_normalized_entropy": finite_mean(entropy),
        "mean_all_margin": finite_mean(margin),
    }


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    arr = np.asarray(probabilities, dtype=np.float32)
    denom = np.clip(arr.sum(axis=0, keepdims=True), 1e-8, None)
    return arr / denom


def copy_or_link(src: str | Path, dst: str | Path, mode: str) -> None:
    src = Path(src)
    dst = Path(dst)
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def remove_small_components(
    mask: np.ndarray,
    probability: np.ndarray,
    voxel_volume_mm3: float,
    min_volume_mm3: float,
    min_mean_probability: float,
    min_max_probability: float,
    connectivity: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if ndimage is None:
        raise RuntimeError("scipy.ndimage is required for component filtering")
    mask = np.asarray(mask, dtype=bool)
    probability = np.asarray(probability, dtype=np.float32)
    if connectivity == 6:
        structure = ndimage.generate_binary_structure(mask.ndim, 1)
    elif connectivity == 18:
        structure = ndimage.generate_binary_structure(mask.ndim, 2)
    else:
        structure = np.ones((3, 3, 3), dtype=bool)
    cc, count = ndimage.label(mask, structure=structure)
    kept = np.zeros_like(mask, dtype=bool)
    removed_components = 0
    removed_voxels = 0
    kept_components = 0
    for component_id in range(1, count + 1):
        component = cc == component_id
        voxels = int(component.sum())
        volume_mm3 = float(voxels * voxel_volume_mm3)
        values = probability[component]
        mean_prob = float(values.mean()) if values.size else 0.0
        max_prob = float(values.max()) if values.size else 0.0
        remove = volume_mm3 < min_volume_mm3 and mean_prob < min_mean_probability and max_prob < min_max_probability
        if remove:
            removed_components += 1
            removed_voxels += voxels
        else:
            kept[component] = True
            kept_components += 1
    return kept, {
        "component_count": int(count),
        "kept_components": int(kept_components),
        "removed_components": int(removed_components),
        "removed_voxels": int(removed_voxels),
        "removed_mm3": float(removed_voxels * voxel_volume_mm3),
    }


def summary_stats(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }
