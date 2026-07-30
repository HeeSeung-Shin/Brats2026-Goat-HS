#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${nnUNet_raw:?Set nnUNet_raw before running this script}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed before running this script}"
: "${nnUNet_results:?Set nnUNet_results before running this script}"

NNUNET_PREDICT="${NNUNET_PREDICT:-$(command -v nnUNetv2_predict || true)}"
IMAGES_UN="${IMAGES_UN:-${nnUNet_raw}/Dataset005_Brats26_Goat_With_GroundTruth/imagesUn}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/private_assets/pseudolabels_resencML_5fold_best}"
GPU_ID="2"
EXPECTED_CASES=1138
EXECUTE=false
RUN_M=false
RUN_L=false
OVERWRITE=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Dry-run by default. Use --execute to run nnUNetv2_predict.

Options:
  --gpu-id ID       Default: ${GPU_ID}
  --run-m          Run/print ResEnc-M 5fold ensemble only
  --run-l          Run/print ResEnc-L 5fold ensemble only
  --overwrite      Allow replacing existing raw_resencM_5fold/raw_resencL_5fold output
  --execute        Actually execute prediction commands
  -h, --help

If neither --run-m nor --run-l is provided, both commands are printed/run
sequentially: M first, then L.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-id) GPU_ID="${2:?--gpu-id requires a value}"; shift 2 ;;
    --run-m) RUN_M=true; shift ;;
    --run-l) RUN_L=true; shift ;;
    --overwrite) OVERWRITE=true; shift ;;
    --execute) EXECUTE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${RUN_M}" == "false" && "${RUN_L}" == "false" ]]; then
  RUN_M=true
  RUN_L=true
fi

if [[ ! -x "${NNUNET_PREDICT}" ]]; then
  echo "nnUNetv2_predict not found or not executable: ${NNUNET_PREDICT}" >&2
  exit 1
fi

prepare_output_dir() {
  local output_dir="$1"
  if [[ -d "${output_dir}" && -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    if [[ "${OVERWRITE}" != "true" ]]; then
      echo "Output already exists and is not empty: ${output_dir}" >&2
      echo "Use --overwrite only if you intentionally want to replace this model's raw predictions." >&2
      exit 1
    fi
    rm -rf "${output_dir}"
  fi
  mkdir -p "${output_dir}"
}

print_cmd() {
  printf '+ CUDA_VISIBLE_DEVICES=%q' "${GPU_ID}"
  printf ' %q' "$@"
  printf '\n'
}

check_prediction_counts() {
  local output_dir="$1"
  local label_count
  local npz_count
  label_count="$(find "${output_dir}" -maxdepth 1 -name '*.nii.gz' | wc -l)"
  npz_count="$(find "${output_dir}" -maxdepth 1 -name '*.npz' | wc -l)"
  echo "Prediction count check: ${output_dir}"
  echo "  labels: ${label_count}/${EXPECTED_CASES}"
  echo "  npz:    ${npz_count}/${EXPECTED_CASES}"
  if [[ "${label_count}" -ne "${EXPECTED_CASES}" || "${npz_count}" -ne "${EXPECTED_CASES}" ]]; then
    echo "Prediction count check failed for ${output_dir}" >&2
    exit 1
  fi
}

run_predict() {
  local plan="$1"
  local output_dir="$2"
  local -a cmd=(
    "${NNUNET_PREDICT}"
    -i "${IMAGES_UN}"
    -o "${output_dir}"
    -d 005
    -c 3d_fullres
    -tr nnUNetTrainer
    -p "${plan}"
    -f 0 1 2 3 4
    -chk checkpoint_best.pth
    --save_probabilities
    -device cuda
  )

  print_cmd "${cmd[@]}"
  if [[ "${EXECUTE}" != "true" ]]; then
    return
  fi

  prepare_output_dir "${output_dir}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}"
  check_prediction_counts "${output_dir}"
}

echo "Mode: $([[ "${EXECUTE}" == "true" ]] && echo execute || echo dry-run)"
echo "GPU_ID: ${GPU_ID}"
echo "nnUNetv2_predict: ${NNUNET_PREDICT}"
echo "OUT_ROOT: ${OUT_ROOT}"

if [[ "${RUN_M}" == "true" ]]; then
  run_predict "nnUNetResEncUNetMPlans" "${OUT_ROOT}/raw_resencM_5fold"
fi
if [[ "${RUN_L}" == "true" ]]; then
  run_predict "nnUNetResEncUNetLPlans" "${OUT_ROOT}/raw_resencL_5fold"
fi

if [[ "${EXECUTE}" != "true" ]]; then
  echo "Dry-run only. Add --execute to run the long inference."
fi
