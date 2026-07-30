#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
: "${nnUNet_raw:?Set nnUNet_raw before running this script}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/private_assets/pseudolabels_resencML_5fold_best}"
IMAGES_UN="${IMAGES_UN:-${nnUNet_raw}/Dataset005_Brats26_Goat_With_GroundTruth/imagesUn}"
M_PRED_DIR="${M_PRED_DIR:-${OUT_ROOT}/raw_resencM_5fold}"
L_PRED_DIR="${L_PRED_DIR:-${OUT_ROOT}/raw_resencL_5fold}"
OVERWRITE=false
MAX_CASES=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [extra fuse_resencM_L_pseudolabels.py args]

Options:
  --overwrite       Replace existing fused_labels/qc/manifests/reports outputs.
  --max_cases N     Process only the first N imagesUn cases for smoke testing.
  -h, --help

Examples:
  $(basename "$0") --max_cases 2 --overwrite
  $(basename "$0") --overwrite
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite) OVERWRITE=true; shift ;;
    --max_cases) MAX_CASES="${2:?--max_cases requires a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

cmd=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/fuse_resencM_L_pseudolabels.py"
  --imagesUn "${IMAGES_UN}"
  --m_pred_dir "${M_PRED_DIR}"
  --l_pred_dir "${L_PRED_DIR}"
  --out_root "${OUT_ROOT}"
  --mode regionwise
  --prob_spatial_permutation auto
  --et_threshold 0.50
  --tc_threshold 0.50
  --wt_threshold 0.50
  --w_et_m 0.70
  --w_tc_m 0.50
  --w_wt_m 0.40
)

if [[ "${OVERWRITE}" == "true" ]]; then
  cmd+=(--overwrite)
fi
if [[ -n "${MAX_CASES}" ]]; then
  cmd+=(--max_cases "${MAX_CASES}")
fi
cmd+=("${EXTRA_ARGS[@]}")

printf '+'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
