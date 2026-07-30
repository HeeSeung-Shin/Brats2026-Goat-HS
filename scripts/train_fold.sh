#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT
source "${REPO_ROOT}/config/experiment.env"

usage() { printf 'Usage: %s FOLD [--gpu-id ID] [--dry-run]\n' "$(basename "$0")"; }
[[ $# -ge 1 ]] || { usage >&2; exit 2; }
FOLD="$1"; shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-id) GPU_ID="${2:?--gpu-id requires a value}"; export GPU_ID; shift 2 ;;
    --dry-run) DRY_RUN=1; export DRY_RUN; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

log() { printf '[train-fold %s] %s\n' "${FOLD}" "$*"; }
die() { printf '[train-fold %s] ERROR: %s\n' "${FOLD}" "$*" >&2; exit 1; }
warn() { printf '[train-fold %s] WARNING: %s\n' "${FOLD}" "$*" >&2; }
print_cmd() { printf '+'; printf ' %q' "$@"; printf '\n'; }

verify_asset() {
  local path="$1" expected="$2" label="$3"
  if [[ ! -f "${path}" ]]; then
    [[ "${ALLOW_NONEXACT}" == "1" ]] && { warn "Missing ${label}: ${path}"; return 1; }
    die "Missing ${label}: ${path}"
  fi
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    [[ "${ALLOW_NONEXACT}" == "1" ]] && { warn "${label} hash mismatch: ${actual}"; return 1; }
    die "${label} hash mismatch: expected ${expected}, got ${actual}"
  fi
}

[[ "${FOLD}" =~ ^[0-4]$ ]] || die "FOLD must be 0..4"
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
[[ "${SOFTMOE_NUM_EXPERTS}" == "4" || "${ALLOW_NONEXACT}" == "1" ]] || die "Exact experiment requires SOFTMOE_NUM_EXPERTS=4"
[[ "${SOFTMOE_MAX_EPOCHS}" == "500" || "${ALLOW_NONEXACT}" == "1" ]] || die "Exact experiment requires 500 epochs"
[[ "${D007_ETAWARE_DISABLE_CASE_WEIGHTS}" == "0" || "${ALLOW_NONEXACT}" == "1" ]] || die "Exact experiment requires fold case weights"
[[ "${DEVICE}" == "cuda" || "${ALLOW_NONEXACT}" == "1" ]] || die "Exact experiment requires CUDA"

train_bin="${VENV_DIR}/bin/nnUNetv2_train"
[[ -x "${PYTHON_BIN}" ]] || die "Missing Python; run scripts/bootstrap.sh first"
[[ -x "${train_bin}" ]] || die "Missing nnUNetv2_train; run scripts/bootstrap.sh first"
runtime_args=()
[[ "${ALLOW_NONEXACT}" == "1" ]] || runtime_args+=(--strict)
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/verify_environment.py" "${runtime_args[@]}"

[[ -d "${RAW_DATASET_DIR}/imagesTr" ]] || die "Missing Dataset007 imagesTr"
[[ -d "${RAW_DATASET_DIR}/labelsTr" ]] || die "Missing Dataset007 labelsTr"
[[ -d "${PREPROCESSED_DATA_DIR}" ]] || die "Missing preprocessed 3d_fullres data"

verify_asset "${REPO_ROOT}/config/dataset.json" "${DATASET_JSON_SHA256}" "frozen dataset.json"
verify_asset "${RAW_DATASET_DIR}/dataset.json" "${DATASET_JSON_SHA256}" "installed dataset.json"
verify_asset "${PRIVATE_MANIFEST}" "${PRIVATE_MANIFEST_SHA256}" "private manifest"
verify_asset "${PRIVATE_SPLITS}" "${PRIVATE_SPLITS_SHA256}" "private splits"
verify_asset "${RAW_DATASET_DIR}/dataset007_case_manifest.csv" "${PRIVATE_MANIFEST_SHA256}" "installed manifest"
verify_asset "${PREPROCESSED_DATASET_DIR}/splits_final.json" "${PRIVATE_SPLITS_SHA256}" "installed splits"
verify_asset "${PREPROCESSED_DATASET_DIR}/${PLANS}.json" "${PLANS_SHA256}" "installed plans"
verify_asset "${CASE_META_FILE}" "${CASE_META_SHA256}" "case metadata"

weight_var="CASE_WEIGHT_FOLD${FOLD}_SHA256"
verify_asset "${CASE_WEIGHT_ROOT}/case_weights_fold${FOLD}.json" "${!weight_var}" "fold ${FOLD} case weights"
checkpoint_var="D005_FOLD${FOLD}_SHA256"
init_checkpoint="${D005_PRETRAINED_ROOT}/fold_${FOLD}/checkpoint_best.pth"
verify_asset "${init_checkpoint}" "${!checkpoint_var}" "fold ${FOLD} D005 ResEnc-M checkpoint_best"

fold_dir="${RESULT_ROOT}/fold_${FOLD}"
if [[ -f "${fold_dir}/checkpoint_final.pth" ]]; then
  [[ "${SKIP_COMPLETED}" == "1" ]] && { log "checkpoint_final.pth exists; skipping."; exit 0; }
  die "Completed fold exists: ${fold_dir}"
fi

cmd=("${train_bin}" "${DATASET_ID}" "${CONFIGURATION}" "${FOLD}" -tr "${TRAINER}" -p "${PLANS}" -device "${DEVICE}")
mode="fresh from fold-matched D005 checkpoint_best"
if [[ -f "${fold_dir}/checkpoint_latest.pth" ]]; then
  [[ "${CONTINUE_IF_POSSIBLE}" == "1" ]] || die "checkpoint_latest.pth exists but resume is disabled"
  cmd+=(--c)
  init_checkpoint=""
  mode="resume checkpoint_latest.pth"
elif [[ -d "${fold_dir}" && -n "$(find "${fold_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  die "Non-empty fold directory without checkpoint_latest.pth: ${fold_dir}"
fi

log "mode: ${mode}"
log "result: ${fold_dir}"
log "K=${SOFTMOE_NUM_EXPERTS}, epochs=${SOFTMOE_MAX_EPOCHS}, GPU=${GPU_ID}"
log "SOFTMOE_INIT_CHECKPOINT=${init_checkpoint:-<empty>}"
print_cmd env CUDA_VISIBLE_DEVICES="${GPU_ID}" SOFTMOE_INIT_CHECKPOINT="${init_checkpoint}" SOFTMOE_SKIP_ACTUAL_VALIDATION=1 "${cmd[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  log "dry-run: training was not started."
  exit 0
fi
log_dir="${REPO_ROOT}/runs/logs/fold_${FOLD}"
mkdir -p "${log_dir}"
log_file="${log_dir}/train_$(date +%Y%m%d_%H%M%S).log"
env CUDA_VISIBLE_DEVICES="${GPU_ID}" SOFTMOE_INIT_CHECKPOINT="${init_checkpoint}" SOFTMOE_SKIP_ACTUAL_VALIDATION=1 "${cmd[@]}" 2>&1 | tee "${log_file}"
