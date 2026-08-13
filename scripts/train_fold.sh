#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT

usage() {
  printf 'Usage: %s FOLD [--gpu-id ID] [--dry-run]\n' "$(basename "$0")"
}
[[ $# -ge 1 ]] || { usage >&2; exit 2; }
FOLD="$1"
shift
GPU_ARG=""
REQUEST_DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-id) GPU_ARG="${2:?--gpu-id requires a value}"; shift 2 ;;
    --dry-run) REQUEST_DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
source "${REPO_ROOT}/config/experiment.env"
[[ -z "${GPU_ARG}" ]] || export GPU_ID="${GPU_ARG}"
[[ "${REQUEST_DRY_RUN}" == "0" ]] || { DRY_RUN=1; export DRY_RUN; }

log() { printf '[train-fold %s] %s\n' "${FOLD}" "$*"; }
die() { printf '[train-fold %s] ERROR: %s\n' "${FOLD}" "$*" >&2; exit 1; }
print_cmd() { printf '+'; printf ' %q' "$@"; printf '\n'; }

[[ "${FOLD}" =~ ^[0-4]$ ]] || die "FOLD must be 0..4"
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
[[ "${SOFTMOE_NUM_EXPERTS}" == "4" ]] || die "Final system requires SOFTMOE_NUM_EXPERTS=4"
[[ "${DEVICE}" == "cuda" ]] || die "Final training requires CUDA"

train_bin="${VENV_DIR}/bin/nnUNetv2_train"
init_checkpoint="${D005_PRETRAINED_ROOT}/fold_${FOLD}/checkpoint_best.pth"
cmd=("${train_bin}" "${DATASET_ID}" "${CONFIGURATION}" "${FOLD}" -tr "${TRAINER}" -p "${PLANS}" -device "${DEVICE}")
fold_dir="${RESULT_ROOT}/fold_${FOLD}"

log "student=ResEnc-M K=4 dense adapter"
log "dataset=${DATASET_NAME}; expected_total=${EXPECTED_TOTAL_CASES}; expected_pseudo=${EXPECTED_PSEUDO_CASES}"
log "raw=${RAW_DATASET_DIR}"
log "preprocessed=${PREPROCESSED_DATASET_DIR}"
log "result=${fold_dir}"
log "case_weights=${CASE_WEIGHT_ROOT}/case_weights_fold${FOLD}.json"
log "K=${SOFTMOE_NUM_EXPERTS}, epochs=500, GPU=${GPU_ID}"
print_cmd env CUDA_VISIBLE_DEVICES="${GPU_ID}" SOFTMOE_INIT_CHECKPOINT="${init_checkpoint}" "${cmd[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  log "dry-run: private assets were not required and training was not started."
  exit 0
fi

[[ -x "${PYTHON_BIN}" ]] || die "Missing Python; run scripts/bootstrap.sh first"
[[ -x "${train_bin}" ]] || die "Missing nnUNetv2_train; run scripts/bootstrap.sh first"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/verify_environment.py" --strict
[[ -d "${RAW_DATASET_DIR}/imagesTr" ]] || die "Missing imagesTr"
[[ -d "${RAW_DATASET_DIR}/labelsTr" ]] || die "Missing labelsTr"
[[ -d "${PREPROCESSED_DATA_DIR}" ]] || die "Missing preprocessed 3d_fullres data"
[[ -f "${RAW_DATASET_DIR}/dataset007_case_manifest.csv" ]] || die "Missing Dataset007 manifest"
[[ -f "${PREPROCESSED_DATASET_DIR}/splits_final.json" ]] || die "Missing final splits"
[[ -f "${CASE_WEIGHT_ROOT}/case_weights_fold${FOLD}.json" ]] || die "Missing final fold case weights"
[[ -f "${PREPROCESSED_DATASET_DIR}/${PLANS}.json" ]] || die "Missing installed plans"
[[ -f "${init_checkpoint}" ]] || die "Missing fold ${FOLD} D005 ResEnc-M checkpoint_best"

if [[ -f "${fold_dir}/checkpoint_final.pth" ]]; then
  [[ "${SKIP_COMPLETED}" == "1" ]] && { log "checkpoint_final.pth exists; skipping."; exit 0; }
  die "Completed fold exists: ${fold_dir}"
fi
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
log "SOFTMOE_INIT_CHECKPOINT=${init_checkpoint:-<empty>}"
log_dir="${REPO_ROOT}/runs/logs/fold_${FOLD}"
mkdir -p "${log_dir}"
log_file="${log_dir}/train_$(date +%Y%m%d_%H%M%S).log"
env CUDA_VISIBLE_DEVICES="${GPU_ID}" SOFTMOE_INIT_CHECKPOINT="${init_checkpoint}" "${cmd[@]}" 2>&1 | tee "${log_file}"
