#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT
source "${REPO_ROOT}/config/experiment.env"

usage() { printf 'Usage: %s FOLD [--final] [--gpu-id ID] [--dry-run]\n' "$(basename "$0")"; }
[[ $# -ge 1 ]] || { usage >&2; exit 2; }
FOLD="$1"; shift
USE_BEST=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --final) USE_BEST=0; shift ;;
    --gpu-id) GPU_ID="${2:?--gpu-id requires a value}"; export GPU_ID; shift 2 ;;
    --dry-run) DRY_RUN=1; export DRY_RUN; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
die() { printf '[validate-fold %s] ERROR: %s\n' "${FOLD}" "$*" >&2; exit 1; }
log() { printf '[validate-fold %s] %s\n' "${FOLD}" "$*"; }
print_cmd() { printf '+'; printf ' %q' "$@"; printf '\n'; }
[[ "${FOLD}" =~ ^[0-4]$ ]] || die "FOLD must be 0..4"
train_bin="${VENV_DIR}/bin/nnUNetv2_train"
[[ -x "${PYTHON_BIN}" && -x "${train_bin}" ]] || die "Run scripts/bootstrap.sh first"
runtime_args=()
[[ "${ALLOW_NONEXACT}" == "1" ]] || runtime_args+=(--strict)
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/verify_environment.py" "${runtime_args[@]}"

checkpoint_name=checkpoint_best.pth
[[ "${USE_BEST}" == "1" ]] || checkpoint_name=checkpoint_final.pth
checkpoint="${RESULT_ROOT}/fold_${FOLD}/${checkpoint_name}"
[[ -f "${checkpoint}" ]] || die "Missing validation checkpoint: ${checkpoint}"
cmd=("${train_bin}" "${DATASET_ID}" "${CONFIGURATION}" "${FOLD}" -tr "${TRAINER}" -p "${PLANS}" --val -device "${DEVICE}")
[[ "${USE_BEST}" == "1" ]] && cmd+=(--val_best)
log "checkpoint=${checkpoint_name}; GPU=${GPU_ID}"
print_cmd env CUDA_VISIBLE_DEVICES="${GPU_ID}" SOFTMOE_INIT_CHECKPOINT= SOFTMOE_SKIP_ACTUAL_VALIDATION=0 "${cmd[@]}"
if [[ "${DRY_RUN}" == "1" ]]; then log "dry-run: validation was not started."; exit 0; fi
env CUDA_VISIBLE_DEVICES="${GPU_ID}" SOFTMOE_INIT_CHECKPOINT= SOFTMOE_SKIP_ACTUAL_VALIDATION=0 "${cmd[@]}"
