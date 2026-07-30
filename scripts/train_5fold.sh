#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT
# shellcheck source=../config/experiment.env
source "${REPO_ROOT}/config/experiment.env"

EXECUTE=0
usage() {
  cat <<EOF
Usage: $(basename "$0") [--folds "0 1 2 3 4"] [--gpu-id ID] [--execute] [--dry-run]

Default behavior is preflight/dry-run only. Full sequential training requires:
  CONFIRM_FULL_TRAINING=YES $(basename "$0") --execute
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --folds) FOLDS="${2:?--folds requires a value}"; export FOLDS; shift 2 ;;
    --gpu-id) GPU_ID="${2:?--gpu-id requires a value}"; export GPU_ID; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    --dry-run) DRY_RUN=1; export DRY_RUN; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

die() { printf '[train-5fold] ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '[train-5fold] %s\n' "$*"; }

read -r -a fold_array <<< "${FOLDS}"
[[ "${#fold_array[@]}" -gt 0 ]] || die "No folds selected"
for fold in "${fold_array[@]}"; do [[ "${fold}" =~ ^[0-4]$ ]] || die "Invalid fold: ${fold}"; done

if [[ "${EXECUTE}" != "1" || "${DRY_RUN}" == "1" ]]; then
  log "Preflight only; no fold will be trained."
  for fold in "${fold_array[@]}"; do
    DRY_RUN=1 "${SCRIPT_DIR}/train_fold.sh" "${fold}" --gpu-id "${GPU_ID}"
  done
  log "To start sequential training, set CONFIRM_FULL_TRAINING=YES and pass --execute."
  exit 0
fi

[[ "${CONFIRM_FULL_TRAINING:-}" == "YES" ]] || die "Full training requires CONFIRM_FULL_TRAINING=YES in addition to --execute"
log "Explicit confirmation received. Running folds sequentially on visible GPU ${GPU_ID}."
for fold in "${fold_array[@]}"; do
  "${SCRIPT_DIR}/train_fold.sh" "${fold}" --gpu-id "${GPU_ID}"
done
