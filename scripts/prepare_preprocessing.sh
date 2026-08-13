#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT

NP=4
SKIP_FINGERPRINT=0
SKIP_PREPROCESS=0
REQUEST_DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --np) NP="${2:?--np requires a value}"; shift 2 ;;
    --skip-fingerprint) SKIP_FINGERPRINT=1; shift ;;
    --skip-preprocess) SKIP_PREPROCESS=1; shift ;;
    --dry-run) REQUEST_DRY_RUN=1; shift ;;
    -h|--help)
      printf 'Usage: %s [--np N] [--skip-fingerprint] [--skip-preprocess] [--dry-run]\n' "$(basename "$0")"
      exit 0
      ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
source "${REPO_ROOT}/config/experiment.env"
[[ "${REQUEST_DRY_RUN}" == "0" ]] || DRY_RUN=1

log() { printf '[prepare] %s\n' "$*"; }
die() { printf '[prepare] ERROR: %s\n' "$*" >&2; exit 1; }
print_cmd() { printf '+'; printf ' %q' "$@"; printf '\n'; }
run() { print_cmd "$@"; [[ "${DRY_RUN}" == "1" ]] || "$@"; }
install_equal() {
  local source="$1" destination="$2" label="$3"
  if [[ -f "${destination}" ]]; then
    cmp -s "${source}" "${destination}" || die "Existing ${label} differs: ${destination}"
    log "${label} already installed."
  else
    run install -D -m 0644 "${source}" "${destination}"
  fi
}

[[ "${NP}" =~ ^[1-9][0-9]*$ ]] || die "--np must be a positive integer"
log "dataset=${DATASET_NAME}; strict_pseudo=${EXPECTED_PSEUDO_CASES}; expected_total=${EXPECTED_TOTAL_CASES}"
log "raw=${RAW_DATASET_DIR}; preprocessed=${PREPROCESSED_DATASET_DIR}; weights=${CASE_WEIGHT_ROOT}"

fingerprint="${VENV_DIR}/bin/nnUNetv2_extract_fingerprint"
preprocess="${VENV_DIR}/bin/nnUNetv2_preprocess"
if [[ "${DRY_RUN}" == "1" ]]; then
  [[ "${SKIP_FINGERPRINT}" == "1" ]] || print_cmd "${fingerprint}" -d "${DATASET_ID}" --verify_dataset_integrity -np "${NP}"
  [[ "${SKIP_PREPROCESS}" == "1" ]] || print_cmd "${preprocess}" -d "${DATASET_ID}" -plans_name "${PLANS}" -c "${CONFIGURATION}" -np "${NP}"
  log "dry-run: private assets were not required and preprocessing was not started."
  exit 0
fi

[[ -d "${RAW_DATASET_DIR}/imagesTr" ]] || die "Missing imagesTr"
[[ -d "${RAW_DATASET_DIR}/labelsTr" ]] || die "Missing labelsTr"
[[ -f "${RAW_DATASET_DIR}/dataset.json" ]] || die "Missing dataset.json"
[[ -f "${RAW_DATASET_DIR}/dataset007_case_manifest.csv" ]] || die "Missing Dataset007 manifest"
[[ -f "${PRIVATE_SPLITS}" ]] || die "Missing final splits"
for fold in 0 1 2 3 4; do
  [[ -f "${CASE_WEIGHT_ROOT}/case_weights_fold${fold}.json" ]] || die "Missing final weight file for fold ${fold}"
  [[ -f "${D005_PRETRAINED_ROOT}/fold_${fold}/checkpoint_best.pth" ]] || die "Missing D005 checkpoint for fold ${fold}"
done
run mkdir -p "${PREPROCESSED_DATASET_DIR}"
install_equal "${REPO_ROOT}/config/${PLANS}.json" "${PREPROCESSED_DATASET_DIR}/${PLANS}.json" "plans"
install_equal "${PRIVATE_SPLITS}" "${PREPROCESSED_DATASET_DIR}/splits_final.json" "splits"
[[ -x "${fingerprint}" && -x "${preprocess}" ]] || die "Run scripts/bootstrap.sh first"
[[ "${SKIP_FINGERPRINT}" == "1" ]] || run "${fingerprint}" -d "${DATASET_ID}" --verify_dataset_integrity -np "${NP}"
[[ "${SKIP_PREPROCESS}" == "1" ]] || run "${preprocess}" -d "${DATASET_ID}" -plans_name "${PLANS}" -c "${CONFIGURATION}" -np "${NP}"
log "Preparation complete. No training was started."
