#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT
source "${REPO_ROOT}/config/experiment.env"

NP=4
SKIP_FINGERPRINT=0
SKIP_PREPROCESS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --np) NP="${2:?--np requires a value}"; shift 2 ;;
    --skip-fingerprint) SKIP_FINGERPRINT=1; shift ;;
    --skip-preprocess) SKIP_PREPROCESS=1; shift ;;
    --dry-run) DRY_RUN=1; export DRY_RUN; shift ;;
    -h|--help) printf 'Usage: %s [--np N] [--skip-fingerprint] [--skip-preprocess] [--dry-run]\n' "$(basename "$0")"; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
log() { printf '[prepare] %s\n' "$*"; }
die() { printf '[prepare] ERROR: %s\n' "$*" >&2; exit 1; }
print_cmd() { printf '+'; printf ' %q' "$@"; printf '\n'; }
run() { print_cmd "$@"; [[ "${DRY_RUN}" == "1" ]] || "$@"; }
verify() {
  local path="$1" expected="$2" label="$3"
  [[ -f "${path}" ]] || die "Missing ${label}: ${path}"
  local actual; actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || die "${label} hash mismatch: ${actual}"
}
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
[[ -d "${RAW_DATASET_DIR}/imagesTr" ]] || die "Missing Dataset007 imagesTr"
[[ -d "${RAW_DATASET_DIR}/labelsTr" ]] || die "Missing Dataset007 labelsTr"
verify "${REPO_ROOT}/config/dataset.json" "${DATASET_JSON_SHA256}" "dataset.json"
verify "${REPO_ROOT}/config/${PLANS}.json" "${PLANS_SHA256}" "plans"
verify "${PRIVATE_MANIFEST}" "${PRIVATE_MANIFEST_SHA256}" "private manifest"
verify "${PRIVATE_SPLITS}" "${PRIVATE_SPLITS_SHA256}" "private splits"
verify "${CASE_META_FILE}" "${CASE_META_SHA256}" "case metadata"
for fold in 0 1 2 3 4; do
  weight_var="CASE_WEIGHT_FOLD${fold}_SHA256"
  checkpoint_var="D005_FOLD${fold}_SHA256"
  verify "${CASE_WEIGHT_ROOT}/case_weights_fold${fold}.json" "${!weight_var}" "fold ${fold} case weights"
  verify "${D005_PRETRAINED_ROOT}/fold_${fold}/checkpoint_best.pth" "${!checkpoint_var}" "fold ${fold} D005 checkpoint_best"
done

install_equal "${REPO_ROOT}/config/dataset.json" "${RAW_DATASET_DIR}/dataset.json" "dataset.json"
install_equal "${PRIVATE_MANIFEST}" "${RAW_DATASET_DIR}/dataset007_case_manifest.csv" "manifest"
run mkdir -p "${PREPROCESSED_DATASET_DIR}"
install_equal "${REPO_ROOT}/config/${PLANS}.json" "${PREPROCESSED_DATASET_DIR}/${PLANS}.json" "plans"
install_equal "${PRIVATE_SPLITS}" "${PREPROCESSED_DATASET_DIR}/splits_final.json" "splits"

fingerprint="${VENV_DIR}/bin/nnUNetv2_extract_fingerprint"
preprocess="${VENV_DIR}/bin/nnUNetv2_preprocess"
if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -x "${fingerprint}" && -x "${preprocess}" ]] || die "Run scripts/bootstrap.sh first"
fi
[[ "${SKIP_FINGERPRINT}" == "1" ]] || run "${fingerprint}" -d "${DATASET_ID}" --verify_dataset_integrity -np "${NP}"
[[ "${SKIP_PREPROCESS}" == "1" ]] || run "${preprocess}" -d "${DATASET_ID}" -plans_name "${PLANS}" -c "${CONFIGURATION}" -np "${NP}"
log "Preparation complete. No training was started."
