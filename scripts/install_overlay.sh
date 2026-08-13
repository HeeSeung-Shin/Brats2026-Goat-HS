#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT
# shellcheck source=../config/experiment.env
source "${REPO_ROOT}/config/experiment.env"

log() { printf '[install-overlay] %s\n' "$*"; }
die() { printf '[install-overlay] ERROR: %s\n' "$*" >&2; exit 1; }
warn() { printf '[install-overlay] WARNING: %s\n' "$*" >&2; }
print_cmd() { printf '+'; printf ' %q' "$@"; printf '\n'; }
run() { print_cmd "$@"; [[ "${DRY_RUN}" == "1" ]] || "$@"; }

[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
[[ -d "${NNUNET_OVERLAY_DIR}/nnunetv2" ]] || die "Missing overlay tree: ${NNUNET_OVERLAY_DIR}/nnunetv2"

if [[ ! -d "${NNUNET_SOURCE_DIR}/.git" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    warn "nnU-Net checkout is not present yet; bootstrap will clone it before installing the overlay."
  else
    die "Pinned nnU-Net checkout not found: ${NNUNET_SOURCE_DIR}"
  fi
else
  current="$(git -C "${NNUNET_SOURCE_DIR}" rev-parse HEAD)"
  if [[ "${current}" != "${NNUNET_COMMIT}" ]]; then
    die "nnU-Net commit mismatch: expected ${NNUNET_COMMIT}, got ${current}"
  fi
fi

count=0
while IFS= read -r -d '' source_file; do
  relative="${source_file#${NNUNET_OVERLAY_DIR}/}"
  destination="${NNUNET_SOURCE_DIR}/${relative}"
  run install -D -m 0644 "${source_file}" "${destination}"
  count=$((count + 1))
done < <(find "${NNUNET_OVERLAY_DIR}" -type f -name '*.py' -print0 | sort -z)

[[ "${count}" -gt 0 ]] || die "No Python overlay files were found"

log "Installed ${count} overlay Python files. No training was started."
