#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT
source "${REPO_ROOT}/config/experiment.env"

log() { printf '[bootstrap] %s\n' "$*"; }
die() { printf '[bootstrap] ERROR: %s\n' "$*" >&2; exit 1; }
warn() { printf '[bootstrap] WARNING: %s\n' "$*" >&2; }
print_cmd() { printf '+'; printf ' %q' "$@"; printf '\n'; }
run() { print_cmd "$@"; [[ "${DRY_RUN}" == "1" ]] || "$@"; }

[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
[[ "${ALLOW_NONEXACT}" == "0" || "${ALLOW_NONEXACT}" == "1" ]] || die "ALLOW_NONEXACT must be 0 or 1"
command -v git >/dev/null 2>&1 || die "git is required"
command -v "${PYTHON_BOOTSTRAP_BIN}" >/dev/null 2>&1 || die "Python not found: ${PYTHON_BOOTSTRAP_BIN}"

actual_python="$("${PYTHON_BOOTSTRAP_BIN}" -c 'import platform; print(platform.python_version())')"
if [[ "${actual_python}" != "${EXPECTED_PYTHON_VERSION}" ]]; then
  if [[ "${ALLOW_NONEXACT}" == "1" ]]; then
    warn "Python ${actual_python} differs from audited ${EXPECTED_PYTHON_VERSION}."
  else
    die "Exact bootstrap requires Python ${EXPECTED_PYTHON_VERSION}; found ${actual_python}."
  fi
fi

if [[ -d "${NNUNET_SOURCE_DIR}/.git" ]]; then
  current="$(git -C "${NNUNET_SOURCE_DIR}" rev-parse HEAD)"
  if [[ "${current}" != "${NNUNET_COMMIT}" ]]; then
    [[ -z "$(git -C "${NNUNET_SOURCE_DIR}" status --porcelain)" ]] || die "nnU-Net checkout is dirty at wrong commit: ${current}"
    run git -C "${NNUNET_SOURCE_DIR}" fetch origin "${NNUNET_COMMIT}"
    run git -C "${NNUNET_SOURCE_DIR}" checkout --detach "${NNUNET_COMMIT}"
  else
    log "nnU-Net already at ${NNUNET_COMMIT}."
  fi
else
  if [[ -e "${NNUNET_SOURCE_DIR}" && -n "$(find "${NNUNET_SOURCE_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    die "Refusing to replace non-empty non-Git path: ${NNUNET_SOURCE_DIR}"
  fi
  run mkdir -p "$(dirname -- "${NNUNET_SOURCE_DIR}")"
  run git clone --no-checkout "${NNUNET_REPOSITORY}" "${NNUNET_SOURCE_DIR}"
  run git -C "${NNUNET_SOURCE_DIR}" checkout --detach "${NNUNET_COMMIT}"
fi

[[ -f "${REPO_ROOT}/requirements/base.txt" ]] || die "Missing requirements/base.txt"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  run "${PYTHON_BOOTSTRAP_BIN}" -m venv "${VENV_DIR}"
fi
run "${PIP_BIN}" install -r "${REPO_ROOT}/requirements/base.txt"
run "${REPO_ROOT}/scripts/install_overlay.sh"
run "${PIP_BIN}" install --no-deps -e "${NNUNET_SOURCE_DIR}"

verify_code='import platform, torch; from batchgenerators.utilities.file_and_folder_operations import join; import nnunetv2; from nnunetv2.utilities.find_class_by_name import recursive_find_python_class; name="nnUNetTrainer_D007ETTCRegionAuxHead_SoftMoEConfigK_Pilot"; cls=recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"), name, "nnunetv2.training.nnUNetTrainer"); assert cls is not None, name; print({"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda, "trainer": cls.__module__})'
run env PYTHONPATH="${NNUNET_SOURCE_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -c "${verify_code}"

log "Bootstrap complete. No preprocessing or training was started."
