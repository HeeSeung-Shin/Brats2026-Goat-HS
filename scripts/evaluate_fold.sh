#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT
source "${REPO_ROOT}/config/experiment.env"

[[ $# -ge 1 ]] || { printf 'Usage: %s FOLD [--dry-run]\n' "$(basename "$0")" >&2; exit 2; }
FOLD="$1"; shift
[[ "${FOLD}" =~ ^[0-4]$ ]] || { printf 'FOLD must be 0..4\n' >&2; exit 2; }
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi
prediction_dir="${RESULT_ROOT}/fold_${FOLD}/validation"
output_dir="${REPO_ROOT}/runs/metrics"
cmd=("${PYTHON_BIN}" "${REPO_ROOT}/scripts/evaluate_original_gt.py"
  --prediction-dir "${prediction_dir}"
  --dataset005-labels "${D005_LABELS_DIR}"
  --splits-json "${PRIVATE_SPLITS}"
  --fold "${FOLD}"
  --output-csv "${output_dir}/metrics_K4_fold${FOLD}.csv"
  --summary-json "${output_dir}/metrics_K4_fold${FOLD}.summary.json")
printf '+'; printf ' %q' "${cmd[@]}"; printf '\n'
if [[ "${DRY_RUN}" == "1" ]]; then printf '[evaluate-fold] dry-run: evaluation was not started.\n'; exit 0; fi
[[ -d "${prediction_dir}" ]] || { printf 'Missing predictions: %s\n' "${prediction_dir}" >&2; exit 1; }
[[ -d "${D005_LABELS_DIR}" ]] || { printf 'Missing D005 labels: %s\n' "${D005_LABELS_DIR}" >&2; exit 1; }
mkdir -p "${output_dir}"
"${cmd[@]}"
