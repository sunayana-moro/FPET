#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-fpet}"
ENV_FILE="${ROOT_DIR}/environment.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found on PATH" >&2
  exit 1
fi

conda env create -f "${ENV_FILE}" -n "${ENV_NAME}" || conda env update -f "${ENV_FILE}" -n "${ENV_NAME}" --prune

cat <<EOF

Environment ready.
Activate it with: conda activate ${ENV_NAME}
If you downloaded CIFAR-10 manually, keep it in: data/cifar-10-batches-py
Run the experiment with: python experiment.py
EOF
