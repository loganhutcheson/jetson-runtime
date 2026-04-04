#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SERVICE_SRC="${SCRIPT_DIR}/posture-runtime.service"
SERVICE_DST="/etc/systemd/system/posture-runtime.service"
ENV_DST="/etc/default/posture-runtime"
ENV_EXAMPLE="${SCRIPT_DIR}/posture-runtime.env.example"

if [[ ! -f "${SERVICE_SRC}" ]]; then
  echo "missing ${SERVICE_SRC}" >&2
  exit 1
fi

sudo install -m 0644 "${SERVICE_SRC}" "${SERVICE_DST}"
if [[ ! -f "${ENV_DST}" ]]; then
  sudo install -m 0644 "${ENV_EXAMPLE}" "${ENV_DST}"
fi
sudo chmod +x "${SCRIPT_DIR}/run_posture_runtime.sh"
sudo systemctl daemon-reload
echo "installed ${SERVICE_DST}"
echo "env file: ${ENV_DST}"
echo "next steps:"
echo "  sudo systemctl enable --now posture-runtime.service"
echo "developer flow:"
echo "  sudo systemctl stop posture-runtime.service"
echo "  cd ${REPO_ROOT} && ./jetson/services/run_posture_runtime.sh"
