#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_SRC="${SCRIPT_DIR}/posture-dashboard.service"
SERVICE_DST="/etc/systemd/system/posture-dashboard.service"
ENV_DST="/etc/default/posture-dashboard"
ENV_EXAMPLE="${SCRIPT_DIR}/posture-dashboard.env.example"

sudo install -m 0644 "${SERVICE_SRC}" "${SERVICE_DST}"
if [[ ! -f "${ENV_DST}" ]]; then
  sudo install -m 0644 "${ENV_EXAMPLE}" "${ENV_DST}"
fi
sudo chmod +x "${SCRIPT_DIR}/run_posture_dashboard.sh"
sudo systemctl daemon-reload
echo "installed ${SERVICE_DST}"
echo "env file: ${ENV_DST}"
echo "next steps:"
echo "  sudo systemctl enable --now posture-dashboard.service"
