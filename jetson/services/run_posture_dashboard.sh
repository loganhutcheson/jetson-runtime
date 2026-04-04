#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

db_path="${POSTURE_DASHBOARD_DB:-/home/logan/posture/posture_history.sqlite3}"
host="${POSTURE_DASHBOARD_HOST:-0.0.0.0}"
port="${POSTURE_DASHBOARD_PORT:-8787}"

echo "[posture-dashboard] db=${db_path} host=${host} port=${port}"
exec python3 -u jetson/web/posture_dashboard.py --db "${db_path}" --host "${host}" --port "${port}"
