#!/usr/bin/env bash
set -euo pipefail

# Make sure volumes exist
mkdir -p "${DATA_ROOT}" "${DUCKDB_PATH%/*}"

# Optional: show env useful for debugging
echo "[env] DATA_ROOT=${DATA_ROOT}"
echo "[env] DUCKDB_PATH=${DUCKDB_PATH}"

# If you have a one-time migration/DDL step, you can call a small Python script here.
# python -m app.backend.prepare_db --db "${DUCKDB_PATH}"

# Launch the app (change app/app.py if your entrypoint is different)
exec streamlit run app/app.py \
  --server.port "${STREAMLIT_SERVER_PORT}" \
  --server.headless true

