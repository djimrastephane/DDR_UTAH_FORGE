#!/usr/bin/env bash
set -euo pipefail

export DEMO_MODE="${DEMO_MODE:-1}"
export UI_DATA_ROOT="${UI_DATA_ROOT:-data/processed}"
export UI_TABLE_CHUNKING_MODE="${UI_TABLE_CHUNKING_MODE:-row_blocks}"

python -m uvicorn app.api.main:app --reload --port "${API_PORT:-8000}"
