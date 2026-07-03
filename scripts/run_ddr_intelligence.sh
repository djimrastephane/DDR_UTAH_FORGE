#!/usr/bin/env bash
# Run the DDR Operational Intelligence Streamlit app from the repo root.
set -e
cd "$(dirname "$0")/.."
streamlit run app/ui/ddr_intelligence.py \
  --server.port 8502 \
  --server.headless false \
  --browser.gatherUsageStats false
