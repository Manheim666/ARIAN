#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# ARIAN — Run the full 6-notebook pipeline end-to-end
# Uses papermill to execute each notebook in order.
#
# Usage:
#   bash run_pipeline.sh          # run all 6 notebooks
#   bash run_pipeline.sh 3 6      # run NB03 through NB06 only
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NB_DIR="${SCRIPT_DIR}/notebooks"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

NOTEBOOKS=(
    "01_Data_Ingestion"
    "02_EDA_FeatureEngineering"
    "03_Weather_TimeSeries"
    "04_Wildfire_Detection"
    "05_Risk_Prediction_Dashboard"
    "06_Climate_Report"
)

# Optional range arguments (1-indexed): start [end]
START=${1:-1}
END=${2:-${#NOTEBOOKS[@]}}

echo "════════════════════════════════════════════════════════"
echo "  ARIAN Pipeline  —  notebooks ${START}..${END}"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════"

FAILED=0

for i in $(seq "$START" "$END"); do
    idx=$((i - 1))
    nb="${NOTEBOOKS[$idx]}"
    input="${NB_DIR}/${nb}.ipynb"
    log="${LOG_DIR}/${nb}.log"

    echo ""
    echo "▶ [${i}/${END}] ${nb}"
    echo "  input : ${input}"
    echo "  log   : ${log}"

    if [ ! -f "${input}" ]; then
        echo "  ✗ notebook not found — skipping"
        FAILED=1
        continue
    fi

    START_TS=$(date +%s)

    if papermill "${input}" "${input}" \
         --cwd "${SCRIPT_DIR}" \
         --log-output \
         --no-progress-bar \
         > "${log}" 2>&1; then
        ELAPSED=$(( $(date +%s) - START_TS ))
        echo "  ✓ done  (${ELAPSED}s)"
    else
        ELAPSED=$(( $(date +%s) - START_TS ))
        echo "  ✗ FAILED after ${ELAPSED}s — see ${log}"
        FAILED=1
        break
    fi
done

echo ""
echo "════════════════════════════════════════════════════════"
if [ "${FAILED}" -eq 0 ]; then
    echo "  Pipeline finished successfully."
else
    echo "  Pipeline stopped due to errors. Check logs above."
fi
echo "  Ended: $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════"

exit "${FAILED}"
