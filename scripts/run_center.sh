#!/usr/bin/env bash
# One-command site runner (Linux, e.g., DGX Spark): single-site 5-fold training, then the FL client. Launch with nohup so it keeps running if the session disconnects.
#   nohup bash scripts/run_center.sh exp4c <data_root> <site> [run_name] [<spacing_file>|-] [<label_map_json>|-] > /dev/null 2>&1 &
# Each run is saved separately under outputs/<exp>/<run>/ (default run name = start time). To resume, pass the same run name as the 4th argument.
# To stop: create outputs/<exp>/<run>/client_<site>/STOP.txt. On restart, completed folds/methods are skipped.
set -u; cd "$(dirname "$0")/.."
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
if [ "$FREE_GB" -lt 25 ]; then echo "[ERROR] Free disk space ${FREE_GB}GB < 25GB - free up space and rerun (this experiment needs about 60-120GB per site)"; exit 1; fi
if [ "$FREE_GB" -lt 100 ]; then echo "[WARNING] Free disk space ${FREE_GB}GB - may run out during the experiment (120GB+ recommended)"; fi
EXP=${1:?exp}; DATA=${2:?data dir}; SITE=${3:?site}; RUN=${4:-${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}}
SPACING=${5:-}; LABELMAP=${6:-}; [ "$SPACING" = "-" ] && SPACING=""; [ "$LABELMAP" = "-" ] && LABELMAP=""
SPACING_ARG=""; [ -n "$SPACING" ] && SPACING_ARG="--spacing-file $SPACING"
LABEL_ARG="";  [ -n "$LABELMAP" ] && LABEL_ARG="--lesion-labels $LABELMAP"
OUT=outputs/$EXP/$RUN/client_$SITE; mkdir -p "$OUT"; LOG=$OUT/run_center.log; echo "$RUN" > outputs/$EXP/LAST_RUN
# Pre-run GPU check: prevents silently running on CPU for days (seen before with a cu130 wheel on a 12.6 driver)
if ! .venv/bin/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>>"$LOG"; then
  echo "[ERROR] No GPU detected. Check the CUDA version in nvidia-smi and rerun scripts/install.sh (do not use uv sync/uv run - they remove torch)" | tee -a "$LOG"; exit 1
fi
# Server connectivity pre-check: the FL port (9595) is first contacted only after single-site training ends, so check it up front.
# Otherwise a blocked firewall would stall the FL stage after hours of single-site training. To bypass: SKIP_SERVER_CHECK=1
if [ "${SKIP_SERVER_CHECK:-0}" != "1" ]; then
  CHK=$(.venv/bin/python scripts/check_server.py --exp "$EXP" 2>&1); CS=$?
  echo "$CHK" | tee -a "$LOG"
  if [ "$CS" -ne 0 ]; then
    echo "[ERROR] Server connectivity check failed - verify the server is running, the firewall, and the SERVER_IP substitution, then rerun (set SKIP_SERVER_CHECK=1 to skip)" | tee -a "$LOG"; exit 1
  fi
fi
# Fetch pretrained weights automatically (downloaded from the server on port 9598 if missing)
IW=$(grep -E "^init_weights:" configs/$EXP.yaml | awk '{print $2}' | tr -d '"')
UURL=$(grep -E "^upload_url:" configs/$EXP.yaml | awk '{print $2}' | tr -d '"' | sed 's|/upload$||')
if [ -n "$IW" ] && [ ! -s "$IW" ] && [ -n "$UURL" ]; then
  mkdir -p "$(dirname "$IW")"; echo "[WEIGHTS] $IW not found - downloading from $UURL/weights" >> "$LOG"
  curl -fsSL -o "$IW" "$UURL/weights" >> "$LOG" 2>&1 \
    || curl -fsSL -o "$IW" "https://github.com/AISeedHub/CouinaudFL/releases/download/weights-v2.0/best.pth" >> "$LOG" 2>&1 \
    || { echo "[ERROR] Pretrained weight download failed (both server and GitHub) - place the file at $IW manually" | tee -a "$LOG"; exit 1; }
fi
echo "[$(date)] start $EXP $SITE run=$RUN" >> "$LOG"
while true; do
  [ -f "$OUT/STOP.txt" ] && { echo "STOP.txt - exit" >> "$LOG"; break; }
  .venv/bin/python scripts/patient_catalog.py --data "$DATA" --exp $EXP --site "$SITE" --run "$RUN" $SPACING_ARG $LABEL_ARG >> "$LOG" 2>&1
  RS=$(grep -E "^run_single:" configs/$EXP.yaml | awk '{print $2}'); RS=${RS:-1}
  if [ "$RS" != "0" ]; then
    .venv/bin/python scripts/single.py --config configs/$EXP.yaml --data "$DATA" --site "$SITE" --run "$RUN" $SPACING_ARG >> "$LOG" 2>&1
    if ! grep -q "SINGLE_DONE" "$OUT/single.log" 2>/dev/null; then echo "[$(date)] single.py not finished - retry in 60s" >> "$LOG"; sleep 60; continue; fi
  fi
  # Send the true local (single-site) models and metrics to the server before FL starts
  .venv/bin/python scripts/export_results.py --exp $EXP --site "$SITE" --run "$RUN" >> "$LOG" 2>&1 && .venv/bin/python scripts/upload_results.py --exp $EXP --site "$SITE" --run "$RUN" >> "$LOG" 2>&1
  .venv/bin/python scripts/client.py --config configs/$EXP.yaml --data "$DATA" --site "$SITE" --run "$RUN" $SPACING_ARG >> "$LOG" 2>&1
  [ -f "$OUT/STOP.txt" ] && break
  grep -q "ALL_SESSIONS_DONE" "$LOG" && break
  echo "[$(date)] process exited unexpectedly - restart in 60s" >> "$LOG"; sleep 60
done
# After completion: anonymized export, then upload to the server (on failure only the zip is kept)
.venv/bin/python scripts/export_results.py --exp $EXP --site "$SITE" --run "$RUN" >> "$LOG" 2>&1
.venv/bin/python scripts/upload_results.py --exp $EXP --site "$SITE" --run "$RUN" >> "$LOG" 2>&1
echo "[$(date)] done" >> "$LOG"
