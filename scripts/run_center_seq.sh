#!/usr/bin/env bash
# Run experiments sequentially with optional auxiliary files: run_center_seq.sh <data_root> <site> [<spacing_file>] [<label_map_json>] exp4c
#   File arguments are detected in any order (.json = label map, anything else = spacing). If omitted, the data folder convention / built-in mapping is used.
#   nohup bash scripts/run_center_seq.sh <data_root> A exp4c > /dev/null 2>&1 &
cd "$(dirname "$0")/.."; DATA=${1:?data}; SITE=${2:?site}; shift 2
SPACING="-"; LABELMAP="-"; EXPS=""
for a in "$@"; do
  if [ -f "configs/$a.yaml" ]; then EXPS="$EXPS $a"
  elif [ -f "$a" ]; then
    case "$a" in *.json) LABELMAP="$a";; *) SPACING="$a";; esac
  else echo "Unknown argument (neither an experiment yaml nor a file): $a"; exit 1; fi
done
for EXP in $EXPS; do bash scripts/run_center.sh "$EXP" "$DATA" "$SITE" "${RUN_NAME:-}" "$SPACING" "$LABELMAP"; done
