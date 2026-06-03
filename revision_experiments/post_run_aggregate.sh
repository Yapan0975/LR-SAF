#!/usr/bin/env bash
# Post-run aggregation for the Major Rev EXP-C multi-seed batch.
# train_full.py chdir's into ../afm_baseline before writing logs, so
# the JSON files end up at code/afm_baseline/logs/revision/.
# Pull them into the canonical lr_saf/logs/revision/ dir, then run
# aggregate_multiseed.py.
set -uo pipefail

cd "$(dirname "$0")/.."

SRC_DIR=../afm_baseline/logs/revision
DST_DIR=logs/revision
AGG_OUT=$DST_DIR/exp_c_aggregate.json

mkdir -p "$DST_DIR"

cp -uv "$SRC_DIR"/train_seed*.json "$DST_DIR"/ 2>&1 | tee -a "$DST_DIR/launch_all.summary.log"

python3 revision_experiments/aggregate_multiseed.py \
    --pattern "$DST_DIR/train_seed*.json" \
    --out "$AGG_OUT" 2>&1 | tee -a "$DST_DIR/launch_all.summary.log"

echo
echo "=== Aggregate result ==="
cat "$AGG_OUT"
