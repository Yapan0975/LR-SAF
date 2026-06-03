#!/usr/bin/env bash
# Sequential batch driver for Paper 1 Major Revision experiments.
# Runs on evidlife-server, GPU 3 (others are occupied by other users' jobs).
# Estimated total: 4-6 h sequential.
#
# Usage (from remote shell):
#   cd ~/Documents/yping/LR-SAF-LSD/code/lr_saf
#   nohup bash revision_experiments/launch_all_revision.sh \
#       > revision_experiments/launch_all.log 2>&1 &
#
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT_RUN_LOG=logs/revision/launch_all.summary.log

start=$(date +%s)
{
    echo "=== Paper 1 Major Rev batch driver started $(date -Iseconds) ==="
} | tee -a "$ROOT_RUN_LOG"

SEEDS=(42 17 2024)

# EXP-C: 3-seed YorkUrban fine-tune
for SEED in "${SEEDS[@]}"; do
    LOG=logs/revision/train_seed${SEED}.json
    RUN=logs/revision/run_seed${SEED}.log
    CKPT=checkpoints/revision/lr_saf_seed${SEED}.pth
    if [ -f "$CKPT" ]; then
        echo "[skip] seed=$SEED already has checkpoint $CKPT" | tee -a "$ROOT_RUN_LOG"
        continue
    fi
    echo "[exp-c] launching seed=$SEED at $(date -Iseconds)" | tee -a "$ROOT_RUN_LOG"
    CUDA_VISIBLE_DEVICES=3 python3 train_full.py \
        --seed "$SEED" --epochs 20 \
        --out "$CKPT" --log "$LOG" \
        > "$RUN" 2>&1
    rc=$?
    echo "[exp-c] seed=$SEED rc=$rc at $(date -Iseconds)" | tee -a "$ROOT_RUN_LOG"
    tail -n 3 "$RUN" | tee -a "$ROOT_RUN_LOG"
done

# Aggregate
echo "[aggregate] computing mean ± std across seeds" | tee -a "$ROOT_RUN_LOG"
python3 revision_experiments/aggregate_multiseed.py \
    --pattern "logs/revision/train_seed*.json" \
    --out "logs/revision/exp_c_aggregate.json" \
    2>&1 | tee -a "$ROOT_RUN_LOG" || true

elapsed=$(( $(date +%s) - start ))
echo "=== Batch driver done at $(date -Iseconds), elapsed ${elapsed}s ===" | tee -a "$ROOT_RUN_LOG"
echo DONE_LAUNCH_ALL >> "$ROOT_RUN_LOG"
