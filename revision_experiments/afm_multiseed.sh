#!/usr/bin/env bash
# Sequential AFM YU-80 fine-tune for seeds 17 and 2024 (seed 42 already done).
set -uo pipefail
cd "$(dirname "$0")/.."

SUMMARY=logs/revision/afm_multiseed.summary.log
echo "=== AFM multi-seed (17,2024) started $(date -Iseconds) ===" | tee -a $SUMMARY

for SEED in 17 2024; do
    CKPT=checkpoints/revision/afm_yu80_seed${SEED}.pth
    LOG=logs/revision/afm_yu80_seed${SEED}.json
    RUN=logs/revision/afm_yu80_seed${SEED}.log
    if [ -f "$CKPT" ]; then
        echo "[skip] $CKPT exists" | tee -a $SUMMARY
        continue
    fi
    echo "[launch] seed=$SEED at $(date -Iseconds)" | tee -a $SUMMARY
    CUDA_VISIBLE_DEVICES=3 python3 revision_experiments/train_afm_yu80_finetune.py \
        --epochs 20 --seed $SEED --out "$CKPT" --log "$LOG" > "$RUN" 2>&1
    echo "[done] seed=$SEED rc=$? at $(date -Iseconds)" | tee -a $SUMMARY
    tail -n 3 "$RUN" | tee -a $SUMMARY
done

echo "=== AFM multi-seed done at $(date -Iseconds) ===" | tee -a $SUMMARY
echo AFM_MULTISEED_DONE >> $SUMMARY
