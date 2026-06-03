#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
SUMMARY=logs/revision/afm_head_multiseed.summary.log
echo "=== AFM+head multi-seed started $(date -Iseconds) ===" | tee -a $SUMMARY
for SEED in 17 2024; do
    AFM_CKPT=$(pwd)/checkpoints/revision/afm_yu80_seed${SEED}.pth
    HEAD_CKPT=checkpoints/revision/conf_head_on_afm_seed${SEED}.pth
    LOG_JSON=logs/revision/conf_on_afm_seed${SEED}.json
    LOG_RUN=logs/revision/conf_on_afm_seed${SEED}.log
    if [ ! -f "$AFM_CKPT" ]; then
        echo "[skip] seed=$SEED AFM ckpt missing: $AFM_CKPT" | tee -a $SUMMARY; continue
    fi
    echo "[launch] seed=$SEED at $(date -Iseconds)" | tee -a $SUMMARY
    CUDA_VISIBLE_DEVICES=3 python3 revision_experiments/train_conf_on_afm.py \
        --epochs 15 --seed $SEED \
        --afm_ckpt "$AFM_CKPT" \
        --out_head "$HEAD_CKPT" --out_json "$LOG_JSON" \
        > "$LOG_RUN" 2>&1
    echo "[done] seed=$SEED rc=$? at $(date -Iseconds)" | tee -a $SUMMARY
    tail -n 3 "$LOG_RUN" | tee -a $SUMMARY
done
echo "=== done $(date -Iseconds) ===" | tee -a $SUMMARY
echo AFM_HEAD_MULTISEED_DONE >> $SUMMARY
