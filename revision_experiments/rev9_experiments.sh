#!/usr/bin/env bash
# rev9 experiments: P3 no-TNNR multi-seed + P4 v9 + P5 v10
set -uo pipefail
cd "$(dirname "$0")/.."

SUMMARY=logs/revision/rev9_summary.log
echo "=== rev9 batch started $(date -Iseconds) ===" | tee -a $SUMMARY

# P3: no-TNNR seeds 17 and 2024 (seed 42 already done as lr_saf_no_tnnr.pth)
for SEED in 17 2024; do
    CKPT=checkpoints/revision/lr_saf_no_tnnr_seed${SEED}.pth
    LOG=logs/revision/no_tnnr_seed${SEED}.json
    RUN=logs/revision/no_tnnr_seed${SEED}.log
    if [ -f "$CKPT" ]; then
        echo "[skip] $CKPT exists" | tee -a $SUMMARY; continue
    fi
    echo "[P3] no-TNNR seed=$SEED at $(date -Iseconds)" | tee -a $SUMMARY
    CUDA_VISIBLE_DEVICES=3 python3 train_full.py \
        --seed $SEED --epochs 20 --lam_tnnr 0 \
        --reg none \
        --out "$CKPT" --log "$LOG" > "$RUN" 2>&1
    echo "[P3] no-TNNR seed=$SEED rc=$? at $(date -Iseconds)" | tee -a $SUMMARY
    tail -n 3 "$RUN" | tee -a $SUMMARY
done

# P4: v9 ablation = K=1 + bounded encoding (seed 42, single run, matches Tab IV style)
CKPT=checkpoints/revision/lr_saf_v9_K1bounded_seed42.pth
LOG=logs/revision/v9_K1bounded_seed42.json
RUN=logs/revision/v9_K1bounded_seed42.log
echo "[P4] v9 K=1 + bounded at $(date -Iseconds)" | tee -a $SUMMARY
CUDA_VISIBLE_DEVICES=3 python3 train_full.py \
    --seed 42 --epochs 20 --K 1 --encoding tanh \
    --out "$CKPT" --log "$LOG" > "$RUN" 2>&1
echo "[P4] v9 rc=$? at $(date -Iseconds)" | tee -a $SUMMARY
tail -n 3 "$RUN" | tee -a $SUMMARY

# P5: v10 = TV regularizer instead of TNNR (seed 42, single run, matches Tab IV style)
CKPT=checkpoints/revision/lr_saf_v10_tv_seed42.pth
LOG=logs/revision/v10_tv_seed42.json
RUN=logs/revision/v10_tv_seed42.log
echo "[P5] v10 TV regularizer at $(date -Iseconds)" | tee -a $SUMMARY
CUDA_VISIBLE_DEVICES=3 python3 train_full.py \
    --seed 42 --epochs 20 --reg tv \
    --out "$CKPT" --log "$LOG" > "$RUN" 2>&1
echo "[P5] v10 rc=$? at $(date -Iseconds)" | tee -a $SUMMARY
tail -n 3 "$RUN" | tee -a $SUMMARY

echo "=== rev9 batch done $(date -Iseconds) ===" | tee -a $SUMMARY
echo REV9_DONE >> $SUMMARY
