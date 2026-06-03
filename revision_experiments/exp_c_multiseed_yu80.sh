#!/usr/bin/env bash
# EXP-C: 3-seed mean ± std for LR-SAF on YorkUrban-80 fine-tune.
# Run on evidlife-server from ~/Documents/yping/LR-SAF-LSD/code/lr_saf/
set -euo pipefail

cd "$(dirname "$0")/.."   # cd to code/lr_saf

SEEDS=(42 17 2024)
OUT_DIR=../revision_experiments/logs
mkdir -p "$OUT_DIR"

for seed in "${SEEDS[@]}"; do
    echo "==== seed=$seed ===="
    CUDA_VISIBLE_DEVICES=0 \
        python train_full.py --seed $seed --epochs 20 \
        --out "checkpoints/lr_saf_yu80_seed${seed}.pth" \
        2>&1 | tee "$OUT_DIR/exp_c_yu80_seed${seed}.log"

    CUDA_VISIBLE_DEVICES=0 \
        python eval_compare.py \
        --ckpt "checkpoints/lr_saf_yu80_seed${seed}.pth" \
        --out "$OUT_DIR/exp_c_yu80_seed${seed}_eval.json"
done

python revision_experiments/aggregate_multiseed.py \
    --pattern "$OUT_DIR/exp_c_yu80_seed*_eval.json" \
    --out "$OUT_DIR/exp_c_yu80_aggregate.json"
echo "EXP-C YorkUrban complete. Results in $OUT_DIR/exp_c_yu80_aggregate.json"
