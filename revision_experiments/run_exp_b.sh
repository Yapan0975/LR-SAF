#!/usr/bin/env bash
# EXP-B: Wireframe-trained LR-SAF, zero-shot on YorkUrban val 22.
set -uo pipefail
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES=3 python3 revision_experiments/eval_yu_from_ckpt.py \
    --ckpt /home/server/Documents/yping/LR-SAF-LSD/checkpoints/lr_saf_wireframe_best.pth \
    --out logs/revision/exp_b_zeroshot.json \
    --label wireframe_only_zeroshot \
    > logs/revision/exp_b.log 2>&1
echo EXP_B_DONE
