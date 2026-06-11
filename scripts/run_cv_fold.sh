#!/bin/bash
# One CV fold: train AFM + LR-SAF backbones (held-out test = fold), then the
# geom head on each, all on the same fold partition. Absolute paths throughout.
set -e
FOLD=$1; GPU=$2
ROOT=/home/server/Documents/yping/LR-SAF-LSD
RD=$ROOT/code/lr_saf/logs/revision/cv
CK=$ROOT/checkpoints/revision/cv
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$RD" "$CK"
cd "$ROOT/code/lr_saf"

echo "[fold $FOLD gpu $GPU] AFM backbone $(date)"
python3 revision_experiments/train_afm_yu80_finetune.py --epochs 20 --fold "$FOLD" \
  --out "$CK/afm_fold${FOLD}.pth" --log "$RD/afm_bb_fold${FOLD}.json"

echo "[fold $FOLD] LR-SAF backbone $(date)"
python3 train_full.py --epochs 20 --fold "$FOLD" \
  --out "$CK/lrsaf_fold${FOLD}.pth" --log "$RD/lrsaf_bb_fold${FOLD}.json"

echo "[fold $FOLD] AFM+head $(date)"
python3 revision_experiments/train_conf_on_afm.py --epochs 15 --fold "$FOLD" \
  --afm_ckpt "$CK/afm_fold${FOLD}.pth" --out_json "$RD/afm_head_fold${FOLD}.json"

echo "[fold $FOLD] LR-SAF+head $(date)"
python3 train_confidence.py --epochs 30 --fold "$FOLD" \
  --ckpt "$CK/lrsaf_fold${FOLD}.pth" --out_json "$RD/lrsaf_head_fold${FOLD}.json"

echo "[fold $FOLD] DONE $(date)"
