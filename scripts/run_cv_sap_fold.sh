#!/bin/bash
# Re-evaluate one CV fold under the CORRECTED standard LCNN sAP (dataset-level,
# squared-distance threshold, one-to-one). Reuses the existing fold backbones
# (F-measure unchanged); only re-scores backbone-only sAP and RE-TRAINS the
# heads with corrected one-to-one labels. Dumps per-image records for bootstrap.
set -e
FOLD=$1; GPU=$2
ROOT=/home/server/Documents/yping/LR-SAF-LSD
CK=$ROOT/checkpoints/revision/cv
RD=$ROOT/code/lr_saf/logs/revision/cv_sap
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$RD"
cd "$ROOT/code/lr_saf"

echo "[fold $FOLD gpu $GPU] AFM backbone-only sAP $(date)"
python3 dump_backbone_records.py --kind afm --ckpt "$CK/afm_fold${FOLD}.pth" \
  --fold "$FOLD" --out_json "$RD/afm_bb_fold${FOLD}.json"

echo "[fold $FOLD] LR-SAF backbone-only sAP $(date)"
python3 dump_backbone_records.py --kind lrsaf --ckpt "$CK/lrsaf_fold${FOLD}.pth" \
  --fold "$FOLD" --out_json "$RD/lrsaf_bb_fold${FOLD}.json"

echo "[fold $FOLD] AFM+head (corrected labels) $(date)"
python3 revision_experiments/train_conf_on_afm.py --epochs 15 --fold "$FOLD" \
  --afm_ckpt "$CK/afm_fold${FOLD}.pth" --out_json "$RD/afm_head_fold${FOLD}.json"

echo "[fold $FOLD] LR-SAF+head (corrected labels) $(date)"
python3 train_confidence.py --epochs 30 --fold "$FOLD" \
  --ckpt "$CK/lrsaf_fold${FOLD}.pth" --out_json "$RD/lrsaf_head_fold${FOLD}.json"

echo "[fold $FOLD] DONE $(date)"
