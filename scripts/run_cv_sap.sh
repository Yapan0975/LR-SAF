#!/bin/bash
# Re-run all 5 CV folds under the corrected standard sAP (3 GPUs, 2 waves).
ROOT=/home/server/Documents/yping/LR-SAF-LSD
cd "$ROOT"
rm -f logs/cv_sap_done.flag
echo "## CV-SAP START $(date)" > logs/cv_sap.log
nohup bash scripts/run_cv_sap_fold.sh 0 0 > logs/cv_sap_fold0.log 2>&1 &
nohup bash scripts/run_cv_sap_fold.sh 1 1 > logs/cv_sap_fold1.log 2>&1 &
nohup bash scripts/run_cv_sap_fold.sh 2 2 > logs/cv_sap_fold2.log 2>&1 &
wait
echo "## wave1 done $(date)" >> logs/cv_sap.log
nohup bash scripts/run_cv_sap_fold.sh 3 0 > logs/cv_sap_fold3.log 2>&1 &
nohup bash scripts/run_cv_sap_fold.sh 4 1 > logs/cv_sap_fold4.log 2>&1 &
wait
echo "## CV-SAP DONE $(date)" >> logs/cv_sap.log
touch logs/cv_sap_done.flag
