#!/bin/bash
# 5-fold CV launcher: 3 folds in parallel on GPU 0/1/2, then the last 2.
ROOT=/home/server/Documents/yping/LR-SAF-LSD
cd "$ROOT"
rm -f logs/cv_done.flag
echo "######## CV START $(date) ########" > logs/cv.log

nohup bash scripts/run_cv_fold.sh 0 0 > logs/cv_fold0.log 2>&1 &
nohup bash scripts/run_cv_fold.sh 1 1 > logs/cv_fold1.log 2>&1 &
nohup bash scripts/run_cv_fold.sh 2 2 > logs/cv_fold2.log 2>&1 &
wait
echo "wave 1 (folds 0,1,2) done $(date)" >> logs/cv.log

nohup bash scripts/run_cv_fold.sh 3 0 > logs/cv_fold3.log 2>&1 &
nohup bash scripts/run_cv_fold.sh 4 1 > logs/cv_fold4.log 2>&1 &
wait
echo "wave 2 (folds 3,4) done $(date)" >> logs/cv.log

echo "######## CV DONE $(date) ########" >> logs/cv.log
touch logs/cv_done.flag
