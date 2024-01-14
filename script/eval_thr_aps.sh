#!/bin/bash

source ~/.bashrc
source /data/software/anaconda3/etc/profile.d/conda.sh
conda activate py3

predictors=(SplitPredictor ClassWisePredictor ClusterPredictor)
scores=(THR APS)

for predictor in "${predictors[@]}"
do
    for score in "${scores[@]}"
    do
        echo ==========================
        echo Predictor: $predictor, Score: $score
        echo evaluate_thr_aps.py --predictor $predictor --score $score
        echo ==========================
        python evaluate_thr_aps.py --predictor $predictor --score $score
    done
done

# nohup bash ./script/eval_thr_aps.sh 1>log/thr_aps.out 2>log/thr_aps.err &