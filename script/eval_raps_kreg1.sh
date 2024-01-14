#!/bin/bash

source ~/.bashrc
source /data/software/anaconda3/etc/profile.d/conda.sh
conda activate py3

predictors=(SplitPredictor ClassWisePredictor ClusterPredictor)
penalties=(0.2 0.4 0.6 0.8 1)

for predictor in "${predictors[@]}"
do
    for penalty in "${penalties[@]}"
    do
        echo ==========================
        echo Predictor: $predictor, Score: $penalty
        echo evaluate_raps.py --predictor $predictor --penalty $penalty
        echo ==========================
        python evaluate_raps.py --predictor $predictor --penalty $penalty --kreg 1
    done
done

# nohup bash ./script/eval_raps_kreg1.sh 1>log/raps.out 2>log/raps.err &