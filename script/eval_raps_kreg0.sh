#!/bin/bash

source ~/.bashrc
source /data/software/anaconda3/etc/profile.d/conda.sh
conda activate py3

predictors=(SplitPredictor ClassWisePredictor ClusterPredictor)
penalty=0.1

for predictor in "${predictors[@]}"
do
    echo ==========================
    echo Predictor: $predictor, Score: $penalty
    echo evaluate_raps.py --predictor $predictor --penalty $penalty
    echo ==========================
    python evaluate_raps.py --predictor $predictor --penalty $penalty --kreg 0
done

# nohup bash ./script/eval_raps_kreg0.sh 1>log/raps_kreg0.out 2>log/raps_kreg0.err &