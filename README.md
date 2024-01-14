# CPOOD

## Abstract

Conformal prediction is a framework for producing set-valued predictions that
have a formal guarantee of coverage probability. And OOD detection is a task to
detect out-of-distribution examples. One way to address the challenge of OOD
detection is measure how unusual a new example is to the model. In this paper,
we propose a new method CPOOD for OOD detection based on conformal predic-
tion. The main idea of our method is to use the size of the prediction set as a metric
to detect out-of-distribution examples. We introduce TorchCP, a Python toolbox
for conformal prediction research on deep learning models, using PyTorch. We
conduct experiments and the results show that our CPOOD method has a better performance than the baseline method Maximum Softmax Probability (MSP)
(Hendrycks & Gimpel, 2017). 

## Setup

If you are using conda, you can create a new environment with the following
command (we suggest the use of miniconda): 
```bash
conda create -n cpood python=3.11
```

Activate the environment:

```bash
conda activate cpood
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

You can go the the `scripts` folder to check the parameters for each script. The
following is an examplary script:

```bash
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
```

If you want to run the experiments directly, you can run the following commands:

```bash
cd <root path of CPOOD>
```

Evaluate the CPOOD method with THR and APS scores:

```bash
nohup bash ./script/eval_thr_aps.sh 1>log/thr_aps.out 2>log/thr_aps.err &
```

Evaluate the CPOOD method with RAPS score:

```bash
nohup bash ./script/eval_raps_kreg1.sh 1>log/raps.out 2>log/raps.err &
nohup bash ./script/eval_raps_kreg0.sh 1>log/raps_kreg0.out 2>log/raps_kreg0.err &
```

After executing the above commands, you can check the results in the `log` folder.

**Please pay attention that you must modify the `model_device` of each python script
according to you own device condition.**

## Reference bib

```bibtex
@misc{huang2023conformal,
  title={Conformal Prediction for Deep Classifier via Label Ranking}, 
  author={Jianguo Huang and Huajun Xi and Linjun Zhang and Huaxiu Yao and Yue Qiu and Hongxin Wei},
  year={2023},
  eprint={2310.06430},
  archivePrefix={arXiv},
  primaryClass={cs.LG}}

@article{huang2023torchcp,
  title={Conformal Prediction for Deep Classifier via Label Ranking},
  author={Huang, Jianguo and Xi, Huajun and Zhang, Linjun and Yao, Huaxiu and Qiu, Yue and Wei, Hongxin},
  journal={arXiv preprint arXiv:2310.06430},
  year={2023}
}

@article{ding2019advertorch,
  title={{AdverTorch} v0.1: An Adversarial Robustness Toolbox based on PyTorch},
  author={Ding, Gavin Weiguang and Wang, Luyu and Jin, Xiaomeng},
  journal={arXiv preprint arXiv:1902.07623},
  year={2019}
}

@book{vovk2005algorithmic,
  title={Algorithmic learning in a random world},
  author={Vovk, Vladimir and Gammerman, Alexander and Shafer, Glenn},
  volume={29},
  year={2005},
  publisher={Springer}
}

@inproceedings{ming2022delving,
  title={Delving into Out-of-Distribution Detection with Vision-Language Representations},
  author={Ming, Yifei and Cai, Ziyang and Gu, Jiuxiang and Sun, Yiyou and Li, Wei and Li, Yixuan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

@article{sadinle2019least,
  title={Least ambiguous set-valued classifiers with bounded error levels},
  author={Sadinle, Mauricio and Lei, Jing and Wasserman, Larry},
  journal={Journal of the American Statistical Association},
  volume={114},
  number={525},
  pages={223--234},
  year={2019},
  publisher={Taylor \& Francis}
}

@article{romano2020classification,
  title={Classification with valid and adaptive coverage},
  author={Romano, Yaniv and Sesia, Matteo and Candes, Emmanuel},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={3581--3591},
  year={2020}
}

@article{angelopoulos2020uncertainty,
  title={Uncertainty sets for image classifiers using conformal prediction},
  author={Angelopoulos, Anastasios and Bates, Stephen and Malik, Jitendra and Jordan, Michael I},
  journal={International Conference on Learning Representations},
  year={2020}
}

@inproceedings{shi2013applications,
  title={Applications of class-conditional conformal predictor in multi-class classification},
  author={Shi, Fan and Ong, Cheng Soon and Leckie, Christopher},
  booktitle={2013 12th International Conference on Machine Learning and Applications},
  volume={1},
  pages={235--239},
  year={2013},
  organization={IEEE}
}

@article{lei2018distribution,
  title={Distribution-free predictive inference for regression},
  author={Lei, Jing and G’Sell, Max and Rinaldo, Alessandro and Tibshirani, Ryan J and Wasserman, Larry},
  journal={Journal of the American Statistical Association},
  volume={113},
  number={523},
  pages={1094--1111},
  year={2018},
  publisher={Taylor \& Francis}
}

@article{ding2023classconditional,
  title={Class-Conditional Conformal Prediction with Many Classes},
  author={Ding, Tiffany and Angelopoulos, Anastasios N and Bates, 
          Stephen and Jordan, Michael I and Tibshirani, Ryan J},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}

@article{hendrycks17baseline,
  author    = {Dan Hendrycks and Kevin Gimpel},
  title     = {A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks},
  journal = {Proceedings of International Conference on Learning Representations},
  year = {2017},
}
```
