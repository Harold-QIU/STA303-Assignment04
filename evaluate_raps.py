from torchcp.classification.scores import RAPS
from torchcp.classification.predictors import SplitPredictor, ClassWisePredictor, ClusterPredictor
from common.dataset import build_dataset
from common.utils import evaluate_plus
from torchcp.utils import fix_randomness
import torch
import torchvision
import numpy as np
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--penalty', type=float, default="1")
argparser.add_argument('--predictors', type=str, default="SplitPredictor")
argparser.add_argument('--kreg', type=int, default="0")
args = argparser.parse_args()

fix_randomness(seed=0)

#######################################
# Loading ImageNet dataset and a pytorch model
#######################################
model_name = 'ResNet101'
model = torchvision.models.resnet101(weights="IMAGENET1K_V1", progress=True)
model_device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model.to(model_device)

dataset = build_dataset('imagenet')

cal_dataset, test_dataset = torch.utils.data.random_split(dataset, [25000, 25000])
cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=1024, shuffle=False, pin_memory=True, num_workers=8)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, pin_memory=True, num_workers=8)

#######################################
# A standard process of conformal prediction
#######################################    
alpha = 0.1
print(f"Experiment--Data : ImageNet, Model : {model_name}, Score : RAPS, Predictor : {args.predictors}, Alpha : {alpha}")
print(f"RAPS setting: penalty = {args.penalty}, kreg = {args.kreg}")

score_function = RAPS(penalty=args.penalty, kreg=args.kreg)

if args.predictors == "SplitPredictor":
    predictor = SplitPredictor(score_function, model)
elif args.predictors == "ClassWisePredictor":
    predictor = ClassWisePredictor(score_function, model)
elif args.predictors == "ClusterPredictor":
    predictor = ClusterPredictor(score_function, model)
else:
    raise ValueError("Invalid predictor.")

print(f"The size of calibration set is {len(cal_dataset)}.")
predictor.calibrate(cal_data_loader, alpha)

#######################################
# Evaluation of conformal prediction
####################################### 
# Calc the distribution of size on ID dataset
id_result_dict = evaluate_plus(predictor, test_data_loader)
id_size_list = torch.tensor(id_result_dict['All_size'])
print(f"In-distribution Average_size: {id_result_dict['Average_size']}, Converge_rate {id_result_dict['Coverage_rate']}")

def eval_ood(dataset_name):
    ood_dataset = build_dataset(dataset_name)
    ood_data_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=1024, shuffle=False, pin_memory=True, num_workers=8)
    return evaluate_plus(predictor, ood_data_loader)


#######################################
# Evaluation of OOD
#######################################
from torchmetrics import AUROC

def eval_auroc(dataset_name):
    # Calculate the distribution of OOD dataset
    ood_result_dict = eval_ood(dataset_name)
    ood_size_list = torch.tensor(ood_result_dict['All_size'])

    auroc = AUROC(task="binary")

    preds = torch.cat([id_size_list, ood_size_list])
    preds = (preds - preds.min()) / (preds.max() - preds.min()) # Min-max normalization
    id_target = torch.tensor([0 for i in id_size_list])
    ood_target = torch.tensor([1 for i in ood_size_list])
    targets = torch.cat([id_target, ood_target])
    auroc_score = auroc(preds, targets)

    print(f"{dataset_name}: The AUROC is {auroc_score}")

for dataset_name in ['iNaturalist', 'SUN', 'Textures', 'Places']:
    eval_auroc(dataset_name)