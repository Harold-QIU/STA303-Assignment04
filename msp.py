from common.dataset import build_dataset
from torchcp.utils import fix_randomness
import torch
import torchvision
from tqdm import tqdm

fix_randomness(seed=0)

#######################################
# Loading ImageNet dataset and a pytorch model
#######################################
model_name = 'ResNet101'
model = torchvision.models.resnet101(weights="IMAGENET1K_V1", progress=True)
model_device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
model.to(model_device)

dataset = build_dataset('imagenet')

cal_dataset, test_dataset = torch.utils.data.random_split(dataset, [25000, 25000])
cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=1024, shuffle=False, pin_memory=True, num_workers=8)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, pin_memory=True, num_workers=8)

#######################################
# Evaluation of ID
####################################### 
# Calc the MSP score on ID dataset: return the max probability of prediction
model.eval()
id_msp_list = []
with torch.no_grad():
    for examples in tqdm(test_data_loader):
        data, target = examples[0].to(model_device), examples[1].to(model_device)
        output = model(data)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        max_probs, predicted_labels = torch.max(probabilities, 1)
        id_msp_list.extend(max_probs.tolist())

    id_msp_list = torch.tensor(id_msp_list)


#######################################
# Evaluation of OOD
#######################################
from torchmetrics import AUROC

def eval_auroc(dataset_name):
    # Calculate the distribution of OOD dataset
    ood_dataset = build_dataset(dataset_name)
    ood_data_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=1024, shuffle=False, pin_memory=True, num_workers=8)

    ood_msp_list = []
    with torch.no_grad():
        for examples in tqdm(ood_data_loader):
            data, target = examples[0].to(model_device), examples[1].to(model_device)
            output = model(data)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            max_probs, predicted_labels = torch.max(probabilities, 1)
            ood_msp_list.extend(max_probs.tolist())

    ood_msp_list = torch.tensor(ood_msp_list)

    auroc = AUROC(task="binary")

    preds = torch.cat([id_msp_list, ood_msp_list])
    preds = (preds - preds.min()) / (preds.max() - preds.min()) # Min-max normalization
    id_target = torch.tensor([1 for i in id_msp_list])
    ood_target = torch.tensor([0 for i in ood_msp_list])
    targets = torch.cat([id_target, ood_target])
    auroc_score = auroc(preds, targets)

    print(f"{dataset_name}: The AUROC is {auroc_score}")

for dataset_name in ['iNaturalist', 'SUN', 'Textures', 'Places']:
    eval_auroc(dataset_name)