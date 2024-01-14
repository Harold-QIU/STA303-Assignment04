import torch
import torch.nn as nn
from tqdm import tqdm



def build_regression_model(model_name="NonLinearNet"):
    if model_name == "NonLinearNet":
        class NonLinearNet(nn.Module):
            def __init__(self, in_shape, out_shape, hidden_size, dropout):
                super(NonLinearNet, self).__init__()
                self.hidden_size = hidden_size
                self.in_shape = in_shape
                self.out_shape = out_shape
                self.dropout = dropout
                self.base_model = nn.Sequential(
                    nn.Linear(self.in_shape, self.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_size, self.out_shape),
                )

            def forward(self, x):
                return self.base_model(x)

        return NonLinearNet
    else:
        raise NotImplementedError


def evaluate_plus(self, val_dataloader):
        prediction_sets = []
        labels_list = []
        prediction_size = []
        with torch.no_grad():
            for examples in tqdm(val_dataloader):
                tmp_x, tmp_label = examples[0].to(self._device), examples[1].to(self._device)
                prediction_sets_batch = self.predict(tmp_x)
                prediction_sets.extend(prediction_sets_batch)
                labels_list.append(tmp_label)
        val_labels = torch.cat(labels_list)

        res_dict = {}
        res_dict["Coverage_rate"] = self._metric('coverage_rate')(prediction_sets, val_labels)
        res_dict["Average_size"] = self._metric('average_size')(prediction_sets, val_labels)
        res_dict["All_size"] = all_size(prediction_sets, val_labels)
        return res_dict

def all_size(prediction_sets, labels):
    all_size = []
    for index, ele in enumerate(prediction_sets):
        all_size.append(len(ele))
    return all_size
