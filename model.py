import torch
from torch import nn
import torch.nn.functional as F

class TargetSelectionModel(nn.Module):
    def __init__(self):
        super(TargetSelectionModel, self).__init__()
        self.dense1 = nn.Linear(32, 64)
        self.dense2 = nn.Linear(64, 128)
        self.dense3 = nn.Linear(128, 64)
        self.dense4 = nn.Linear(64, 8)

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)

    def loss(self, pred_y_distribution, y):
        # predictions range from 0 to 1, 1 is the models prediction and gives 0 loss if it is correct
        normalized_pred_y_distribution = self.normalize(pred_y_distribution)
        loss = 1 - normalized_pred_y_distribution[y]
        print(loss)
        return loss
    
    # Normalize tensor to be in range 0-1
    def normalize(self, x):
        normalized_output = []
        x_max = x.max()
        x_min = x.min()
        for i, value in enumerate(x.detach().numpy()):
            normalized_value = (value-x_min)/(x_max-x_min)
            normalized_output.append(normalized_value)
        return torch.tensor(normalized_output, requires_grad=True)
