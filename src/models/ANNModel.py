import torch
import torch.nn as nn
import torch.nn.functional as F

class SpamHamANNclassify(nn.Module):
    def __init__(self, input_dim: int):
        super(SpamHamANNclassify, self).__init__()
        self.Layer1 = nn.Linear(input_dim, 1) #int(input_dim/32))
        # self.layer2 = nn.Linear(int(input_dim/32), 1)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self,x):
        x = self.Layer1(x)
        # x = self.dropout(x)
        # x = self.layer2(x)
        output = torch.sigmoid(x)
        return output
    
