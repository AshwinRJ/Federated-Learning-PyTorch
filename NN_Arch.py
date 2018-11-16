import torch
import torch.nn as nn
import torch.nn.Functional as F
# MLP Arch with 1 Hidden layer


class MLP(nn.Module):
    def __init__(self, input_dim, hidden, out_dim):

        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        return self.softmax(x)
