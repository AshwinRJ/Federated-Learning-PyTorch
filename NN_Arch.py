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

# CNN Arch for MNIST


class CNN_Mnist(nn.Module):
    def __init__(self, args):

        super(CNN_Mnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout_2d = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.relu(x)
        x = F.max_pool2d(nn.Dropout2d(self.conv2(x)), 2)
        x = F.relu(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
