from collections import OrderedDict

import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearModel, self).__init__()
        
        self.model = nn.Sequential(
            OrderedDict([
                ("Linear", nn.Linear(in_dim, out_dim))
            ])
        )

    def forward(self, x):
        return self.model(x)


class SmallModel(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(512,)):
        super(SmallModel, self).__init__()

        self.model = nn.Sequential(
            OrderedDict([
                ("Linear1", nn.Linear(in_dim, hidden[0])),
                ("BatchNorm1", nn.BatchNorm1d(hidden[0])),
                ("ReLU1", nn.ELU(inplace=True)),
                ("Linear2", nn.Linear(hidden[0], out_dim))
            ])
        )

    def forward(self, x):
        return self.model(x)


class MediumModel(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(512, 256)):
        super(MediumModel, self).__init__()

        self.model = nn.Sequential(
            OrderedDict([
                ("Linear1", nn.Linear(in_dim, hidden[0])),
                ("BatchNorm1", nn.BatchNorm1d(hidden[0])),
                ("ReLU1", nn.ELU(inplace=True)),
                ("Linear2", nn.Linear(hidden[0], hidden[1])),
                ("BatchNorm2", nn.BatchNorm1d(hidden[1])),
                ("ReLU2", nn.ELU(inplace=True)),
                ("Linear3", nn.Linear(hidden[1], out_dim))
            ])
        )

    def forward(self, x):
        return self.model(x)


class ResModel(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(512,)):
        super(ResModel, self).__init__()

        self.linear_in = nn.Linear(in_dim, hidden[0]) 
        self.hidden = nn.Linear(hidden[0], hidden[0])
        self.linear_out = nn.Linear(hidden[0], out_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        
        x = self.linear_in(x)
        x = self.activation(x)

        res = x
        x = self.hidden(x)
        x += res
        x = self.activation(x)

        res = x
        x = self.hidden(x)
        x += res
        x = self.activation(x)

        x = self.linear_out(x)

        return x


class DeepModel(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 512, 1024)):
        super(DeepModel, self).__init__()

        self.model = nn.Sequential(
            OrderedDict([
                ("Linear1", nn.Linear(in_dim, hidden[0])),
                ("BatchNorm1", nn.BatchNorm1d(hidden[0])),
                ("ReLU1", nn.ELU(inplace=True)),
                ("Linear2", nn.Linear(hidden[0], hidden[1])),
                ("BatchNorm2", nn.BatchNorm1d(hidden[1])),
                ("ReLU2", nn.ELU(inplace=True)),
                ("Linear3", nn.Linear(hidden[1], hidden[2])),
                ("BatchNorm3", nn.BatchNorm1d(hidden[2])),
                ("ReLU3", nn.ELU(inplace=True)),
                ("Linear4", nn.Linear(hidden[2], out_dim)),
            ])
        )

    def forward(self, x):
        return self.model(x)


class NewDeepModel(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(300, 300, 300, 300)):
        super(NewDeepModel, self).__init__()

        self.model = nn.Sequential(
            OrderedDict([
                ("Linear1", nn.Linear(in_dim, hidden[0])),
                ("BatchNorm1", nn.BatchNorm1d(hidden[0])),
                ("ReLU1", nn.ELU(inplace=True)),
                ("Linear2", nn.Linear(hidden[0], hidden[1])),
                ("BatchNorm1", nn.BatchNorm1d(hidden[1])),
                ("ReLU2", nn.ELU(inplace=True)),
                ("Linear3", nn.Linear(hidden[1], hidden[2])),
                ("BatchNorm1", nn.BatchNorm1d(hidden[2])),
                ("ReLU3", nn.ELU(inplace=True)),
                ("Linear4", nn.Linear(hidden[2], hidden[3])),
                ("BatchNorm1", nn.BatchNorm1d(hidden[3])),
                ("ReLU4", nn.ELU(inplace=True)),
                ("Linear5", nn.Linear(hidden[3], out_dim))
            ])
        )

    def forward(self, x):
        return self.model(x)
