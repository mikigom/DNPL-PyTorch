from collections import OrderedDict

import torch.nn as nn


class DeepModel(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 512)):
        super(DeepModel, self).__init__()
        if hidden is None:
            hidden = (256, 512)

        self.model = nn.Sequential(
            OrderedDict([
                ("Linear1", nn.Linear(in_dim, hidden[0])),
                ("BatchNorm1", nn.BatchNorm1d(hidden[0])),
                ("ReLU1", nn.ReLU(inplace=True)),
                ("Linear2", nn.Linear(hidden[0], hidden[1])),
                ("BatchNorm2", nn.BatchNorm1d(hidden[1])),
                ("ReLU2", nn.ReLU(inplace=True)),
                ("Linear3", nn.Linear(hidden[1], out_dim)),
            ])
        )

    def forward(self, x):
        return x
