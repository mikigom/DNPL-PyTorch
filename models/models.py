from collections import OrderedDict

import torch
import torch.nn as nn

from .meta_module import MetaModule
from .meta_layers import MetaLinear, MetaBatchNorm1d


class DeepModel(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(512, 512)):
        super(DeepModel, self).__init__()
        if hidden is None:
            hidden = (512, 1024)

        self.model = nn.Sequential(
            OrderedDict([
                ("Linear1", nn.Linear(in_dim, hidden[0])),
                ("BatchNorm1", nn.BatchNorm1d(hidden[0])),
                ("ReLU1", nn.ReLU(inplace=True)),
                ("Linear2", nn.Linear(hidden[0], hidden[1])),
                ("BatchNorm2", nn.BatchNorm1d(hidden[1])),
                ("ReLU2", nn.ReLU(inplace=True)),
                ("Dropout", nn.Dropout(0.2)),
                ("Linear3", nn.Linear(hidden[1], out_dim)),
            ])
        )

    def forward(self, x):
        return self.model(x)


class DeepModelTransition(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(512, 512)):
        super(DeepModelTransition, self).__init__()
        if hidden is None:
            hidden = (512, 1024)

        self.model = nn.Sequential(
            OrderedDict([
                ("Linear1", nn.Linear(in_dim, hidden[0])),
                ("BatchNorm1", nn.BatchNorm1d(hidden[0])),
                ("ReLU1", nn.ReLU(inplace=True)),
                ("Linear2", nn.Linear(hidden[0], hidden[1])),
                ("BatchNorm2", nn.BatchNorm1d(hidden[1])),
                ("ReLU2", nn.ReLU(inplace=True)),
                ("Dropout", nn.Dropout(0.2)),
                ("Linear3", nn.Linear(hidden[1], out_dim)),
            ])
        )

        transition_matrix = torch.zeros(out_dim, out_dim)
        self.transition_matrix = torch.nn.Parameter(transition_matrix, requires_grad=True)
        for c in range(out_dim):
            self.transition_matrix[c, c] = 1.
            self.transition_matrix[c, c] = self.transition_matrix[c, c].detach()

        diagonal_idx = torch.diag(torch.ones(out_dim)).double()
        self.diagonal_idx = torch.DoubleTensor(diagonal_idx).cuda().nonzero(as_tuple=True)
        non_diagonal_idx = 1 - torch.diag(torch.ones(out_dim)).double()
        self.non_diagonal_idx = torch.DoubleTensor(non_diagonal_idx).cuda().nonzero(as_tuple=True)

    def forward(self, x):
        softmax = torch.softmax(self.model(x), dim=1)

        self.transition_matrix[self.diagonal_idx] = 1.
        self.transition_matrix[self.non_diagonal_idx] = torch.sigmoid(self.transition_matrix[self.non_diagonal_idx])
        return softmax, softmax[:, None, :].bmm(self.transition_matrix.repeat((softmax.size(0), 1, 1)))


class NewDeepModel(MetaModule):
    def __init__(self, in_dim, out_dim, hidden=(128, 256, 512, 512)):
        super(NewDeepModel, self).__init__()
        if hidden is None:
            hidden = (128, 256, 512, 512)

        self.model = nn.Sequential(
            OrderedDict([
                ("Linear1", MetaLinear(in_dim, hidden[0])),
                ("BatchNorm1", MetaBatchNorm1d(hidden[0])),
                ("ReLU1", nn.ReLU(inplace=True)),
                ("Linear2", MetaLinear(hidden[0], hidden[1])),
                ("BatchNorm2", MetaBatchNorm1d(hidden[1])),
                ("ReLU2", nn.ReLU(inplace=True)),
                ("Linear3", MetaLinear(hidden[1], hidden[2])),
                ("BatchNorm3", MetaBatchNorm1d(hidden[2])),
                ("ReLU3", nn.ReLU(inplace=True)),
                ("Linear4", MetaLinear(hidden[2], hidden[3])),
                ("BatchNorm4", MetaBatchNorm1d(hidden[3])),
                ("ReLU4", nn.ReLU(inplace=True)),
                ("Linear5", MetaLinear(hidden[3], out_dim)),
            ])
        )

    def forward(self, x):
        return self.model(x)
