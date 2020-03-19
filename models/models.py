from collections import OrderedDict

import torch.nn as nn

from .meta_module import MetaModule
from .meta_layers import MetaLinear, MetaBatchNorm1d


class DeepModel(MetaModule):
    def __init__(self, in_dim, out_dim, hidden=(256, 512)):
        super(DeepModel, self).__init__()
        if hidden is None:
            hidden = (256, 512)

        self.model = nn.Sequential(
            OrderedDict([
                ("Linear1", MetaLinear(in_dim, hidden[0])),
                ("BatchNorm1", MetaBatchNorm1d(hidden[0])),
                ("ReLU1", nn.ReLU(inplace=True)),
                ("Linear2", MetaLinear(hidden[0], hidden[1])),
                ("BatchNorm2", MetaBatchNorm1d(hidden[1])),
                ("ReLU2", nn.ReLU(inplace=True)),
                ("Linear3", MetaLinear(hidden[1], out_dim)),
            ])
        )

    def forward(self, x):
        return self.model(x)
