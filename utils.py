import torch
import torch.nn as nn
from torch.autograd import Variable


def to_torch_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class ExampleLabelWeights(nn.Module):
    def __init__(self, cardinality):
        super(ExampleLabelWeights, self).__init__()

        self.example_label_weights = []
        for card in cardinality:
            self.example_label_weights.append(torch.nn.Parameter(data=torch.ones(card), requires_grad=True).cuda())

    def forward(self, losses, inputs_idx):
        weights = []
        for example_label_idx in inputs_idx:
            example_label_weight = self.example_label_weights[example_label_idx]
            example_label_weight = torch.softmax(example_label_weight, dim=0)
            weights.append(example_label_weight)
        weights = torch.cat(weights, dim=0)

        losses = losses * weights
        return torch.sum(losses)
