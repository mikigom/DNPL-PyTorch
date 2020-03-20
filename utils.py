import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def to_torch_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class ExampleLabelWeights(nn.Module):
    def __init__(self, cardinality):
        super(ExampleLabelWeights, self).__init__()

        self.params = nn.ParameterList([])
        for card in cardinality:
            parameter = torch.nn.Parameter(data=torch.ones(card), requires_grad=True)
            self.params.append(parameter)

        self.params = self.params.cuda()

    def forward(self, losses, inputs_idx):
        if self.training:
            self.lastly_used_params = []

            weights = []
            for example_label_idx in inputs_idx:
                example_label_weight = self.params[example_label_idx]
                self.lastly_used_params.append(self.params[example_label_idx])
                example_label_weight = torch.softmax(example_label_weight, dim=0)
                weights.append(example_label_weight)
            self.weights = torch.cat(weights, dim=0)

            losses = losses * self.weights
            self.lastly_used_params = torch.cat(self.lastly_used_params, dim=0)
            return torch.sum(losses)
        else:
            weights = []
            for example_label_idx in inputs_idx:
                example_label_weight = self.params[example_label_idx]
                example_label_weight = torch.softmax(example_label_weight, dim=0)
                argmax_label = torch.argmax(example_label_weight, dim=0)
                example_label_weight_onehot = torch.zeros_like(example_label_weight)
                example_label_weight_onehot[argmax_label] = 1.

                weights.append(example_label_weight_onehot)

                # Initialize parameter
                self.params[example_label_idx].data = torch.ones_like(self.params[example_label_idx].data)
            weights = torch.cat(weights, dim=0)

            losses = losses * weights
            return torch.sum(losses)

    def update_last_used_weights(self, grads, lr):
        for param, grad in zip(self.params, grads):
            if grad is not None:
                param.data = param.data - lr * grad
