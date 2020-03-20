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

        params = []
        for card in cardinality:
            parameter = torch.nn.Parameter(data=torch.ones(card), requires_grad=True)
            params.append(parameter)

        self.params = nn.ParameterList(params).cuda()

    def forward(self, inputs_idx):
        weights = []
        for i, example_label_idx in enumerate(inputs_idx):
            example_label_weight = self.params[example_label_idx]
            example_label_weight_softmax = torch.softmax(example_label_weight, dim=0)
            weights.append(example_label_weight_softmax)

        return weights

    def update_last_used_weights(self, grads, lr):
        for param, grad in zip(self.params, grads):
            if grad is not None:
                param.data = param.data - lr * grad


if __name__ == '__main__':
    hloss = HLoss()

    a = torch.ones((1, 4))
    b = torch.tensor([[1., 2., 3, 4]])

    print(F.softmax(a, dim=1), F.softmax(b, dim=1))

    print(hloss(a))
    print(hloss(b))
