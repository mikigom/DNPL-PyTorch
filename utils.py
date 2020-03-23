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


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = torch.log(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


if __name__ == '__main__':
    hloss = HLoss()

    a = torch.ones((1, 4))
    b = torch.tensor([[1., 2., 3, 4]])

    print(F.softmax(a, dim=1), F.softmax(b, dim=1))

    print(hloss(a))
    print(hloss(b))
