import torch
import copy
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

def to_torch_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def sharpen(p, T):
    p = torch.pow(p, 1.0/T)
    p /= p.sum(1).view(-1,1).expand(-1, p.size(1))
    return p

def loss_monitor(model, datasets, norm_params=None):

    datasets_copy = copy.deepcopy(datasets)
    model_copy = copy.deepcopy(model)
    dataloader = DataLoader(datasets_copy, batch_size=min((256, len(datasets_copy))),
        num_workers=4, drop_last=False)
    data_iterator = iter(dataloader)
    
    model_copy.eval()

    surrogate_risk_val = 0.
    partial_risk_val = 0.
    zeroone_risk_val = 0.

    current_iter = 0

    is_correct = []
    for data, y_partial, y, idx in data_iterator:
        current_iter += 1
        
        x = to_torch_var(data, requires_grad=False).float()
        s = torch.DoubleTensor(y_partial).cuda().float()
        y = to_torch_var(y, requires_grad=False).long()
        y = torch.argmax(y, dim=1)

        if norm_params is not None:
            feature_mean = norm_params[0]
            inv_feature_std = norm_params[1]
            x = (x - feature_mean) * inv_feature_std
        
        s_hat = model_copy(x)
        s_hat = F.softmax(s_hat, dim=1)
        ss_hat = s * s_hat
        ss_hat_dp = ss_hat.sum(1)
        ss_hat_dp = torch.clamp(ss_hat_dp, 0., 1.)
        l = -torch.log(ss_hat_dp + 1e-10)
        surrogate_risk_val += torch.mean(l).data.tolist()

        y_hat = sharpen(s_hat, .1)
        sy_hat = s * y_hat
        sy_hat_dp = sy_hat.sum(1)
        sy_hat_dp = torch.clamp(sy_hat_dp, 0., 1.)
        partial_risk_val += torch.mean(sy_hat_dp).data.tolist()

        y_hat = torch.argmax(s_hat, dim=1)
        is_correct.append(y_hat == y)

    surrogate_risk_val /= current_iter 
    partial_risk_val /= current_iter
    is_correct = torch.cat(is_correct, dim=0)
    zeroone_risk_val = torch.mean(is_correct.float()).detach().cpu().numpy()

    del model_copy
    del datasets_copy

    return surrogate_risk_val, partial_risk_val, zeroone_risk_val