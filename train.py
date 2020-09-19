import copy

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from models.models import *
from utils import to_torch_var
from yogi.yogi import Yogi
from vat import VATLoss


def mixup(x, y, alpha):
    #indices = torch.randperm(x.size(0))
    indices = np.random.choice(x.size(0),x.size(0))
    x2 = x[indices]
    y2 = y[indices]

    lamb = np.random.beta(alpha, alpha)
    x = x * lamb + x2 * (1 - lamb)
    y = y * lamb + y2 * (1 - lamb)

    return x, y, lamb, indices


def sharpen(p, T):
    p = torch.pow(p, 1.0/T)
    p /= p.sum(1).view(-1,1).expand(-1, p.size(1))

    return p


def update_model_ema(model, ema_model, gamma, step):
    # Use the true average until the exponential average is more correct
    # Barrowed from github.com/CuriousAI/mean-teacher/blob/master/pytorch/main.py
    # ArXiv 1703.01780
    gamma = min(1. - 1. / (step + 1), gamma)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(gamma).add_(1. - gamma, param.data)


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
        #s_hat = sharpen(s_hat, .1)
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


def main(train_datasets, test_datasets, bs, beta=1., num_epoch=100, use_norm=False, model_name='medium', 
        args_etc = {'use_mixup': False, 'alpha': 0.2, 'self_teach': False, 'gamma': 0.999, 'eta': 0.5, 'use_vat': True, 'xi': 10.0, 'eps': 1.0, 'ip': 10}):

    monitor = False
    
    #auto_beta = True if beta < 0. else False
    auto_beta = False
    use_mixup = args_etc['use_mixup']
    alpha = args_etc['alpha']
    self_teach = args_etc['self_teach']
    gamma = args_etc['gamma']
    eta = args_etc['eta']
    use_vat = args_etc['use_vat']
    xi = args_etc['xi']
    eps = args_etc['eps']
    ip = args_etc['ip']

    assert train_datasets.dataset_name == test_datasets.dataset_name

    #train_datasets = copy.deepcopy(datasets)
    #train_datasets.set_mode('custom', train_idx)
    train_dataloader = DataLoader(train_datasets, batch_size=min((bs, len(train_datasets))),
            num_workers=4, drop_last=True, shuffle=True)

    #test_datasets = copy.deepcopy(datasets)
    #test_datasets.set_mode('custom', test_idx)
    test_dataloader = DataLoader(test_datasets, batch_size=min((256, len(test_datasets))),
            num_workers=4, drop_last=False)

    if use_norm:
        feature_mean = torch.Tensor(train_datasets.X.mean(0)[np.newaxis]).cuda()
        feature_std = torch.Tensor(train_datasets.X.std(0)[np.newaxis]).cuda()
        inv_feature_std = 1.0 / feature_std
        inv_feature_std[torch.isnan(inv_feature_std)] = 1.

    in_dim, out_dim = train_datasets.get_dims
    if model_name == 'linear':
        model = LinearModel(in_dim, out_dim).cuda()
    elif model_name == 'small':
        model = SmallModel(in_dim, out_dim).cuda()
    elif model_name == 'medium':
        model = MediumModel(in_dim, out_dim).cuda()
    elif model_name == 'residual':
        model = ResModel(in_dim, out_dim).cuda()
    elif model_name == 'deep':
        model = DeepModel(in_dim, out_dim).cuda()
    elif model_name == 'newdeep':
        model = NewDeepModel(in_dim, out_dim).cuda()
    elif model_name == 'convnet':
        model = ConvNet(in_dim, out_dim).cuda()

    if self_teach:
        model_ema = copy.deepcopy(model) 

        for param in model_ema.parameters():
            param.detach_()

    opt = Yogi(model.parameters(), lr=1e-3)
    #opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    #opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_data_iterator = iter(train_dataloader)
    current_iter = 0
    total_iter = 0
    current_epoch = 0

    l_val = 0.
    h_val = 0.
    if auto_beta:
        beta = 0.

    while True:
        current_iter += 1
        total_iter += 1

        model.train()
        if self_teach:
            model_ema.train()
        # Line 2) Get Batch from Training Dataset
        # Expected Question: "Why is this part so ugly?"
        # Answer: "Please refer https://github.com/pytorch/pytorch/issues/1917 ."
        try:
            data_train, y_partial_train, _, idx_train = next(train_data_iterator)
        except StopIteration:
            current_epoch += 1

            #if current_epoch == num_epoch:
            if True:
                model.eval()

                is_correct = []
                for X, y_partial, y, idx in test_dataloader:

                    x = to_torch_var(X, requires_grad=False).float()
                    s = torch.DoubleTensor(y_partial).cuda().float()

                    if use_norm:
                        x = (x - feature_mean) * inv_feature_std
                    y = to_torch_var(y, requires_grad=False).long()
                    y = torch.argmax(y, dim=1)

                    y_hat = model(x)
                    y_hat = torch.softmax(y_hat, dim=1)
                    #y_hat = sharpen(y_hat, .1)

                    is_correct.append(torch.argmax(y_hat, dim=1) == y)

                is_correct = torch.cat(is_correct, dim=0)
                acc = torch.mean(is_correct.float()).detach().cpu().numpy()
                model.train()
            else:
                acc = 0.0

            l_val /= current_iter
            h_val /= current_iter
            #if not monitor and current_epoch % 100 == 0:
            if not monitor:
                print("Epoch [{}], l:{:.2e}, h:{:.2e}, acc:{:.2e}".format(current_epoch, l_val, h_val, acc))
            #else:
            #    sr_tr, pr_tr, zr_tr = loss_monitor(model, train_datasets)
            #    sr_tst, pr_tst, zr_tst = loss_monitor(model, test_datasets)
            #    print(current_epoch, sr_tr, sr_tst, pr_tr, pr_tst, zr_tr, zr_tst) 

            current_iter = 0
            l_val = 0.
            h_val = 0.

            train_data_iterator = iter(train_dataloader)
            data_train, y_partial_train, _, idx_train = next(train_data_iterator)

        if current_epoch == num_epoch // 4 and auto_beta:
            beta = 1.

        if current_epoch == num_epoch:
            #sr_tr, pr_tr, zr_tr = loss_monitor(model, train_datasets)
            #sr_tst, pr_tst, zr_tst = loss_monitor(model, test_datasets)
            #print(current_epoch, sr_tr, sr_tst, pr_tr, pr_tst, zr_tr, zr_tst)
            break

        x = to_torch_var(data_train, requires_grad=False).float()
        s = torch.DoubleTensor(y_partial_train).cuda().float()
        if use_norm:
            x = (x - feature_mean) * inv_feature_std
        if use_mixup:
            #s /= s.sum(1).view(-1,1).expand(-1, s.size(1))
            x, s, lamb, indices = mixup(x, s, alpha)
        if use_vat and beta != .0:
            vat_loss = VATLoss(xi=xi, eps=eps, ip=ip)
            lds = vat_loss(model, x)

        # Line 12
        f = model(x)
        s_hat = F.softmax(f, dim=1)
        ss_hat = s * s_hat
        ss_hat_dp = ss_hat.sum(1)
        ss_hat_dp = torch.clamp(ss_hat_dp, 0., 1.)
        l = -torch.log(ss_hat_dp + 1e-10)
        l_mean = torch.mean(l)
        
        if beta != .0:
            #h = -(s_hat * torch.log(s_hat + 1e-10)).sum(1)
            #h_mean = torch.mean(h)
            L = l_mean + beta * lds
        else:
            L = l_mean

        if self_teach:
            y_f_ema = model_ema(x)
            s_ema = F.softmax(y_f_ema, dim=1)
            s_ema = sharpen(s_ema, T)
            s_ema = torch.autograd.Variable(s_ema.detach().data, requires_grad=False)
            k = (s_hat * ( torch.log(s_hat + 1e-10) - torch.log(s_ema + 1e-10) )).sum(1)
            L += eta * torch.mean(k)

        l_val += l_mean.data.tolist()
        if beta != .0:
            h_val += lds.data.tolist()

        if torch.isnan(L).any():
            print("Warning: NaN Loss")
            break

        # Line 13-14
        opt.zero_grad()
        L.backward()
        opt.step()
    
        if self_teach:
            update_model_ema(model, model_ema, gamma, total_iter)

    return acc
