import copy

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from models.models import *
from utils import to_torch_var
from yogi.yogi import Yogi


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


def total_loss(model, datasets, data_idx, use_norm, simp_loss):

    model.eval()

    val_datasets = copy.deepcopy(datasets)
    val_datasets.set_mode('custom', data_idx)
    val_dataloader = DataLoader(val_datasets, batch_size=128, num_workers=4, drop_last=False)
    val_data_iterator = iter(val_dataloader)
    
    g_val = 0.
    h_val = 0.
    current_iter = 0

    while True:
        current_iter += 1
        try:
            data_val, y_partial_val, _, idx_val = next(val_data_iterator)
        except StopIteration:
            return g_val/current_iter, h_val/current_iter

        x = to_torch_var(data_val, requires_grad=False).float()
        s = torch.DoubleTensor(y_partial_val).cuda().float()
        if use_norm:
            x = (x - feature_mean) / feature_std
            x[torch.isnan(x)] = 0.

        y_f_hat = model(x)
        s_hat = F.softmax(y_f_hat, dim=1)
        #s_hat = sharpen(s_hat, T=0.4)
        ss_hat = s * s_hat
        dot_product = ss_hat.sum(1)
        dot_product = torch.clamp(dot_product, 0., 1.)
        g = -torch.log(dot_product + 1e-10)
        
        if simp_loss:
            h = -(s_hat * torch.log(s_hat + 1e-10)).sum(1)
        else: 
            ss_hat /= dot_product.view(dot_product.size(0),-1)
            h = -(ss_hat * torch.log(ss_hat + 1e-10)).sum(1)

        g_val += torch.mean(g).data.tolist()
        h_val += torch.mean(h).data.tolist()
 

def main(train_datasets, test_datasets, bs, beta=1., num_epoch=25, use_norm=False, model_name='medium', simp_loss=False, 
        args_etc = {'use_mixup': False, 'alpha': 0.2, 'self_teach': False, 'gamma': 0.999, 'eta': 0.5}):
    
    lamd = 1e-6
    auto_beta = True if beta < 0. else False

    use_mixup = args_etc['use_mixup']
    alpha = args_etc['alpha']
    self_teach = args_etc['self_teach']
    gamma = args_etc['gamma']
    eta = args_etc['eta']

    assert train_datasets.dataset_name == test_datasets.dataset_name

    #train_datasets = copy.deepcopy(datasets)
    #train_datasets.set_mode('custom', train_idx)
    train_dataloader = DataLoader(train_datasets, batch_size=bs, num_workers=8, drop_last=True, shuffle=True)

    #test_datasets = copy.deepcopy(datasets)
    #test_datasets.set_mode('custom', test_idx)
    test_dataloader = DataLoader(test_datasets, batch_size=bs, num_workers=8, drop_last=False)

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
        if model_name == 'linear':
            model_ema = LinearModel(in_dim, out_dim).cuda()
        elif model_name == 'small':
            model_ema = SmallModel(in_dim, out_dim).cuda()
        elif model_name == 'medium':
            model_ema = MediumModel(in_dim, out_dim).cuda()
        elif model_name == 'residual':
            model_ema = ResModel(in_dim, out_dim).cuda()
        elif model_name == 'deep':
            model_ema = DeepModel(in_dim, out_dim).cuda()
        elif model_name == 'newdeep':
            model_ema = NewDeepModel(in_dim, out_dim).cuda()
        elif model_name == 'convnet':
            model_ema = ConvNet(in_dim, out_dim).cuda()

        for param in model_ema.parameters():
            param.detach_()

    opt = Yogi(model.parameters(), lr=1e-3)

    train_data_iterator = iter(train_dataloader)
    current_iter = 0
    total_iter = 0
    current_epoch = 0

    g_val = 0.
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

            model.eval()

            is_correct = []
            for X, y_partial, y, idx in test_dataloader:
                x = to_torch_var(X, requires_grad=False).float()
                if use_norm:
                    x = (x - feature_mean) * inv_feature_std
                y = to_torch_var(y, requires_grad=False).long()
                y = torch.argmax(y, dim=1)

                s_bar = model(x)
                y_bar = torch.softmax(s_bar, dim=1)
                y_bar = torch.argmax(y_bar, dim=1)
                is_correct.append(y_bar == y)

            is_correct = torch.cat(is_correct, dim=0)
            acc = torch.mean(is_correct.float()).detach().cpu().numpy()

            g_val /= current_iter
            h_val /= current_iter
            #reg_val = reg.data.tolist()
            #g_val, h_val = total_loss(model, datasets, train_idx, use_norm, simp_loss)
            print("Epoch [{}], g:{:.2e}, h:{:.2e}, acc:{:.2e}".format(current_epoch+1, g_val, h_val, acc))
            #print("Epoch [{}], g:{:.2e}, h:{:.2e}, w:{:.2e}".format(current_epoch+1, g_val, h_val, reg_val))
            train_data_iterator = iter(train_dataloader)
            data_train, y_partial_train, _, idx_train = next(train_data_iterator)
            current_epoch += 1
            current_iter = 0
            g_val = 0.
            h_val = 0.

            model.train()

        if current_epoch == num_epoch // 2 and auto_beta:
            beta = 1.

        if current_epoch == num_epoch:
            break

        x = to_torch_var(data_train, requires_grad=False).float()
        s = torch.DoubleTensor(y_partial_train).cuda().float()
        if use_norm:
            x = (x - feature_mean) * inv_feature_std
        if use_mixup:
            s /= s.sum(1).view(-1,1).expand(-1, s.size(1))
            x, s, lamb, indices = mixup(x, s, alpha)
        # candidate_train_idx = torch.DoubleTensor(y_partial_train).cuda().nonzero(as_tuple=True)

        # Line 12
        y_f_hat = model(x)
        s_hat = F.softmax(y_f_hat, dim=1)
        #s_hat = sharpen(s_hat, T=0.4)
        ss_hat = s * s_hat
        dot_product = ss_hat.sum(1)
        dot_product = torch.clamp(dot_product, 0., 1.)
        g = -torch.log(dot_product + 1e-10)
        
        if simp_loss:
            h = -(s_hat * torch.log(s_hat + 1e-10)).sum(1)

            L = torch.mean(g) + beta * torch.mean(h)
        else: 
            ss_hat /= dot_product.view(dot_product.size(0),-1)
            h = -(ss_hat * torch.log(ss_hat + 1e-10)).sum(1)
            
            L = torch.mean(g) + beta * torch.mean(h)

        if self_teach:
            y_f_ema = model_ema(x)
            s_ema = F.softmax(y_f_ema, dim=1)
            #s_ema = sharpen(s_ema, T=0.4)
            s_ema = torch.autograd.Variable(s_ema.detach().data, requires_grad=False)
            k = (s_hat * ( torch.log(s_hat + 1e-10) - torch.log(s_ema + 1e-10) )).sum(1)
            L += eta * torch.mean(k)

        g_val += torch.mean(g).data.tolist()
        h_val += torch.mean(h).data.tolist()

        #reg = 0
        #for param in model.parameters():
        #    reg += torch.norm(param, 1)
        #L += 0.01*reg

        if torch.isnan(L).any():
            print("Warning: NaN Loss")
            break

        # Line 13-14
        opt.zero_grad()
        L.backward()
        opt.step()
    
        if self_teach:
            update_model_ema(model, model_ema, gamma, total_iter)

    model.eval()

    is_correct = []
    if train_datasets.dataset_name == "fgnet":
        is_mae3 = []
        is_mae5 = []
    for X, y_partial, y, idx in test_dataloader:
        x = to_torch_var(X, requires_grad=False).float()
        if use_norm:
            x = (x - feature_mean) * inv_feature_std
        y = to_torch_var(y, requires_grad=False).long()
        y = torch.argmax(y, dim=1)

        s_bar = model(x)
        y_bar = torch.softmax(s_bar, dim=1)
        y_bar = torch.argmax(y_bar, dim=1)
        is_correct.append(y_bar == y)
        if train_datasets.dataset_name == "fgnet":
            is_mae3.append(torch.abs(y_bar - y) <= 3)
            is_mae5.append(torch.abs(y_bar - y) <= 5)

    is_correct = torch.cat(is_correct, dim=0)
    acc = torch.mean(is_correct.float()).detach().cpu().numpy()

    if train_datasets.dataset_name != "fgnet":
        print("%s" % acc)
        return acc
    else:
        is_mae3 = torch.cat(is_mae3, dim=0)
        is_mae5 = torch.cat(is_mae5, dim=0)
        mae3 = torch.mean(is_mae3.float()).detach().cpu().numpy()
        mae5 = torch.mean(is_mae5.float()).detach().cpu().numpy()
        print("%s / %s / %s" % (acc, mae3, mae5))
        return acc, mae3, mae5

