import copy

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from datasets.uci_datasets import UCI_Datasets
from models.models import MediumModel, ResModel, DeepModel
from utils import to_torch_var

LEARNING_RATE = 3e-4

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


def main(dataset_name, r=1, p=0.2, eps=None, beta=1., lamd=1e-3, num_epoch=25, use_norm = False, 
        args_etc = {'mod_loss': True, 'use_mixup': False, 'alpha': 0.4, 'self_teach': False, 'gamma': 0.999, 'eta': 0.5}):
    
    mod_loss = args_etc['mod_loss']
    use_mixup = args_etc['use_mixup']
    alpha = args_etc['alpha']
    self_teach = args_etc['self_teach']
    gamma = args_etc['gamma']
    eta = args_etc['eta']

    datasets = UCI_Datasets(dataset_name, r=r, p=p, eps=eps, test_fold=10, val_fold=0)

    train_datasets = copy.deepcopy(datasets)
    train_datasets.set_mode('train')
    train_dataloader = DataLoader(train_datasets, batch_size=32, num_workers=4, drop_last=True, shuffle=True)

    test_datasets = copy.deepcopy(datasets)
    test_datasets.set_mode('test')
    test_dataloader = DataLoader(test_datasets, batch_size=16, num_workers=4, drop_last=False)

    feature_mean = torch.Tensor(train_datasets.X.mean(0)[np.newaxis]).cuda()
    feature_std = torch.Tensor(train_datasets.X.std(0)[np.newaxis]).cuda()

    in_dim, out_dim = datasets.get_dims
    model = MediumModel(in_dim, out_dim, hidden = (512, 256)).cuda()
    #model = ResModel(in_dim, out_dim, hidden = (512,)).cuda()
    #model = DeepModel(in_dim, out_dim, hidden = (256, 512, 1024)).cuda()

    if self_teach:
        model_ema = MediumModel(in_dim, out_dim, hidden = (512, 256)).cuda()
        #model_ema = ResModel(in_dim, out_dim, hidden = (512,)).cuda()
        #model_ema = DeepModel(in_dim, out_dim, hidden = (256, 512, 1024)).cuda()
        for param in model_ema.parameters():
            param.detach_()

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=lamd)

    train_data_iterator = iter(train_dataloader)
    current_iter = 0
    current_epoch = 0
    total_step = 0

    while True:
        current_iter += 1

        model.train()
        if self_teach:
            model_ema.train()
        # Line 2) Get Batch from Training Dataset
        # Expected Question: "Why is this part so ugly?"
        # Answer: "Please refer https://github.com/pytorch/pytorch/issues/1917 ."
        try:
            data_train, y_partial_train, _, idx_train = next(train_data_iterator)
        except StopIteration:
            train_data_iterator = iter(train_dataloader)
            data_train, y_partial_train, _, idx_train = next(train_data_iterator)
            current_epoch += 1
            current_iter = 0
            print("Epoch [%s]" % current_epoch)

        if current_epoch == num_epoch:
            break

        x = to_torch_var(data_train, requires_grad=False).float()
        s = torch.DoubleTensor(y_partial_train).cuda().float()
        #s /= s.sum(1).view(-1,1).expand(-1, s.size(1))
        
        if use_norm:
            x = (x - feature_mean) / feature_std
            x[torch.isnan(x)] = 0.
        if use_mixup:
            x, s, lamb, indices = mixup(x, s, alpha)
        # candidate_train_idx = torch.DoubleTensor(y_partial_train).cuda().nonzero(as_tuple=True)

        # Line 12
        y_f_hat = model(x)
        s_hat = F.softmax(y_f_hat, dim=1)
        #s_hat = sharpen(s_hat, T=0.4)
        ss_hat = s * s_hat
        dot_product = ss_hat.sum(1)
        dot_product = torch.clamp(dot_product, 0., 1.)
        g = -torch.log(dot_product + 1e-7)
        
        if mod_loss:
            h = -(s_hat * torch.log(s_hat + 1e-7)).sum(1)

            L = torch.mean(g) + beta * torch.mean(h)
        else: 
            ss_hat /= dot_product.view(dot_product.size(0),-1)
            h = -(ss_hat * torch.log(ss_hat + 1e-7)).sum(1)
            
            L = torch.mean(g) + beta * torch.mean(h)

        if self_teach:
            y_f_ema = model_ema(x)
            s_ema = F.softmax(y_f_ema, dim=1)
            s_ema = sharpen(s_ema, T=0.4)
            s_ema = torch.autograd.Variable(s_ema.detach().data, requires_grad=False)
            k = (s_hat * ( torch.log(s_hat + 1e-7) - torch.log(s_ema + 1e-7) )).sum(1)
            L += eta * torch.mean(k)

        if torch.isnan(L).any():
            print("Warning: NaN Loss")
            break

        # Line 13-14
        opt.zero_grad()
        L.backward()
        opt.step()
        total_step += 1
        if self_teach:
            update_model_ema(model, model_ema, gamma, total_step)

    model.eval()

    is_correct = []
    for X, y_partial, y, idx in test_dataloader:
        x = to_torch_var(X, requires_grad=False).float()
        if use_norm:
            x = (x - feature_mean) / feature_std
            x[torch.isnan(x)] = 0.
        y = to_torch_var(y, requires_grad=False).long()
        y = torch.argmax(y, dim=1)

        s_hat = model(x)
        y_hat = torch.softmax(s_hat, dim=1)
        y_hat = torch.argmax(y_hat, dim=1)
        is_correct.append(y_hat == y)

    is_correct = torch.cat(is_correct, dim=0)
    acc = torch.mean(is_correct.float()).detach().cpu().numpy()

    del model
    if self_teach:
        del model_ema

    print("%s" % acc)
    return acc


if __name__ == '__main__':
    main("segment", r=1, p=0.7, beta=0., lamd=0., num_epoch=20, use_norm=False)
