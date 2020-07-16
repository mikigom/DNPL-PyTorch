import copy

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from datasets.uci_datasets import UCI_Datasets
from models.models import MediumModel
from utils import to_torch_var

LEARNING_RATE = 3e-4


def main(dataset_name, r=1, p=0.2, eps=None, beta=0.01, lamd=1e-3, num_epoch=20, use_norm=False, mod_loss = True):
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
    model = MediumModel(in_dim, out_dim, hidden=(512, 256)).cuda()

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=lamd)

    train_data_iterator = iter(train_dataloader)
    current_iter = 0.
    current_epoch = 0
    while True:
        current_iter += 1

        model.train()
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
        if use_norm:
            x = (x - feature_mean) / feature_std
        # candidate_train_idx = torch.DoubleTensor(y_partial_train).cuda().nonzero(as_tuple=True)

        # Line 12
        y_f_hat = model(x)
        
        if not mod_loss:
            s_bar = F.softmax(y_f_hat, dim=1)
            h = -(s_bar * torch.log(s_bar + 1e-7)).sum(1)

            s_bar = s_bar.view(s_bar.size(0), 1, -1)

            dot_product = torch.bmm(s_bar, s.view(s.size(0), -1, 1))
            dot_product = torch.clamp(dot_product, 0., 1.)
            g = -torch.log(dot_product + 1e-7)

            L = torch.mean(g) + beta * torch.mean(h)
        else: 
            s_bar = F.softmax(y_f_hat, dim=1)
            ss_bar = s * s_bar            
            dot_product = ss_bar.sum(1)
            dot_product = torch.clamp(dot_product, 0., 1.)

            ss_bar /= dot_product.view(dot_product.size(0),-1)
            h = -(ss_bar * torch.log(ss_bar + 1e-7)).sum(1)
            g = -torch.log(dot_product + 1e-7)
            
            L = torch.mean(g) + beta * torch.mean(h)

        # Line 13-14
        opt.zero_grad()
        L.backward()
        opt.step()

    model.eval()

    is_correct = []
    for X, y_partial, y, idx in test_dataloader:
        x = to_torch_var(X, requires_grad=False).float()
        if use_norm:
            x = (x - feature_mean) / feature_std
        y = to_torch_var(y, requires_grad=False).long()
        y = torch.argmax(y, dim=1)

        s_bar = model(x)
        y_bar = torch.softmax(s_bar, dim=1)
        y_bar = torch.argmax(y_bar, dim=1)
        is_correct.append(y_bar == y)

    is_correct = torch.cat(is_correct, dim=0)
    acc = torch.mean(is_correct.float()).detach().cpu().numpy()

    print("%s" % acc)
    return acc


if __name__ == '__main__':
    main("segment", r=1, p=0.7, beta=0., lamd=0., num_epoch=20, use_norm=False)
