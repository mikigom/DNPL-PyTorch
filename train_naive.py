import copy

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from datasets.datasets import Datasets
from models.models import DeepModel
from utils import to_torch_var

from yogi.yogi import Yogi

LEARNING_RATE = 1e-2


def main(dataset_name, lamd=0.01, num_epoch=20, use_norm=False):
    datasets = Datasets(dataset_name, test_fold=10, val_fold=0)

    train_datasets = copy.deepcopy(datasets)
    train_datasets.set_mode('train')
    train_dataloader = DataLoader(train_datasets, batch_size=128, num_workers=4, drop_last=True, shuffle=True)

    test_datasets = copy.deepcopy(datasets)
    test_datasets.set_mode('test')
    test_dataloader = DataLoader(test_datasets, batch_size=16, num_workers=4, drop_last=False)

    feature_mean = torch.Tensor(train_datasets.X.mean(0)[np.newaxis]).cuda()
    feature_std = torch.Tensor(train_datasets.X.std(0)[np.newaxis]).cuda()

    in_dim, out_dim = datasets.get_dims
    model = DeepModel(in_dim, out_dim).cuda()

    opt = Yogi(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

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
        s_bar = F.softmax(y_f_hat, dim=1)
        h = -(s_bar * torch.log(s_bar + 1e-7)).sum(1)
        h = torch.mean(h)

        s_bar = s_bar.view(s_bar.size(0), 1, -1)

        dot_product = torch.bmm(s_bar, s.view(s.size(0), -1, 1))
        dot_product = torch.clamp(dot_product, 0., 1.)
        g = -torch.log(dot_product + 1e-7)

        L = torch.mean(g) + lamd * h

        # Line 13-14
        opt.zero_grad()
        L.backward()
        opt.step()

    model.eval()

    is_correct = []
    if dataset_name == "FG-NET":
        is_mae3 = []
        is_mae5 = []
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
        if dataset_name == "FG-NET":
            is_mae3.append(torch.abs(y_bar - y) <= 3)
            is_mae5.append(torch.abs(y_bar - y) <= 5)

    is_correct = torch.cat(is_correct, dim=0)
    acc = torch.mean(is_correct.float()).detach().cpu().numpy()
    if dataset_name != "FG-NET":
        print("%s" % acc)
        return acc
    else:
        is_mae3 = torch.cat(is_mae3, dim=0)
        is_mae5 = torch.cat(is_mae5, dim=0)
        mae3 = torch.mean(is_mae3.float()).detach().cpu().numpy()
        mae5 = torch.mean(is_mae5.float()).detach().cpu().numpy()
        print("%s / %s / %s" % (acc, mae3, mae5))
        return acc, mae3, mae5


if __name__ == '__main__':
    main("Soccer Player", lamd=1e-4, num_epoch=30, use_norm=False)
