import copy

import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader

from models.models import *
from utils import to_torch_var, sharpen, loss_monitor
from yogi.yogi import Yogi

def train(train_datasets, test_datasets, batch_size, num_epoch=100, use_norm=False, model_name='medium', monitor=False):

    assert train_datasets.dataset_name == test_datasets.dataset_name

    train_dataloader = DataLoader(train_datasets, batch_size=min((batch_size, len(train_datasets))),
            num_workers=4, drop_last=True, shuffle=True)

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

    opt = Yogi(model.parameters(), lr=1e-3)
    #opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    #opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_data_iterator = iter(train_dataloader)
    current_iter = 0
    total_iter = 0
    current_epoch = 0

    loss_val = 0.

    while True:
        current_iter += 1
        total_iter += 1

        model.train()
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

                    is_correct.append(torch.argmax(y_hat, dim=1) == y)

                is_correct = torch.cat(is_correct, dim=0)
                acc = torch.mean(is_correct.float()).detach().cpu().numpy()
                model.train()
            else:
                acc = 0.0

            loss_val /= current_iter

            if not monitor:
                print("Epoch [{}], Loss:{:.2e}, acc:{:.2e}".format(current_epoch, loss_val, acc))
            else:
                sr_tr, pr_tr, zr_tr = loss_monitor(model, train_datasets)
                sr_tst, pr_tst, zr_tst = loss_monitor(model, test_datasets)
                print("Epoch [{}], Loss:{:.2e} (Train) / {:.2e} (Test), PL-Loss:{:.2e} (Train) / {:.2e} (Test), 0-1 Loss:{:.2e} (Train) / {:.2e} (Test)".format(current_epoch, sr_tr, sr_tst, pr_tr, pr_tst, zr_tr, zr_tst))

            current_iter = 0
            loss_val = 0.

            train_data_iterator = iter(train_dataloader)
            data_train, y_partial_train, _, idx_train = next(train_data_iterator)

        if current_epoch == num_epoch:
            if monitor:
                sr_tr, pr_tr, zr_tr = loss_monitor(model, train_datasets)
                sr_tst, pr_tst, zr_tst = loss_monitor(model, test_datasets)
                print("Epoch [{}], Loss:{:.2e} (Train) / {:.2e} (Test), PL-Loss:{:.2e} (Train) / {:.2e} (Test), 0-1 Loss:{:.2e} (Train) / {:.2e} (Test)".format(current_epoch, sr_tr, sr_tst, pr_tr, pr_tst, zr_tr, zr_tst))
            break

        x = to_torch_var(data_train, requires_grad=False).float()
        s = torch.DoubleTensor(y_partial_train).cuda().float()
        if use_norm:
            x = (x - feature_mean) * inv_feature_std

        # Calculate the loss
        f = model(x)
        s_hat = F.softmax(f, dim=1)
        ss_hat = s * s_hat
        ss_hat_dp = ss_hat.sum(1)
        ss_hat_dp = torch.clamp(ss_hat_dp, 0., 1.)
        loss = -torch.mean(torch.log(ss_hat_dp + 1e-10))

        loss_val += loss.data.tolist()

        if torch.isnan(loss).any():
            print("Warning: NaN Loss")
            break

        # Optimizer step
        opt.zero_grad()
        loss.backward()
        opt.step()

    return acc