import copy

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from torch.utils.data import DataLoader

from datasets.datasets import Datasets
from models.models import DeepModel
from utils import to_torch_var

NUM_ITERATIONS = 2000
NUM_ITERATIONS_ON_WEIGHTS = 20
LEARNING_RATE = 1e-3


def main():
    datasets = Datasets("Bird Song", test_fold=10, val_fold=0)

    train_datasets = copy.deepcopy(datasets)
    train_datasets.set_mode('train')
    train_dataloader = DataLoader(train_datasets, batch_size=128, num_workers=4, drop_last=False, shuffle=True)

    test_datasets = copy.deepcopy(datasets)
    test_datasets.set_mode('test')
    test_dataloader = DataLoader(test_datasets, batch_size=16, num_workers=4, drop_last=False)

    in_dim, out_dim = datasets.get_dims
    model = DeepModel(in_dim, out_dim).cuda()

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)

    train_data_iterator = iter(train_dataloader)
    train_cardinality = train_datasets.get_cardinality_possible_partial_set()
    model.train()
    t = tqdm(range(NUM_ITERATIONS))
    for i in t:
        model.train()
        try:
            data_train, _, y_train, idx_train = next(train_data_iterator)
        except StopIteration:
            train_data_iterator = iter(train_dataloader)
            data_train, _, y_train, idx_train = next(train_data_iterator)

        data_train = to_torch_var(data_train, requires_grad=False).float()
        label_train = torch.argmax(to_torch_var(y_train, requires_grad=False), dim=1)

        y_f_hat = model(data_train)
        C = F.cross_entropy(y_f_hat, label_train, reduce=False)
        l_f = torch.sum(C)

        opt.zero_grad()
        l_f.backward()
        opt.step()

    model.eval()

    is_correct = []
    for X, y_partial, y, idx in test_dataloader:
        data_test = to_torch_var(X, requires_grad=False).float()
        y_test = to_torch_var(y, requires_grad=False).long()
        label_test = torch.argmax(y_test, dim=1)

        logits_predicted = model(data_test)
        probs_predicted = torch.softmax(logits_predicted, dim=1)
        label_predicted = torch.argmax(probs_predicted, dim=1)
        is_correct.append(label_predicted == label_test)
    is_correct = torch.cat(is_correct, dim=0)
    print("Test Acc: %s" % torch.mean(is_correct.float()).detach().cpu().numpy())


if __name__ == '__main__':
    main()
