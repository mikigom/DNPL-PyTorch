import copy

import torch
from tqdm import tqdm
import torch.nn.functional as F

from torch.utils.data import DataLoader

from datasets.datasets import Datasets
from models.models import DeepModel
from utils import to_torch_var

NUM_ITERATIONS = 2000
LEARNING_RATE = 1e-3


def main():
    datasets = Datasets("Yahoo! News", test_fold=10, val_fold=0)

    train_datasets = list()
    train_dataloaders = list()
    for i in (1, 2, 3, 4, 5, None):
        train_dataset = copy.deepcopy(datasets)
        train_dataset.set_mode('train', cardinality_constraint=i)
        train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=4, drop_last=True, shuffle=True)

        train_datasets.append(train_dataset)
        train_dataloaders.append(train_dataloader)

    test_datasets = copy.deepcopy(datasets)
    test_datasets.set_mode('test')
    test_dataloader = DataLoader(test_datasets, batch_size=16, num_workers=4, drop_last=False)

    # val_cardinality = val_datasets.get_cardinality_possible_partial_set()

    in_dim, out_dim = datasets.get_dims
    model = DeepModel(in_dim, out_dim).cuda()

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_data_iterator = iter(train_dataloaders[0])
    train_cardinality = train_datasets[0].get_cardinality_possible_partial_set()

    t = tqdm(range(NUM_ITERATIONS))
    for i in t:
        model.train()
        # Line 2) Get Batch from Training Dataset
        # Expected Question: "Why is this part so ugly?"
        # Answer: "Please refer https://github.com/pytorch/pytorch/issues/1917 ."
        dataloader_index = i // (NUM_ITERATIONS // 6)
        try:
            data_train, y_partial_train, _, idx_train = next(train_data_iterator)
        except StopIteration:
            train_data_iterator = iter(train_dataloaders[dataloader_index])
            train_cardinality = train_datasets[dataloader_index].get_cardinality_possible_partial_set()
            data_train, y_partial_train, _, idx_train = next(train_data_iterator)

        data_train = to_torch_var(data_train, requires_grad=False).float()
        candidate_train_idx = torch.DoubleTensor(y_partial_train).cuda().nonzero(as_tuple=True)

        # Line 12
        y_f_hat = model(data_train)
        y_f_hat_softmax_weighted = F.softmax(y_f_hat, dim=1)
        y_f_hat_softmax_indexed = y_f_hat_softmax_weighted[candidate_train_idx]
        y_f_hat_softmax_reduced = torch.split(y_f_hat_softmax_indexed,
                                              train_cardinality[idx_train.numpy()].tolist())

        y_f_hat_softmax_reduced_weighted_sum = []
        for j, y_f_hat_softmax_weighted in enumerate(y_f_hat_softmax_reduced):
            y_f_hat_softmax_reduced_weighted_sum.append(torch.sum(y_f_hat_softmax_weighted))
        y_f_hat_softmax_reduced_weighted_sum = torch.stack(y_f_hat_softmax_reduced_weighted_sum, dim=0)
        # print(y_f_hat_softmax_reduced_weighted_sum)

        target = torch.ones_like(y_f_hat_softmax_reduced_weighted_sum)

        l_f = F.binary_cross_entropy(torch.clamp(y_f_hat_softmax_reduced_weighted_sum, 0., 1.),
                                     target,
                                     reduction='sum')

        loss = l_f

        # Line 13-14
        opt.zero_grad()
        loss.backward()
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
