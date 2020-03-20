import copy

import torch
from tqdm import tqdm
import torch.nn.functional as F

from torch.utils.data import DataLoader

from datasets.datasets import Datasets
from models.models import DeepModel
from utils import to_torch_var, ExampleLabelWeights, HLoss

NUM_ITERATIONS = 5000
NUM_ITERATIONS_ON_WEIGHTS = 20
LEARNING_RATE = 1e-3


def main():
    datasets = Datasets("Lost", test_fold=10, val_fold=0)

    train_datasets = copy.deepcopy(datasets)
    train_datasets.set_mode('train')
    train_dataloader = DataLoader(train_datasets, batch_size=64, num_workers=4, drop_last=True, shuffle=True)

    h_loss = HLoss()

    """
    val_datasets = copy.deepcopy(datasets)
    val_datasets.set_mode('val')
    val_dataloader = DataLoader(val_datasets, batch_size=16, num_workers=4, drop_last=True, shuffle=True)
    """

    test_datasets = copy.deepcopy(datasets)
    test_datasets.set_mode('test')
    test_dataloader = DataLoader(test_datasets, batch_size=16, num_workers=4, drop_last=False)

    train_cardinality = train_datasets.get_cardinality_possible_partial_set()
    # val_cardinality = val_datasets.get_cardinality_possible_partial_set()

    # Used in lower-loop
    example_label_weights = ExampleLabelWeights(train_cardinality)

    in_dim, out_dim = datasets.get_dims
    model = DeepModel(in_dim, out_dim).cuda()

    opt = torch.optim.Adam(model.params(), lr=LEARNING_RATE)

    train_data_iterator = iter(train_dataloader)
    # val_data_iterator = iter(val_dataloader)
    t = tqdm(range(NUM_ITERATIONS))

    for i in t:
        model.train()
        # Line 2) Get Batch from Training Dataset
        # Expected Question: "Why is this part so ugly?"
        # Answer: "Please refer https://github.com/pytorch/pytorch/issues/1917 ."
        try:
            data_train, y_partial_train, _, idx_train = next(train_data_iterator)
        except StopIteration:
            train_data_iterator = iter(train_dataloader)
            data_train, y_partial_train, _, idx_train = next(train_data_iterator)

        data_train = to_torch_var(data_train, requires_grad=False).float()
        candidate_train_idx = torch.DoubleTensor(y_partial_train).cuda().nonzero(as_tuple=True)

        example_label_weights.eval()
        # Line 12
        y_f_hat = model(data_train)
        l_e = h_loss(y_f_hat)
        y_f_hat_softmax_weighted = F.softmax(y_f_hat, dim=1)
        y_f_hat_softmax_indexed = y_f_hat_softmax_weighted[candidate_train_idx]
        y_f_hat_softmax_reduced = torch.split(y_f_hat_softmax_indexed,
                                              train_cardinality[idx_train.numpy()].tolist())

        y_f_hat_softmax_reduced_weighted_sum = []
        for j, y_f_hat_softmax_weighted in enumerate(y_f_hat_softmax_reduced):
            y_f_hat_softmax_reduced_weighted_sum.append(torch.sum(y_f_hat_softmax_weighted))
        y_f_hat_softmax_reduced_weighted_sum = torch.stack(y_f_hat_softmax_reduced_weighted_sum, dim=0)
        l_f = F.binary_cross_entropy(torch.clamp(y_f_hat_softmax_reduced_weighted_sum, 0., 1.),
                                     torch.ones_like(y_f_hat_softmax_reduced_weighted_sum),
                                     reduction='sum')

        loss = l_f + 1e-3 * (i / NUM_ITERATIONS) * l_e

        # Line 13-14
        opt.zero_grad()
        loss.backward()
        opt.step()

    """
    for param in example_label_weights.params:
        print(torch.softmax(param.data, dim=0).detach().cpu().numpy())
    """

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
