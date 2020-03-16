import copy

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from torch.utils.data import DataLoader

from datasets.datasets import Datasets
from models.models import DeepModel
from utils import to_torch_var, ExampleLabelWeights

NUM_ITERATIONS = 2000


def main():
    datasets = Datasets("Bird Song", test_fold=10, val_fold=10)

    train_datasets = copy.deepcopy(datasets)
    train_datasets.set_mode('train')
    train_dataloader = DataLoader(train_datasets, batch_size=32, num_workers=4, drop_last=False, shuffle=True)

    val_datasets = copy.deepcopy(datasets)
    val_datasets.set_mode('val')
    val_dataloader = DataLoader(val_datasets, batch_size=10, num_workers=4, drop_last=False, shuffle=True)

    test_datasets = copy.deepcopy(datasets)
    test_datasets.set_mode('test')
    test_dataloader = DataLoader(test_datasets, batch_size=16, num_workers=4, drop_last=False)

    datasets.set_mode('train')
    cardinality = datasets.get_cardinality_possible_partial_set()

    # Used in lower-loop
    example_label_weights = ExampleLabelWeights(cardinality)

    in_dim, out_dim = datasets.get_dims
    model = DeepModel(in_dim, out_dim).cuda()

    train_data_iterator = iter(train_dataloader)
    for i in tqdm(range(NUM_ITERATIONS)):
        model.train()
        # Line 2) Get Batch from Training Dataset
        # Expected Question: "Why is this part so ugly?"
        # Answer: "Please refer https://github.com/pytorch/pytorch/issues/1917 ."
        try:
            data_train, y_partial_train, _, idx_train = next(train_data_iterator)
        except StopIteration:
            train_data_iterator = iter(train_dataloader)
            data_train, y_partial_train, _, idx_train = next(train_data_iterator)

        data_train = to_torch_var(data_train, requires_grad=False)
        candidate_idx = torch.DoubleTensor(y_partial_train).cuda().nonzero()
        data_train_with_candidates = data_train[candidate_idx[:, 0]]
        y_train_with_candidates = to_torch_var(candidate_idx[:, 1], requires_grad=False)

        # For lower-loop, set meta-model
        # It's instantiation of separation of upper-loop parameter and lower-loop parameter.
        meta_model = DeepModel(in_dim, out_dim).cuda()
        meta_model.load_state_dict(model.state_dict())
        meta_model.cuda()

        # Line 4
        y_f_hat = meta_model(data_train_with_candidates)
        C = F.cross_entropy(y_f_hat, y_train_with_candidates, reduce=False)
        # Line 5
        l_f_meta = example_label_weights(C, idx_train)

        # TODO) Line 6: Get gradient thorough meta-model
        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        meta_model.update_params(hyperparameters['lr'], source_params=grads)
        """
        # Line 4) Forward-pass on lower-loop
        y_f_hat = meta_model(train_data)
        cost = F.cross_entropy(y_f_hat, train_labels, reduce=False)

        # Line 5) TODO: Need to be inter-changed
        eps = to_torch_var(torch.zeros(cost.size()))
        l_f_meta = torch.sum(cost * eps)
    """


if __name__ == '__main__':
    main()
