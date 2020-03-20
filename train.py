import copy

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from torch.utils.data import DataLoader

from datasets.datasets import Datasets
from models.models import DeepModel
from utils import to_torch_var, ExampleLabelWeights

NUM_ITERATIONS = 5000
NUM_ITERATIONS_ON_WEIGHTS = 20
LEARNING_RATE = 1e-4


def main():
    datasets = Datasets("Lost", test_fold=10, val_fold=10)

    train_datasets = copy.deepcopy(datasets)
    train_datasets.set_mode('train')
    train_dataloader = DataLoader(train_datasets, batch_size=32, num_workers=4, drop_last=False, shuffle=True)

    val_datasets = copy.deepcopy(datasets)
    val_datasets.set_mode('val')
    val_dataloader = DataLoader(val_datasets, batch_size=48, num_workers=4, drop_last=False, shuffle=True)

    test_datasets = copy.deepcopy(datasets)
    test_datasets.set_mode('test')
    test_dataloader = DataLoader(test_datasets, batch_size=16, num_workers=4, drop_last=False)

    train_cardinality = train_datasets.get_cardinality_possible_partial_set()
    val_cardinality = val_datasets.get_cardinality_possible_partial_set()

    # Used in lower-loop
    example_label_weights = ExampleLabelWeights(train_cardinality)

    in_dim, out_dim = datasets.get_dims
    model = DeepModel(in_dim, out_dim).cuda()

    opt = torch.optim.Adam(model.params(), lr=LEARNING_RATE)

    train_data_iterator = iter(train_dataloader)
    val_data_iterator = iter(val_dataloader)
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

        data_train = to_torch_var(data_train, requires_grad=False)
        candidate_idx = torch.DoubleTensor(y_partial_train).cuda().nonzero()
        data_train_with_candidates = data_train[candidate_idx[:, 0]].float()
        y_train_with_candidates = to_torch_var(candidate_idx[:, 1], requires_grad=False)

        # For lower-loop, set meta-model
        # It's instantiation of separation of upper-loop parameter and lower-loop parameter.
        meta_model = DeepModel(in_dim, out_dim).cuda()
        meta_model.load_state_dict(model.state_dict())
        meta_model.cuda()
        meta_model.train()

        example_label_weights.train()
        # Line 4
        y_f_hat = meta_model(data_train_with_candidates)
        C = F.cross_entropy(y_f_hat, y_train_with_candidates, reduce=False)
        # Line 5
        # l_f_meta = torch.sum(C)
        l_f_meta = example_label_weights(C, idx_train)

        # Line 6: Get gradient thorough meta-model
        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        # Line 7: Update meta-model parameter
        meta_model.update_params(LEARNING_RATE, source_params=grads)

        # Ready for validation data batch
        try:
            data_val, y_partial_val, _, idx_val = next(val_data_iterator)
        except StopIteration:
            val_data_iterator = iter(val_dataloader)
            data_val, y_partial_val, _, idx_val = next(val_data_iterator)

        data_val = to_torch_var(data_val, requires_grad=False).float()
        candidate_idx = torch.DoubleTensor(y_partial_val).cuda().nonzero(as_tuple=True)
        # Line 8
        meta_model.eval()
        y_g_hat = meta_model(data_val)
        y_g_hat_softmax = F.softmax(y_g_hat, dim=1)
        y_g_hat_softmax_indexed = y_g_hat_softmax[candidate_idx]
        y_g_hat_softmax_reduced = torch.split(y_g_hat_softmax_indexed,
                                              val_cardinality[idx_val.numpy()].tolist())
        y_g_hat_softmax_reduced_sum = []
        for y_g_hat_softmax in y_g_hat_softmax_reduced:
            y_g_hat_softmax_reduced_sum.append(torch.sum(y_g_hat_softmax))
        y_g_hat_softmax_reduced_sum = torch.stack(y_g_hat_softmax_reduced_sum, dim=0)

        t.set_description("Upper Loss: %s" % torch.mean(y_g_hat_softmax_reduced_sum).item())
        l_g_meta = F.binary_cross_entropy(torch.clamp(y_g_hat_softmax_reduced_sum, 0., 1.),
                                          torch.ones_like(y_g_hat_softmax_reduced_sum),
                                          reduction='sum')

        grad_eps = torch.autograd.grad(l_g_meta, example_label_weights.params, only_inputs=True, allow_unused=True)
        example_label_weights.update_last_used_weights(grad_eps, 5e-1)

        example_label_weights.eval()
        # Line 12
        y_f_hat = model(data_train_with_candidates)
        C = F.cross_entropy(y_f_hat, y_train_with_candidates, reduce=False)
        l_f = example_label_weights(C, idx_train)

        # Line 13-14
        opt.zero_grad()
        l_f.backward()
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
