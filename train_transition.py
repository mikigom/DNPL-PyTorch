import copy

import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F

from torch.utils.data import DataLoader

from datasets.datasets import Datasets
from models.models import DeepModel
from utils import to_torch_var

import tools

NUM_INITIAL_TRAINING_EPOCHS = 20
LEARNING_RATE = 1e-3


def main():
    datasets = Datasets("Soccer Player", test_fold=10, val_fold=5)

    train_datasets = copy.deepcopy(datasets)
    train_datasets.set_mode('train')
    train_dataloader = DataLoader(train_datasets, batch_size=48, num_workers=4, drop_last=True, shuffle=True)

    train_estimate_datasets = copy.deepcopy(train_datasets)
    train_estimate_dataloader = DataLoader(train_estimate_datasets, batch_size=48, num_workers=4, drop_last=False, shuffle=False)

    val_datasets = copy.deepcopy(datasets)
    val_datasets.set_mode('val')
    val_dataloader = DataLoader(val_datasets, batch_size=16, num_workers=4, drop_last=False, shuffle=True)

    test_datasets = copy.deepcopy(datasets)
    test_datasets.set_mode('test')
    test_dataloader = DataLoader(test_datasets, batch_size=16, num_workers=4, drop_last=False)

    train_cardinality = train_datasets.get_cardinality_possible_partial_set()
    val_cardinality = val_datasets.get_cardinality_possible_partial_set()

    in_dim, out_dim = datasets.get_dims
    model = DeepModel(in_dim, out_dim).cuda()

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    t = tqdm(range(NUM_INITIAL_TRAINING_EPOCHS))

    val_epoch_accs = []
    train_estimate_epoch_probs = []
    for epoch in t:
        model.train()
        for data_train, y_partial_train, _, idx_train in train_dataloader:
            data_train = to_torch_var(data_train, requires_grad=False).float()
            candidate_train_idx = torch.DoubleTensor(y_partial_train).cuda().nonzero(as_tuple=True)

            # Line 12
            y_f_hat = model(data_train)
            y_f_hat_candidate = torch.sigmoid(y_f_hat)
            target = - torch.ones_like(y_partial_train)
            for j in range(y_partial_train.size(0)):
                label_idx = y_partial_train[j].nonzero()[:, 0]
                target[j, :label_idx.size(0)] = label_idx
            target.detach_()
    
            l_f = F.multilabel_margin_loss(y_f_hat_candidate, target.cuda().long(), reduction='mean')

            """
            y_f_hat_softmax_indexed = y_f_hat_softmax[candidate_train_idx]
            y_f_hat_softmax_reduced = torch.split(y_f_hat_softmax_indexed,
                                                  train_cardinality[idx_train.numpy()].tolist())

            y_f_hat_softmax_reduced_weighted_sum = []
            for j, y_f_hat_softmax_weighted in enumerate(y_f_hat_softmax_reduced):
                y_f_hat_softmax_reduced_weighted_sum.append(torch.sum(y_f_hat_softmax_weighted))
            y_f_hat_softmax_reduced_weighted_sum = torch.stack(y_f_hat_softmax_reduced_weighted_sum, dim=0)

            target = torch.ones_like(y_f_hat_softmax_reduced_weighted_sum)

            l_z = F.binary_cross_entropy(torch.clamp(y_f_hat_softmax_reduced_weighted_sum, 0., 1.),
                                         target, reduction='mean')
            """

            loss = l_f

            # Line 13-14
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_acc = []
            for data_val, y_partial_val, _, idx_val in val_dataloader:
                data_val = to_torch_var(data_val, requires_grad=False).float()
                candidate_val_idx = torch.DoubleTensor(y_partial_val).cuda().nonzero(as_tuple=True)

                y_f_hat = model(data_val)
                y_f_hat_softmax = torch.softmax(y_f_hat, dim=1)

                y_f_hat_softmax_indexed = y_f_hat_softmax[candidate_val_idx]
                y_f_hat_softmax_reduced = torch.split(y_f_hat_softmax_indexed,
                                                      val_cardinality[idx_val.numpy()].tolist())

                y_f_hat_softmax_reduced_weighted_sum = []
                for j, y_f_hat_softmax_weighted in enumerate(y_f_hat_softmax_reduced):
                    y_f_hat_softmax_reduced_weighted_sum.append(torch.sum(y_f_hat_softmax_weighted))
                y_f_hat_softmax_reduced_weighted_sum = torch.stack(y_f_hat_softmax_reduced_weighted_sum, dim=0)
                val_correct = (y_f_hat_softmax_reduced_weighted_sum > 0.5).float()

                val_acc.append(val_correct)
            val_acc = torch.cat(val_acc, dim=0)
            val_acc = torch.mean(val_acc)
            val_epoch_accs.append(val_acc.item())

        with torch.no_grad():
            train_estimate_probs = []
            for data_estimate, y_partial_data_estimate, _, idx_estimate in train_estimate_dataloader:
                data_estimate = to_torch_var(data_estimate, requires_grad=False).float()

                y_f_hat = model(data_estimate)
                y_f_hat_sigmoid = torch.sigmoid(y_f_hat)
                y_f_hat_sigmoid = y_f_hat_sigmoid.cpu()
                train_estimate_probs.append(y_f_hat_sigmoid)
            train_estimate_probs = torch.cat(train_estimate_probs, dim=0)
            train_estimate_epoch_probs.append(train_estimate_probs)

    model.eval()
    val_epoch_accs = np.array(val_epoch_accs)
    model_index = np.argmax(val_epoch_accs)

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

    transition_matrix = tools.fit(train_estimate_epoch_probs[model_index],
                                  train_estimate_datasets.y_partial.T, out_dim, True)
    transition_matrix = torch.tensor(transition_matrix, requires_grad=False).cuda().float()
    print(transition_matrix)

    model = DeepModel(in_dim, out_dim).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    t = tqdm(range(2 * NUM_INITIAL_TRAINING_EPOCHS))
    for epoch in t:
        model.train()
        for data_train, y_partial_train, _, idx_train in train_dataloader:
            data_train = to_torch_var(data_train, requires_grad=False).float()
            candidate_train_idx = torch.DoubleTensor(y_partial_train).cuda().nonzero(as_tuple=True)

            # Line 12
            y_f_hat = model(data_train)
            y_f_hat_softmax = torch.softmax(y_f_hat, dim=1)
            y_f_hat_candidate = torch.bmm(y_f_hat_softmax[:, None],
                                          transition_matrix.repeat((y_f_hat_softmax.size(0), 1, 1)))
            y_f_hat_candidate = y_f_hat_candidate.squeeze()

            target = - torch.ones_like(y_partial_train)
            for j in range(y_partial_train.size(0)):
                label_idx = y_partial_train[j].nonzero()[:, 0]
                target[j, :label_idx.size(0)] = label_idx
            target.detach_()

            l_f = F.multilabel_margin_loss(y_f_hat_candidate, target.cuda().long(), reduction='mean')

            y_f_hat_softmax_indexed = y_f_hat_softmax[candidate_train_idx]
            y_f_hat_softmax_reduced = torch.split(y_f_hat_softmax_indexed,
                                                  train_cardinality[idx_train.numpy()].tolist())

            y_f_hat_softmax_reduced_weighted_sum = []
            for j, y_f_hat_softmax_weighted in enumerate(y_f_hat_softmax_reduced):
                y_f_hat_softmax_reduced_weighted_sum.append(torch.sum(y_f_hat_softmax_weighted))
            y_f_hat_softmax_reduced_weighted_sum = torch.stack(y_f_hat_softmax_reduced_weighted_sum, dim=0)

            target = torch.ones_like(y_f_hat_softmax_reduced_weighted_sum)

            l_z = F.binary_cross_entropy(torch.clamp(y_f_hat_softmax_reduced_weighted_sum, 0., 1.),
                                         target, reduction='mean')

            loss = l_f + l_z

            # Line 13-14
            opt.zero_grad()
            loss.backward()
            opt.step()

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
