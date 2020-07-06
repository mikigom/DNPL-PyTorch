import os
import numpy as np
import random

from torch.utils.data import Dataset


DATASET_NAME_TUPLE = ("segment",
                      "satimage",
                      "usps",
                      "letter")


def load_segment(path):
    # #Example: 2,310
    # #Feature: 18
    # #Class: 7
    line = True
    X_str = []
    y_str = []
    with open(os.path.join(path, "segmentation.data")) as data:
        cnt = 0
        while line:
            cnt += 1
            line = data.readline()
            if cnt <= 5:
                continue
            if line == '':
                break
            line = line.split(',')
            if line is not None:
                y_str.append(line[0])
                x_str = line[1:]
                x_str = [float(i) for i in x_str]
                X_str.append(x_str)

    line = True
    with open(os.path.join(path, "segmentation.test"), 'r') as data:
        cnt = 0
        while line:
            cnt += 1
            line = data.readline()
            if cnt <= 5:
                continue
            if line == '':
                break
            line = line.split(',')
            if line is not None:
                y_str.append(line[0])
                x_str = line[1:]
                x_str = [float(i) for i in x_str]
                X_str.append(x_str)

    X = np.array(X_str)
    keys = list(set(y_str))
    keyToidx = {key: i for i, key in enumerate(keys)}
    y = [keyToidx[label_str] for label_str in y_str]

    return {'data': X, 'target': y}


def load_satimage(path):
    # #Example: 6,345
    # #Feature: 36
    # #Class: 7 (It's 6, really...)
    line = True
    X_str = []
    y_str = []
    with open(os.path.join(path, "sat.trn")) as data:
        cnt = 0
        while line:
            cnt += 1
            line = data.readline()
            if line == '':
                break
            line = line.split(' ')
            if line is not None:
                y_str.append(line[-1])
                x_str = line[:-1]
                x_str = [float(i) for i in x_str]
                X_str.append(x_str)

    line = True
    with open(os.path.join(path, "sat.tst"), 'r') as data:
        cnt = 0
        while line:
            cnt += 1
            line = data.readline()
            if line == '':
                break
            line = line.split(' ')
            if line is not None:
                y_str.append(line[-1])
                x_str = line[:-1]
                x_str = [float(i) for i in x_str]
                X_str.append(x_str)

    X = np.array(X_str)
    keys = list(set(y_str))
    keyToidx = {key: i for i, key in enumerate(keys)}
    y = [keyToidx[label_str] for label_str in y_str]

    return {'data': X, 'target': y}


def load_usps(path):
    # #Example: 9,298
    # #Feature: 256
    # #Class: 10
    import h5py

    with h5py.File(os.path.join(path, "USPS.h5"), 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]

        X = np.concatenate((X_tr, X_te))
        y = np.concatenate((y_tr, y_te))

    return {'data': X, 'target': y}


def load_letter(path):
    # #Example: 20,000
    # #Feature: 16
    # #Class: 26
    line = True
    X_str = []
    y_str = []
    with open(os.path.join(path, "letter-recognition.data")) as data:
        cnt = 0
        while line:
            cnt += 1
            line = data.readline()
            if line == '':
                break
            line = line.split(',')
            if line is not None:
                y_str.append(line[0])
                x_str = line[1:]
                x_str = [float(i) for i in x_str]
                X_str.append(x_str)

    X = np.array(X_str).squeeze()
    keys = list(set(y_str))
    keyToidx = {key: i for i, key in enumerate(keys)}
    y = [keyToidx[label_str] for label_str in y_str]

    return {'data': X, 'target': y}


def toOneHot(target):
    num = np.unique(target, axis=0)
    num = num.shape[0]
    encoding = np.eye(num)[target]
    return encoding


def makePartialLabel(onehot_target, r, p, eps):
    if eps is not None:
        assert r == 1 and p == 1

    target = onehot_target.copy()
    if eps is None:
        idx_list = list(range(target.shape[0]))
        selected_idxes = random.sample(idx_list, k=int(p * target.shape[0]))
        for idx in selected_idxes:
            added_labels = random.sample(list(np.nonzero(1 - target[idx])), k=r)
            for added_label in added_labels:
                target[idx, added_label] = 1
    else:
        coupling = {}
        for label in range(onehot_target.shape[1]):
            labels = list(range(onehot_target.shape[1]))
            labels.remove(label)
            coupling[label] = random.sample(labels, k=1)

        gts = np.argmax(target, axis=-1)
        for i in range(target.shape[0]):
            if np.random.uniform(0., 1.) < eps:
                gt = gts[i]
                target[i, coupling[gt]] = 1

    return target


class UCI_Datasets(Dataset):
    def __init__(self, dataset_name, r, p, eps, path="data/", test_fold=10, val_fold=0):
        assert dataset_name in DATASET_NAME_TUPLE

        if dataset_name == DATASET_NAME_TUPLE[0]:
            dataset = load_segment(path)
        elif dataset_name == DATASET_NAME_TUPLE[1]:
            dataset = load_satimage(path)
        elif dataset_name == DATASET_NAME_TUPLE[2]:
            dataset = load_usps(path)
        elif dataset_name == DATASET_NAME_TUPLE[3]:
            dataset = load_letter(path)
        else:
            raise AttributeError("Dataset Name is not defined.")

        # an Mxd matrix w.r.t. the feature	representations,
        # where M is the number of instances and d is the number of features.
        self.data = dataset['data']
        # a MxQ matrix w.r.t. the candidate	labeling information, where Q is the number of possible class labels.
        self.target = toOneHot(dataset['target'])

        self.M = self.data.shape[0]
        test_num = self.M // test_fold
        self.trainval_num = self.M - test_num
        if val_fold == 0:
            val_num = 0
        else:
            val_num = self.trainval_num // val_fold
        train_num = self.trainval_num - val_num

        self.fold_idx = np.arange(0, self.M)
        np.random.shuffle(self.fold_idx)

        self.train_fold_idx = self.fold_idx[:train_num]
        self.val_fold_idx = self.fold_idx[train_num:train_num+val_num]
        self.test_fold_idx = self.fold_idx[train_num+val_num:]

        self.mode = 'all'
        self.set_mode('all')
        self.r = r
        self.p = p
        self.eps = eps

    def set_mode(self, to, cardinality_constraint=None):
        if to == 'train':
            self.X = self.data[self.train_fold_idx]
            self.y = self.target[self.train_fold_idx]
            self.y_partial = makePartialLabel(self.y, self.r, self.p, self.eps)

            if cardinality_constraint is not None:
                cardinality = self.get_cardinality_possible_partial_set()
                re_indexed = cardinality <= cardinality_constraint

                self.X = self.X[re_indexed]
                self.y = self.y[re_indexed]
                self.y_partial = makePartialLabel(self.y, self.r, self.p, self.eps)
                self.train_fold_idx = self.train_fold_idx[re_indexed]

        elif to == 'val':
            self.X = self.data[self.val_fold_idx]
            self.y = self.target[self.val_fold_idx]
            self.y_partial = makePartialLabel(self.y, self.r, self.p, self.eps)
        elif to == 'trainval':
            fold_idx = np.concatenate((self.train_fold_idx, self.val_fold_idx), axis=0)

            self.X = self.data[fold_idx]
            self.y = self.target[fold_idx]
            self.y_partial = makePartialLabel(self.y, self.r, self.p, self.eps)
        elif to == 'test':
            self.X = self.data[self.test_fold_idx]
            self.y = self.target[self.test_fold_idx]
            self.y_partial = makePartialLabel(self.y, self.r, self.p, self.eps)
        elif to == 'all':
            self.X = self.data
            self.y = self.target
            self.y_partial = makePartialLabel(self.y, self.r, self.p, self.eps)
        else:
            raise AttributeError

    def __getitem__(self, idx):
        X = self.X[idx]

        toarray_op = getattr(self.y, "toarray", None)
        if callable(toarray_op):
            y = self.y[idx].toarray().squeeze().astype(np.float64)
        else:
            y = self.y[idx].squeeze().astype(np.float64)

        toarray_op = getattr(self.y_partial, "toarray", None)
        if callable(toarray_op):
            y_partial = self.y_partial[idx].toarray().squeeze().astype(np.float64)
        else:
            y_partial = self.y_partial[idx].squeeze().astype(np.float64)
        # y_partial = self.y_partial[:, idx].toarray().squeeze()
        return X, y_partial, y, idx

    def __len__(self):
        return self.X.shape[0]

    def get_cardinality_possible_partial_set(self):
        toarray_op = getattr(self.y_partial, "toarray", None)
        if callable(toarray_op):
            y_partial = self.y_partial.toarray()
        else:
            y_partial = self.y_partial
        return np.count_nonzero(y_partial, axis=0)

    def reset_trainval_split(self):
        reshuffle_idx = np.arange(0, self.trainval_num)
        np.random.shuffle(reshuffle_idx)

        current_trainval_fold_idx = np.concatenate((self.train_fold_idx, self.val_fold_idx), axis=0)

        self.train_fold_idx = current_trainval_fold_idx[reshuffle_idx[:self.train_fold_idx.shape[0]]]
        self.val_fold_idx = current_trainval_fold_idx[reshuffle_idx[self.train_fold_idx.shape[0]:]]

    @property
    def get_dims(self):
        return self.data.shape[1], self.target.shape[1]

    @property
    def get_feature_mean(self):
        return np.mean(self.X, axis=0)[np.newaxis]

    @property
    def get_feature_std(self):
        return np.std(self.X, axis=0)[np.newaxis]


if __name__ == '__main__':
    # load_segment(path='')
    # load_satimage(path='')
    # load_usps(path='')
    # load_letter(path='')
    pass
