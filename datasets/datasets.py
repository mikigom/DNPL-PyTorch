import os
import numpy as np

from torch.utils.data import Dataset
from scipy.io import loadmat


DATASET_NAME_TUPLE = ("Bird Song",
                      "FG-NET",
                      "Lost",
                      "MSRCv2",
                      "Soccer Player",
                      "Yahoo! News")


class Datasets(Dataset):
    def __init__(self, dataset_name, path="/home/user/Data/PartialLabel", test_fold=10, val_fold=10):
        assert dataset_name in DATASET_NAME_TUPLE

        if dataset_name == DATASET_NAME_TUPLE[0]:
            matfile_path = os.path.join(path, "BirdSong.mat")
        elif dataset_name == DATASET_NAME_TUPLE[1]:
            matfile_path = os.path.join(path, "FG-NET.mat")
        elif dataset_name == DATASET_NAME_TUPLE[2]:
            matfile_path = os.path.join(path, "lost.mat")
        elif dataset_name == DATASET_NAME_TUPLE[3]:
            matfile_path = os.path.join(path, "MSRCv2.mat")
        elif dataset_name == DATASET_NAME_TUPLE[4]:
            matfile_path = os.path.join(path, "Soccer Player.mat")
        elif dataset_name == DATASET_NAME_TUPLE[5]:
            matfile_path = os.path.join(path, "Yahoo! News.mat")
        else:
            raise AttributeError("Dataset Name is not defined.")
        self.matfile_path = matfile_path

        dataset = loadmat(self.matfile_path)
        # an Mxd matrix w.r.t. the feature	representations,
        # where M is the number of instances and d is the number of features.
        self.data = dataset['data']
        # a QxM matrix w.r.t. the candidate	labeling information, where Q is the number of possible class labels.
        self.target = dataset['target']
        # a QxM matrix w.r.t. the ground-truth labeling	information.
        self.partial_target = dataset['partial_target']

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

    def set_mode(self, to, cardinality_constraint=None):
        if to == 'train':
            self.X = self.data[self.train_fold_idx]
            self.y = self.target[:, self.train_fold_idx]
            self.y_partial = self.partial_target[:, self.train_fold_idx]

            if cardinality_constraint is not None:
                cardinality = self.get_cardinality_possible_partial_set()
                re_indexed = cardinality <= cardinality_constraint

                self.X = self.X[re_indexed]
                self.y = self.y[:, re_indexed]
                self.y_partial = self.y_partial[:, re_indexed]
                self.train_fold_idx = self.train_fold_idx[re_indexed]

                cardinality = self.get_cardinality_possible_partial_set()

        elif to == 'val':
            self.X = self.data[self.val_fold_idx]
            self.y = self.target[:, self.val_fold_idx]
            self.y_partial = self.partial_target[:, self.val_fold_idx]
        elif to == 'test':
            self.X = self.data[self.test_fold_idx]
            self.y = self.target[:, self.test_fold_idx]
            self.y_partial = self.partial_target[:, self.test_fold_idx]
        elif to == 'all':
            self.X = self.data
            self.y = self.target
            self.y_partial = self.partial_target
        else:
            raise AttributeError

    def __getitem__(self, idx):
        X = self.X[idx]

        toarray_op = getattr(self.y, "toarray", None)
        if callable(toarray_op):
            y = self.y[:, idx].toarray().squeeze().astype(np.float64)
        else:
            y = self.y[:, idx].squeeze().astype(np.float64)

        toarray_op = getattr(self.y_partial, "toarray", None)
        if callable(toarray_op):
            y_partial = self.y_partial[:, idx].toarray().squeeze().astype(np.float64)
        else:
            y_partial = self.y_partial[:, idx].squeeze().astype(np.float64)
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
        return self.data.shape[1], self.target.shape[0]
