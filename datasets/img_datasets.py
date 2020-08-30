import os
import numpy as np
import random

from torch.utils.data import Dataset
from torchvision.datasets import MNIST, KMNIST, FashionMNIST, CIFAR10
from torchvision import transforms


DATASET_NAME_TUPLE = ("mnist",
                      "kmnist",
                      "fmnist",
                      "cifar")


def toOneHot(target):
    num = np.unique(target, axis=0)
    num = num.shape[0]
    encoding = np.eye(num)[target]
    return encoding


def makePartialLabel(onehot_target, r, p, eps, binomial = True):
    if eps is not None:
        assert r == 1 and p == 1

    target = onehot_target.copy()
    if eps is None:
        if binomial:
            rs = np.random.binomial(target.shape[1]-1, p, size = target.shape[0])
            rs[np.nonzero(rs==0)] = 1
            for idx in range(target.shape[0]):
                added_labels = random.sample(np.nonzero(1 - target[idx])[0].tolist(), k=rs[idx]) 
                for added_label in added_labels:
                    target[idx, added_label] = 1
        else:
            idx_list = list(range(target.shape[0]))
            selected_idxes = random.sample(idx_list, k=int(p * target.shape[0]))
            for idx in selected_idxes:
                added_labels = random.sample(np.nonzero(1 - target[idx])[0].tolist(), k=r)
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


class MNIST_Partial(MNIST):
    def __init__(self, r, p, eps, binomial = True, path="data/", train=True, transform=None, target_transform=None, download=False):
        super(MNIST_Partial, self).__init__(root=path, train=train, transform=transform, target_transform=target_transform, download=download)
        self.r = r
        self.p = p
        self.eps = eps
        self.binomial = binomial
        self.X = self.data.view(self.data.shape[0], -1)
        self.y = toOneHot(self.targets)
        self.y_partial = makePartialLabel(self.y, self.r, self.p, self.eps, self.binomial)
        self.dataset_name = 'mnist'

    def __getitem__(self, idx):
        X = self.X[idx]
        y_partial = self.y_partial[idx]
        y = self.y[idx]

        return X, y_partial, y, idx

    @property
    def get_dims(self):
        return self.X.shape[1], self.y.shape[1]


class KMNIST_Partial(KMNIST):
    def __init__(self, r, p, eps, binomial = True, path="data/", train=True, transform=None, target_transform=None, download=False):
        super(KMNIST_Partial, self).__init__(root=path, train=train, transform=transform, target_transform=target_transform, download=download)
        self.r = r
        self.p = p
        self.eps = eps
        self.binomial = binomial
        self.X = self.data.view(self.data.shape[0], -1)
        self.y = toOneHot(self.targets)
        self.y_partial = makePartialLabel(self.y, self.r, self.p, self.eps, self.binomial)
        self.dataset_name = 'kmnist'

    def __getitem__(self, idx):
        X = self.X[idx]
        y_partial = self.y_partial[idx]
        y = self.y[idx]

        return X, y_partial, y, idx

    @property
    def get_dims(self):
        return self.X.shape[1], self.y.shape[1]


class FMNIST_Partial(FashionMNIST):
    def __init__(self, r, p, eps, binomial = True, path="data/", train=True, transform=None, target_transform=None, download=False):
        super(FMNIST_Partial, self).__init__(root=path, train=train, transform=transform, target_transform=target_transform, download=download)
        self.r = r
        self.p = p
        self.eps = eps
        self.binomial = binomial
        self.X = self.data.view(self.data.shape[0], -1)
        self.y = toOneHot(self.targets)
        self.y_partial = makePartialLabel(self.y, self.r, self.p, self.eps, self.binomial)
        self.dataset_name = 'fmnist'

    def __getitem__(self, idx):
        X = self.X[idx]
        y_partial = self.y_partial[idx]
        y = self.y[idx]

        return X, y_partial, y, idx

    @property
    def get_dims(self):
        return self.X.shape[1], self.y.shape[1]


class CIFAR_Partial(CIFAR10):
    def __init__(self, r, p, eps, binomial = True, path="data/", train=True, transform=None, target_transform=None, download=False):

        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        super(CIFAR_Partial, self).__init__(root=path, train=train, transform=self.transform, target_transform=target_transform, download=download)
        self.r = r
        self.p = p
        self.eps = eps
        self.binomial = binomial
        self.X = self.data.transpose(0,3,1,2)
        self.y = toOneHot(self.targets)
        self.y_partial = makePartialLabel(self.y, self.r, self.p, self.eps, self.binomial)
        self.dataset_name = 'cifar'

    def __getitem__(self, idx):
        X = self.X[idx]
        y_partial = self.y_partial[idx]
        y = self.y[idx]

        return X, y_partial, y, idx

    @property
    def get_dims(self):
        return self.X.shape[1], self.y.shape[1]


def Img_Datasets(dataset_name):
    assert dataset_name in DATASET_NAME_TUPLE 
    
    if dataset_name == DATASET_NAME_TUPLE[0]: 
        return MNIST_Partial
    elif dataset_name == DATASET_NAME_TUPLE[1]:
        return KMNIST_Partial
    elif dataset_name == DATASET_NAME_TUPLE[2]:
        return FMNIST_Partial
    elif dataset_name == DATASET_NAME_TUPLE[3]:
        return CIFAR_Partial
