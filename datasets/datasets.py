import os

import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat


DATASET_NAME_TUPLE = ("Bird Song",
                      "FG-NET",
                      "Lost",
                      "MSRCv2",
                      "Soccer Player",
                      "Yahoo! News")


class Datasets(Dataset):
    def __init__(self, dataset_name, path="/home/user/Data/PartialLabel"):
        assert dataset_name in DATASET_NAME_TUPLE

        if dataset_name is DATASET_NAME_TUPLE[0]:
            matfile_path = os.path.join(path, "BirdSong.mat")
        elif dataset_name is DATASET_NAME_TUPLE[1]:
            matfile_path = os.path.join(path, "FG-NET.mat")
        elif dataset_name is DATASET_NAME_TUPLE[2]:
            matfile_path = os.path.join(path, "lost.mat")
        elif dataset_name is DATASET_NAME_TUPLE[13]:
            matfile_path = os.path.join(path, "MSRCv2.mat")
        elif dataset_name is DATASET_NAME_TUPLE[4]:
            matfile_path = os.path.join(path, "Soccer Player.mat")
        elif dataset_name is DATASET_NAME_TUPLE[5]:
            matfile_path = os.path.join(path, "Yahoo! News.mat")
        else:
            raise AttributeError("[!] Dataset Name is not defined.")
        self.matfile_path = matfile_path

        dataset = loadmat(self.matfile_path)
        print(dataset)

        