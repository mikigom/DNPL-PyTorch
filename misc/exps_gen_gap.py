### Move this file to the parent directory to run this code.

import os
import numpy as np
import torch
import argparse, random, pickle, copy
from datasets.pll_datasets import PLL_Datasets
from datetime import datetime
from time import sleep

from train import train

DATASET_NAME_TUPLE = ("lost", "msrcv2", "soccer", "yahoo")
MODEL_NAME_TUPLE = ("linear", "small", "medium", "residual", "deep", "newdeep")

parser = argparse.ArgumentParser()

parser.add_argument('-dset', required=True, help='Dataset name from (lost, msrcv2, soccer, yahoo).')
parser.add_argument('-use_norm', required=False, help='Normalize the data.')
parser.add_argument('-model', required=False, help='Model choice from (linear, small, medium, residual, deep, newdeep).')
parser.add_argument('-num_epoch', required=False, help='Number of epochs.')
parser.add_argument('-repeats', required=False, help='Number of repetitions.')
parser.add_argument('-fix_data_seed', required=False, help='Fix the seed of dataloader.')
parser.add_argument('-fix_train_seed', required=False, help='Fix the seed of trainer.')

args = parser.parse_args()
dset_name = args.dset
use_norm = False if args.use_norm == None else args.use_norm.lower() in ('true', '1', 't', 'y')
model_name = "medium" if args.model == None else args.model
num_epoch = 200 if args.num_epoch == None else int(args.num_epoch)
repeats = 10 if args.repeats == None else int(args.repeats)
fix_data_seed = False if args.fix_data_seed == None else args.fix_data_seed.lower() in ('true', '1', 't', 'y')
fix_train_seed = False if args.fix_train_seed == None else args.fix_train_seed.lower() in ('true', '1', 't', 'y')

if not dset_name in DATASET_NAME_TUPLE:
    raise AttributeError("Dataset does not exist!")

if not model_name in MODEL_NAME_TUPLE:
    raise AttributeError("Model does not exist!")

if not fix_data_seed or not fix_train_seed:
    seeds = []
    for i in range(3*(1+repeats)):
        seeds.append(datetime.now().microsecond)
        sleep(0.01)
    seeds = iter(seeds)
else:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    accs = []

    if fix_data_seed:
        torch.manual_seed(1)
        np.random.seed(2)
        random.seed(3)
        datasets = PLL_Datasets(dset_name)
    else:
        torch.manual_seed(next(seeds))
        np.random.seed(next(seeds))
        random.seed(next(seeds))
        datasets = PLL_Datasets(dset_name)

    data_num = len(datasets)
    runid = str(datetime.now().strftime('%Y%m%d%H%M%S'))

    full_idx = list(range(data_num))
    random.shuffle(full_idx)
    fold = data_num // 10
    test_idx = full_idx[:fold]
    train_idx = full_idx[fold:]

    for i in range(repeats):            
        if fix_train_seed:
            torch.manual_seed(i+5)
            np.random.seed(2*i+10)
            random.seed(3*i+15)
        else:
            torch.manual_seed(next(seeds))
            np.random.seed(next(seeds))
            random.seed(next(seeds))

        current_num_train = (len(train_idx) // repeats) * (i + 1)

        train_datasets = copy.deepcopy(datasets)
        train_datasets.set_mode('custom', train_idx[:current_num_train])

        test_datasets = copy.deepcopy(datasets)
        test_datasets.set_mode('custom', test_idx)

        train(train_datasets, test_datasets, batch_size=128, use_norm=use_norm, num_epoch=num_epoch, model_name=model_name, monitor=False)