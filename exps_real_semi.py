import os
import numpy as np
import train_semi
import torch
import argparse, random, pickle
from datasets.real_datasets import Real_Datasets
from sklearn.model_selection import KFold
from datetime import datetime
from time import sleep

DATASET_NAME_TUPLE = ("bird",
                      "lost",
                      "msrcv2",
                      "soccer",
                      "yahoo")

MODEL_NAME_TUPLE = ("linear",
                    "small",
                    "medium",
                    "residual",
                    "deep",
                    "newdeep")

parser = argparse.ArgumentParser()

parser.add_argument('-dset', required=True)
parser.add_argument('-use_norm', required=False)
parser.add_argument('-model', required=False)
parser.add_argument('-simp_loss', required=False)
parser.add_argument('-beta', required=False)
parser.add_argument('-beta2', required=False)
parser.add_argument('-num_epoch', required=False)
parser.add_argument('-cv_fold', required=False)
parser.add_argument('-fix_data_seed', required=False)
parser.add_argument('-fix_train_seed', required=False)
parser.add_argument('-p', required=True)

args = parser.parse_args()
dset_name = args.dset
use_norm = False if args.use_norm == None else args.use_norm.lower() in ('true', '1', 't', 'y')
model_name = "medium" if args.model == None else args.model
simp_loss = True if args.simp_loss == None else args.simp_loss.lower() in ('true', '1', 't', 'y')
num_epoch = 200 if args.num_epoch == None else int(args.num_epoch)
cv_fold = 5 if args.cv_fold == None else int(args.cv_fold)
beta = 1e-3 if args.beta == None else float(args.beta)
beta2 = beta if args.beta2 == None else float(args.beta2)
fix_data_seed = False if args.fix_data_seed == None else args.fix_data_seed.lower() in ('true', '1', 't', 'y')
fix_train_seed = False if args.fix_train_seed == None else args.fix_train_seed.lower() in ('true', '1', 't', 'y')
p = float(args.p)

if not dset_name in DATASET_NAME_TUPLE:
    raise AttributeError("Dataset does not exist!")

if dset_name in ("bird",):
    use_norm = True

if not model_name in MODEL_NAME_TUPLE:
    raise AttributeError("Model does not exist!")

if not fix_data_seed or not fix_train_seed:
    #import quantumrandom as qrng
    #seeds = iter(qrng.get_data(array_length = 3*(1+cv_fold)))
    seeds = []
    for i in range(3*(1+cv_fold)):
        seeds.append(datetime.now().microsecond)
        sleep(0.01)
    seeds = iter(seeds)
else:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    os.makedirs('exp_smi_real_'+dset_name, exist_ok=True)
    f = open('exp_smi_real_'+dset_name+'/records.txt','a')

    accs = list()

    if fix_data_seed:
        torch.manual_seed(1)
        np.random.seed(2)
        random.seed(3)
        kf = KFold(n_splits=cv_fold, shuffle=True, random_state=4)
        datasets = Real_Datasets(dset_name)
    else:
        torch.manual_seed(next(seeds))
        np.random.seed(next(seeds))
        random.seed(next(seeds))
        kf = KFold(n_splits=cv_fold, shuffle=True)
        datasets = Real_Datasets(dset_name)

    data_num = len(datasets)
    runid = str(datetime.now().strftime('%Y%m%d%H%M%S'))

    for i, (train_idx, test_idx) in enumerate(kf.split(range(data_num))):
        random.shuffle(train_idx)
        n_of_semi_train = int((1 - p) * len(train_idx))
        train_semi_idx = train_idx[:n_of_semi_train]
        train_idx = train_idx[n_of_semi_train:]

        if fix_train_seed:
            torch.manual_seed(i+5)
            np.random.seed(2*i+10)
            random.seed(3*i+15)
        else:
            torch.manual_seed(next(seeds))
            np.random.seed(next(seeds))
            random.seed(next(seeds))

        acc = train_semi.main(datasets, train_idx, train_semi_idx, test_idx,
                              bs=128, beta=beta, beta2=beta,
                              use_norm=use_norm, num_epoch=num_epoch,
                              model_name=model_name, simp_loss=simp_loss)
        accs.append(acc)

    with open('exp_smi_real_'+dset_name+'/'+runid+'.acc', 'wb') as acc_out:
        pickle.dump(accs, acc_out)

    acc_avg = np.mean(accs)
    acc_std = np.std(accs)
    f.write("runid: %s, p: %s, beta: %s beta2: %s, model: %s, simp_loss: %s, epoch: %s, use_norm: %s, fix_data_seed: %s, fix_train_seed: %s, acc_avg: %s, acc_std: %s, cv_fold: %s\n" % (runid, str(p), str(beta), str(beta2), str(model_name), str(simp_loss), str(num_epoch), str(use_norm), str(fix_data_seed), str(fix_train_seed), str(acc_avg), str(acc_std), str(cv_fold)))
    f.flush()
    f.close()
