import os
import numpy as np
import train
import torch
import argparse, random, pickle
from datasets.img_datasets import Img_Datasets
from datetime import datetime

DATASET_NAME_TUPLE = ("mnist",
                      "kmnist",
                      "fmnist",
                      "cifar")

MODEL_NAME_TUPLE = ("newdeep",
                    "convnet")

parser = argparse.ArgumentParser()

parser.add_argument('-dset', required=True)
parser.add_argument('-p', required=True)
parser.add_argument('-model', required=False)
parser.add_argument('-use_norm', required=False)
parser.add_argument('-simp_loss', required=False)
parser.add_argument('-beta', required=False)
parser.add_argument('-num_epoch', required=False)
parser.add_argument('-fix_data_seed', required=False)
parser.add_argument('-fix_train_seed', required=False)

args = parser.parse_args()
dset_name = args.dset
p = float(args.p)
model_name = "newdeep" if args.model == None else args.model
use_norm = False if args.use_norm == None else args.use_norm.lower() in ('true', '1', 't', 'y')
simp_loss = True if args.simp_loss == None else args.simp_loss.lower() in ('true', '1', 't', 'y')
num_epoch = 500 if args.num_epoch == None else int(args.num_epoch)
beta = 1e-3 if args.beta == None else float(args.beta)
fix_data_seed = False if args.fix_data_seed == None else args.fix_data_seed.lower() in ('true', '1', 't', 'y')
fix_train_seed = False if args.fix_train_seed == None else args.fix_train_seed.lower() in ('true', '1', 't', 'y')

if not dset_name in DATASET_NAME_TUPLE:
    raise AttributeError("Dataset does not exist!")

if not model_name in MODEL_NAME_TUPLE:
    raise AttributeError("Model does not exist!")

p_list = (p,)

if dset_name == "cifar":
    model_name = "convnet"
    use_norm = False

if not fix_data_seed or not fix_train_seed:
    #import quantumrandom as qrng
    #seeds = iter(qrng.get_data(array_length = len(p_list)*3+len(p_list)*3))
    seeds = []
    for i in range(len(p_list)*3+len(p_list)*3):
        seeds.append(datetime.now().microsecond)
        sleep(0.01)
    seeds = iter(seeds)
else:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    os.makedirs('exp_img_'+dset_name, exist_ok=True)
    #f = open('exp_img_'+dset_name+'/records.txt','a')
    Datasets = Img_Datasets(dset_name)

    for p in p_list:
        accs = []

        if fix_data_seed:
            torch.manual_seed(1)
            np.random.seed(2)
            random.seed(3)
            train_datasets = Datasets(r=None, p=p, eps=None, train=True, binomial=True, download=True)
            test_datasets = Datasets(r=None, p=p, eps=None, train=False, binomial=True, download=True)
        else:
            torch.manual_seed(next(seeds))
            np.random.seed(next(seeds))
            random.seed(next(seeds))
            train_datasets = Datasets(r=None, p=p, eps=None, train=True, binomial=True, download=True)
            test_datasets = Datasets(r=None, p=p, eps=None, train=False, binomial=True, download=True)

        data_num = len(train_datasets)+len(test_datasets)
        runid = str(datetime.now().strftime('%Y%m%d%H%M%S'))

        if fix_train_seed:
            torch.manual_seed(i+5)
            np.random.seed(2*i+10)
            random.seed(3*i+15)
        else:
            torch.manual_seed(next(seeds))
            np.random.seed(next(seeds))
            random.seed(next(seeds))
                
        acc = train.main(train_datasets, test_datasets, bs=256, beta=beta, use_norm=False, num_epoch=num_epoch, model_name=model_name, simp_loss=simp_loss)
        #accs.append(acc)

        #with open('exp_img_'+dset_name+'/'+runid+'.acc', 'wb') as acc_out:
        #    pickle.dump(accs, acc_out)

        #f.write("runid: %s, beta: %s, model: %s, simp_loss: %s, epoch: %s, use_norm: %s, fix_data_seed: %s, fix_train_seed: %s, p: %s\n"  
        #        % (runid, str(beta), str(model_name), str(simp_loss), str(num_epoch), str(use_norm), str(fix_data_seed), 
        #            str(fix_train_seed), str(p)))
        #f.flush()

    #f.close()

