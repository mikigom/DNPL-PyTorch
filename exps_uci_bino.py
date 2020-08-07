import os
import numpy as np
import train
import torch
import argparse
import random
from datasets.uci_datasets_new import UCI_Datasets
from sklearn.model_selection import KFold

DATASET_NAME_TUPLE = ("yeast",
                      "texture",
                      "dermatology",
                      "synthcontrol",
                      "20newsgroups")

MODEL_NAME_TUPLE = ("linear",
                    "small",
                    "medium",
                    "residual",
                    "deep",
                    "newdeep")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()

parser.add_argument('-dset', required=True)
parser.add_argument('-use_norm', required=False)
parser.add_argument('-model', required=False)
parser.add_argument('-simp_loss', required=False)
parser.add_argument('-beta', required=False)
parser.add_argument('-num_epoch', required=False)
parser.add_argument('-cv_fold', required=False)
parser.add_argument('-fix_data_seed', required=False)
parser.add_argument('-fix_train_seed', required=False)


args = parser.parse_args()
dset_name = args.dset
use_norm = True if args.use_norm == None else args.use_norm.lower() in ('true', '1', 't', 'y')
model_name = "medium" if args.model == None else args.model
simp_loss = True if args.simp_loss == None else args.simp_loss.lower() in ('true', '1', 't', 'y')
num_epoch = 50 if args.num_epoch == None else int(args.num_epoch)
cv_fold = 5 if args.cv_fold == None else int(args.cv_fold)
beta = .5 if args.beta == None else float(args.beta)
fix_data_seed = False if args.fix_data_seed == None else args.fix_data_seed.lower() in ('true', '1', 't', 'y')
fix_train_seed = False if args.fix_train_seed == None else args.fix_train_seed.lower() in ('true', '1', 't', 'y')

if not dset_name in DATASET_NAME_TUPLE:
    raise AttributeError("Dataset does not exist!")

if not model_name in MODEL_NAME_TUPLE:
    raise AttributeError("Model does not exist!")

r_list = (0,)
p_list = (0.7,)

if not fix_data_seed or not fix_train_seed:
    import quantumrandom as qrng
    qrngs = iter(qrng.get_data(array_length = len(r_list)*len(p_list)*3+len(r_list)*len(p_list)*cv_fold*3))

if __name__ == '__main__':

    os.makedirs('exp_uci_'+dset_name, exist_ok=True)
    f1 = open('exp_uci_'+dset_name+'/records.txt','a')

    for r in r_list:
        for p in p_list:
            accs = list()

            if fix_data_seed:
                torch.manual_seed(1)
                np.random.seed(2)
                random.seed(3)
                kf = KFold(n_splits=cv_fold, shuffle=True, random_state=4)
                datasets = UCI_Datasets(dset_name, r=r, p=p, eps=None, binomial=True)
            else:
                torch.manual_seed(next(qrngs))
                np.random.seed(next(qrngs))
                random.seed(next(qrngs))
                kf = KFold(n_splits=cv_fold, shuffle=True)
                datasets = UCI_Datasets(dset_name, r=r, p=p, eps=None, binomial=True)

            data_num = len(datasets)

            for i, (train_idx, test_idx) in enumerate(kf.split(range(data_num))):
               
                if fix_train_seed:
                    torch.manual_seed(i+5)
                    np.random.seed(2*i+10)
                    random.seed(3*i+15)
                else:
                    torch.manual_seed(next(qrngs))
                    np.random.seed(next(qrngs))
                    random.seed(next(qrngs))
                
                acc = train.main(datasets, train_idx, test_idx, bs=128, beta=beta, use_norm=use_norm, num_epoch=num_epoch, model_name=model_name, simp_loss=simp_loss)
                print("acc: %f, fold: %s/%s, r: %s, p: %s" % (acc, str(i+1), str(cv_fold), str(r), str(p)) )
                accs.append(acc)

            #with open('exp_uci_'+dset_name+'/beta_%s_model_%s_epoch_%s_norm_%s_fix_seed_%s_r_%s_p_%s.txt' 
            #        % (str(beta), str(model_name), str(num_epoch), str(use_norm), str(fix_seed), str(r), str(p)), 'a') as f2:
            #    for acc in accs:
            #        f2.write("%s\n" % acc)

            acc_avg = np.mean(accs)
            acc_std = np.std(accs)
            f1.write("beta: %s, model: %s, simp_loss: %s, epoch: %s, use_norm: %s, fix_data_seed: %s, fix_train_seed: %s, r: %s, p: %s, acc_avg: %s, acc_std: %s, cv_fold: %s\n"  
                    % (str(beta), str(model_name), str(simp_loss), str(num_epoch), str(use_norm), str(fix_data_seed), 
                        str(fix_train_seed), str(r), str(p), str(acc_avg), str(acc_std), str(cv_fold)))
            f1.flush()

    f1.close()

