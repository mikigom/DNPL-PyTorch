import os
import numpy as np
import train
import torch
import argparse, random, pickle
from datasets.datasets_new import Datasets
from sklearn.model_selection import KFold
from datetime import datetime

DATASET_NAME_TUPLE = ("bird",
                      "fgnet",
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
use_norm = False if args.use_norm == None else args.use_norm.lower() in ('true', '1', 't', 'y')
model_name = "deep" if args.model == None else args.model
simp_loss = True if args.simp_loss == None else args.simp_loss.lower() in ('true', '1', 't', 'y')
num_epoch = 100 if args.num_epoch == None else int(args.num_epoch)
cv_fold = 10 if args.cv_fold == None else int(args.cv_fold)
beta = 1e-3 if args.beta == None else float(args.beta)
fix_data_seed = False if args.fix_data_seed == None else args.fix_data_seed.lower() in ('true', '1', 't', 'y')
fix_train_seed = False if args.fix_train_seed == None else args.fix_train_seed.lower() in ('true', '1', 't', 'y')

if not dset_name in DATASET_NAME_TUPLE:
    raise AttributeError("Dataset does not exist!")

if not model_name in MODEL_NAME_TUPLE:
    raise AttributeError("Model does not exist!")

if not fix_data_seed or not fix_train_seed:
    import quantumrandom as qrng
    qrngs = iter(qrng.get_data(array_length = 3*(1+cv_fold)))

if __name__ == '__main__':

    os.makedirs('exp_real_'+dset_name, exist_ok=True)
    f = open('exp_real_'+dset_name+'/records.txt','a')

    accs = list()

    if fix_data_seed:
        torch.manual_seed(1)
        np.random.seed(2)
        random.seed(3)
        kf = KFold(n_splits=cv_fold, shuffle=True, random_state=4)
        datasets = Datasets(dset_name)
    else:
        torch.manual_seed(next(qrngs))
        np.random.seed(next(qrngs))
        random.seed(next(qrngs))
        kf = KFold(n_splits=cv_fold, shuffle=True)
        datasets = Datasets(dset_name)

    data_num = len(datasets)
    runid = str(datetime.now().strftime('%Y%m%d%H%M%S'))

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
        if dset_name == 'fgnet':
            print("fold: %s/%s" % (str(i+1), str(cv_fold)))
        else:
            print("acc: %f, fold: %s/%s" % (acc, str(i+1), str(cv_fold)) )
        accs.append(acc)

    with open('exp_real_'+dset_name+'/'+runid+'.acc', 'wb') as acc_out:
        pickle.dump(accs, acc_out)

    if dset_name == 'fgnet':
        accs = np.array(accs)
        acc_avg = np.mean(accs[:, 0])
        acc_std = np.std(accs[:, 0])
        mae3_avg = np.mean(accs[:, 1])
        mae3_std = np.std(accs[:, 1])
        mae5_avg = np.mean(accs[:, 2])
        mae5_std = np.std(accs[:, 2])
        f.write("runid: %s, beta: %s, model: %s, simp_loss: %s, epoch: %s, use_norm: %s, fix_data_seed: %s, fix_train_seed: %s, avg: %s %s %s, acc_std: %s %s %s, cv_fold: %s\n"% (runid, str(beta), str(model_name), str(simp_loss), str(num_epoch), str(use_norm), str(fix_data_seed), str(fix_train_seed), str(acc_avg), str(mae3_avg), str(mae5_avg), str(acc_std), str(mae3_std), str(mae5_std), str(cv_fold)))
        f.flush()
        f.close()
    else:
        acc_avg = np.mean(accs)
        acc_std = np.std(accs)
        f.write("runid: %s, beta: %s, model: %s, simp_loss: %s, epoch: %s, use_norm: %s, fix_data_seed: %s, fix_train_seed: %s, acc_avg: %s, acc_std: %s, cv_fold: %s\n"% (runid, str(beta), str(model_name), str(simp_loss), str(num_epoch), str(use_norm), str(fix_data_seed), str(fix_train_seed), str(acc_avg), str(acc_std), str(cv_fold)))
        f.flush()
        f.close()

