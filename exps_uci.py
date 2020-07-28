import os
import numpy as np
import train
import torch
import argparse

DATASET_NAME_TUPLE = ("dermatology",
                      "vehicle",
                      "segment",
                      "satimage",
                      "usps",
                      "letter",
                      "ecoli")

MODEL_NAME_TUPLE = ("medium",
                    "residual",
                    "deep")


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()

parser.add_argument('-dset', required=True)
parser.add_argument('-use_normal', required=False)
parser.add_argument('-model', required=False)
parser.add_argument('-beta', required=False)
parser.add_argument('-fold', required=False)
parser.add_argument('-repeat', required=False)


args = parser.parse_args()
dset_name = args.dset
use_normal = True if args.use_normal == None else  args.use_normal.lower() in ('true', '1', 't', 'y')
model_name = "medium" if args.model == None else args.model
fold = 10 if args.fold == None else int(args.fold)
repeat = 1 if args.repeat == None else int(args.repeat)
beta = 0.5 if args.beta == None else int(args.beta)

if not dset_name in DATASET_NAME_TUPLE:
    raise AttributeError("Dataset does not exist!")

if not model_name in MODEL_NAME_TUPLE:
    raise AttributeError("Model does not exist!")

r_list = (1, 2, 3)
p_list = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

if __name__ == '__main__':
    for r in r_list:
        for p in p_list:
            accs = list()
            for i in range(repeat):
                torch.manual_seed(i)
                np.random.seed(i)

                acc = train.main(dset_name, r=r, p=p, beta=beta, lamd=0., use_norm=use_normal)
                accs.append(acc)

            os.makedirs('exp_uci_'+dset_name, exist_ok=True)
            with open('exp_uci_'+dset_name+'/normal_%s_r_%s_p_%s.txt' % (str(use_normal), str(r), str(p)), 'w') as f:
                for acc in accs:
                    f.write("%s\n" % acc)

            avg = np.mean(accs)
            stdev = np.std(accs)
            with open('exp_uci_'+dset_name+'/records.txt', 'w') as f:
                f.write("%s, %s, %s, %s, %s\n" % (str(use_normal), str(r), str(p), str(avg), str(stdev)))

