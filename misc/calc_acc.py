### Move this file to the parent directory to run this code.

import numpy as np
import pickle, argparse

REAL_DATASET_NAME_TUPLE = ("lost", "msrcv2", "soccer", "yahoo")

parser = argparse.ArgumentParser()
parser.add_argument('-dset', required=True)
args = parser.parse_args()
dset_name = args.dset

path = '../results'+middle+dset_name+'/'
d_list = []
with open(path+'records.txt','r') as f:
    lines = [line.rstrip('\n') for line in f]
    for line in lines:
        d = {}
        keyvals = line.split(', ')
        for keyval in keyvals:
            key, val = keyval.split(': ')
            d[key] = val
        d_list.append(d)

if dset_name in REAL_DATASET_NAME_TUPLE:
    accs = []

    for d in d_list:
        if d['epoch']=='200' and float(d['beta'])==0.0:
            print( (d['runid'], float(d['beta'])) )
            with open(path+d['runid']+'.acc', 'rb') as acc_f:
                accs.append(pickle.load(acc_f))

    accs = np.array(accs).flatten()
    print(accs)
    print(dset_name)
    print('acc: %s, std: %s'% (str(accs.mean()), str(accs.std())) )