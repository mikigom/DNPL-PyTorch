import numpy as np
import pickle, argparse

REAL_DATASET_NAME_TUPLE = ("bird",
                          "fgnet",
                          "lost",
                          "msrcv2",
                          "soccer",
                          "yahoo")

UCI_DATASET_NAME_TUPLE = ("yeast",
                          "texture",
                          "dermatology",
                          "synthcontrol",
                          "20newsgroups")

parser = argparse.ArgumentParser()
parser.add_argument('-dset', required=True)
args = parser.parse_args()
dset_name = args.dset

if dset_name in REAL_DATASET_NAME_TUPLE:
    middle = '_real_'

if dset_name in UCI_DATASET_NAME_TUPLE:
    middle = '_uci_bino_'

path = 'exp'+middle+dset_name+'/'
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

if dset_name in UCI_DATASET_NAME_TUPLE:
    accs1 = []
    accs7 = []

    for d in d_list:
        if float(d['p'])==0.1 and float(d['beta'])==0.0:
            print( (d['runid'], float(d['p'])) )
            with open(path+d['runid']+'.acc', 'rb') as acc_f1:
                accs1.append(pickle.load(acc_f1))
        elif float(d['p'])==0.7 and float(d['beta'])==0.0:
            print( (d['runid'], float(d['p'])) )
            with open(path+d['runid']+'.acc', 'rb') as acc_f7:
                accs7.append(pickle.load(acc_f7))

    accs1 = np.array(accs1).flatten()
    accs7 = np.array(accs7).flatten()
    print(dset_name)
    print('p=0.1 acc: %s, std: %s \np=0.7 acc: %s, std: %s'% (str(accs1.mean()), str(accs1.std()), str(accs7.mean()), str(accs7.std())) )

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
