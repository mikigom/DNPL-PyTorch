import numpy as np
import math
from train import main

repeat = 20
lambda_list = 10 ** np.arange(-4, 1.25, 0.25)
lambda_list = lambda_list.tolist()

if __name__ == '__main__':
    for lamd in lambda_list:
        accs = list()
        for i in range(repeat):
            acc = main("Yahoo! News", lamd=lamd, num_epoch=25, use_norm=False)
            accs.append(acc)

        with open('exp_yahoo/lambda_%s.txt' % str(lamd), 'w') as f:
            for acc in accs:
                f.write("%s\n" % acc)

