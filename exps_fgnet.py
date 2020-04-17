import numpy as np
import math
from train import main

repeat = 50
epoch_list = (25, 50)
lambda_list = 10 ** np.arange(-5, 3.5, 0.5)
lambda_list = lambda_list.tolist()

if __name__ == '__main__':
    for epoch in epoch_list:
        for lamd in lambda_list:
            accs = list()
            for i in range(repeat):
                acc = main("FG-NET", lamd=lamd, num_epoch=epoch, use_norm=False)
                accs.append(acc)

            with open('exp_fgnet/epoch_%s_lambda_%s.txt' % (str(epoch), str(lamd)), 'w') as f:
                for acc in accs:
                    f.write("%s,%s,%s\n" % (acc[0], acc[1], acc[2]))
