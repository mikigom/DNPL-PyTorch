import numpy as np
import math
import train_naive
import train_proximal_cd

repeat = 50
epoch_list = (25, 50, 100)
mu_list = (1e-3, 1e-4, 1e-5)
lambda_list = 10 ** np.arange(-5, 3.5, 0.5)
lambda_list = lambda_list.tolist()

if __name__ == '__main__':
    for epoch in epoch_list:
        for lamd in lambda_list:
            accs = list()
            for i in range(repeat):
                acc = train_naive.main("Lost", lamd=lamd, num_epoch=epoch, use_norm=False)
                accs.append(acc)

            with open('exp_lost/naive/epoch_%s_lambda_%s.txt' % (str(epoch), str(lamd)), 'w') as f:
                for acc in accs:
                    f.write("%s\n" % acc)

    for epoch in epoch_list:
        for lamd in lambda_list:
            for mu in mu_list:
                accs = list()
                for i in range(repeat):
                    acc = train_proximal_cd.main("Lost", lamd=lamd, mu=mu, num_epoch=epoch, use_norm=False)
                    accs.append(acc)

                with open('exp_lost/proximal_dc/epoch_%s_lambda_%s_mu_%s.txt' % (str(epoch), str(lamd), str(mu)), 'w') as f:
                    for acc in accs:
                        f.write("%s\n" % acc)
