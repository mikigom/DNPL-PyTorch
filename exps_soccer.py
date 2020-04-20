import numpy as np
import math
import train_naive
import train_proximal_cd

repeat = 10
epoch_list = (50,)
mu_list = (1e-3, 1e-4, 1e-5)
# lambda_list = 10 ** np.arange(-5, 3.5, 0.5)
# lambda_list = lambda_list.tolist()
beta_list = (1e-4, 5e-4, 1e-3)
lamd_list = (1e-3, 5e-4, 1e-4)

if __name__ == '__main__':
    for epoch in epoch_list:
        for beta in beta_list:
            for lamd in lamd_list:
                accs = list()
                for i in range(repeat):
                    acc = train_naive.main("Soccer Player", beta=beta, lamd=lamd, num_epoch=epoch, use_norm=False)
                    accs.append(acc)

                with open('exp_soccer/naive/epoch_%s_beta_%s_lambda_%s.txt' % (str(epoch), str(beta), str(lamd)), 'w') as f:
                    for acc in accs:
                        f.write("%s\n" % acc)

                avg = np.mean(accs)
                stdev = np.std(accs)
                with open('exp_soccer/naive_records.txt', 'a') as f:
                    f.write("%s, %s, %s, %s, %s\n" % (str(epoch), str(beta), str(lamd), str(avg), str(stdev)))

    """
    for epoch in epoch_list:
        for lamd in lambda_list:
            for mu in mu_list:
                accs = list()
                for i in range(repeat):
                    acc = train_proximal_cd.main("Soccer Player", lamd=lamd, mu=mu, num_epoch=epoch, use_norm=False)
                    accs.append(acc)

                with open('exp_soccer/proximal_dc/epoch_%s_lambda_%s_mu_%s.txt' % (str(epoch), str(lamd), str(mu)), 'w') as f:
                    for acc in accs:
                        f.write("%s\n" % acc)
    """
