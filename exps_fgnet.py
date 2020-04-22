import numpy as np
import math
import train_naive
import train_proximal_cd

repeat = 50
epoch_list = (50,)
mu_list = (1e-3, 1e-4, 1e-5)
# lambda_list = 10 ** np.arange(-5, 3.5, 0.5)
# lambda_list = lambda_list.tolist()
beta_list = (1e-3, 5e-4, 1e-4)
lamd_list = (1e-6, 5e-6, 1e-5)

if __name__ == '__main__':
    for epoch in epoch_list:
        for beta in beta_list:
            for lamd in lamd_list:
                accs = list()
                for i in range(repeat):
                    acc = train_naive.main("FG-NET", beta=beta, lamd=lamd, num_epoch=epoch, use_norm=False)
                    accs.append(acc)

                with open('exp_fgnet/naive/epoch_%s_beta_%s_lambda_%s.txt' % (str(epoch), str(beta), str(lamd)), 'w') as f:
                    for acc in accs:
                        f.write("%s,%s,%s\n" % (acc[0], acc[1], acc[2]))

                accs = np.array(accs)
                with open('exp_fgnet/naive_records.txt', 'a') as f:
                    f.write("%s, %s, %s, %s %s, %s, %s, %s, %s\n" % (str(epoch), str(beta), str(lamd),
                                                                     str(np.mean(accs[:, 0])), str(np.std(accs[:, 0])),
                                                                     str(np.mean(accs[:, 1])), str(np.std(accs[:, 1])),
                                                                     str(np.mean(accs[:, 2])), str(np.std(accs[:, 2])),))

    """
    for epoch in epoch_list:
        for lamd in lambda_list:
            for mu in mu_list:
                accs = list()
                for i in range(repeat):
                    acc = train_proximal_cd.main("FG-NET", lamd=lamd, mu=mu, num_epoch=epoch, use_norm=False)
                    accs.append(acc)

                with open('exp_fgnet/proximal_dc/epoch_%s_lambda_%s_mu_%s.txt' % (str(epoch), str(lamd), str(mu)), 'w') as f:
                    for acc in accs:
                        f.write("%s,%s,%s\n" % (acc[0], acc[1], acc[2]))
    """
