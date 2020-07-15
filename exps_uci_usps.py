import os
import numpy as np
import train_naive_uci
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

repeat = 50
r_list = (1, 2, 3)
p_list = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


if __name__ == '__main__':
    for use_normal in (True,):
        for r in r_list:
            for p in p_list:
                accs = list()
                for i in range(repeat):
                    torch.manual_seed(i)
                    np.random.seed(i)

                    acc = train_naive_uci.main("usps", r=r, p=p, beta=1e-4, lamd=0., use_norm=use_normal)
                    accs.append(acc)

                os.makedirs("exp_uci_usps", exist_ok=True)
                with open('exp_uci_usps/normal_%s_r_%s_p_%s.txt' % (str(use_normal), str(r), str(p)), 'w') as f:
                    for acc in accs:
                        f.write("%s\n" % acc)

                avg = np.mean(accs)
                stdev = np.std(accs)
                with open('exp_uci_usps/records.txt', 'a') as f:
                    f.write("%s, %s, %s, %s, %s\n" % (str(use_normal), str(r), str(p), str(avg), str(stdev)))
