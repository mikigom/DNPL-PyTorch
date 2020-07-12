import os
import numpy as np
import train_naive_uci
import torch

PATH = "C:/Users/s3213/PycharmProjects/BilevelPartialLabel"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

repeat = 20
r_list = (1, 2, 3)
p_list = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


if __name__ == '__main__':
    for r in r_list:
        for p in p_list:
            accs = list()
            for i in range(repeat):
                torch.manual_seed(i)
                np.random.seed(i)

                acc = train_naive_uci.main("segment", r=r, p=p, beta=0., lamd=0., use_norm=False)
                accs.append(acc)

            os.makedirs(os.path.join(PATH, "exp_uci_segment"), exist_ok=True)
            with open(os.path.join(PATH, 'exp_uci_segment', 'r_%s_p_%s.txt' % (str(r), str(p)), 'w')) as f:
                for acc in accs:
                    f.write("%s\n" % acc)

            avg = np.mean(accs)
            stdev = np.std(accs)
            with open(os.path.join(PATH, 'exp_uci_segment', 'records.txt', 'a')) as f:
                f.write("%s, %s, %s, %s\n" % (str(r), str(p), str(avg), str(stdev)))
