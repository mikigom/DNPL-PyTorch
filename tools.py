import numpy as np
import utils


def norm(T):
    row_sum = np.sum(T, 1)
    T_norm = T / row_sum
    return T_norm


def error(T, T_true):
    error = np.sum(np.abs(T-T_true)) / np.sum(np.abs(T_true))
    return error


def transition_matrix_generate(noise_rate=0.5, num_classes=10):
    P = np.ones((num_classes, num_classes))
    n = noise_rate
    P = (n / (num_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, num_classes-1):
            P[i, i] = 1. - n
        P[num_classes-1, num_classes-1] = 1. - n
    return P


def fit(X, y_partial, num_classes, symmetric=False):
    # number of classes
    C = num_classes
    T = np.zeros((C, C))

    estimated_y_count = np.zeros(num_classes)
    for i in range(X.shape[0]):
        estimated_y = np.argmax(X[i])
        estimated_y_count[estimated_y] += 1
        for j in range(C):
            if y_partial[i, j] == 1:
                T[estimated_y, j] += X[estimated_y, j]

    for c in range(C):
        if estimated_y_count[c] != 0:
            T[c] = T[c] / estimated_y_count[c]
        T[c, c] = 1.

    """
    for i in np.arange():
        if not filter_outlier:
            idx_best = np.argmax(eta_corr[:, i])
        else:
            eta_thresh = np.percentile(eta_corr[:, i], 97, interpolation='higher')
            robust_eta = eta_corr[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
            idx_best = np.argmax(robust_eta)
        for j in np.arange(c):
            T[i, j] = eta_corr[idx_best, j]
    """
    if symmetric:
        return (T + T.T) / 2
    else:
        return T


# flip clean labels to noisy labels
# train set and val set split
def dataset_split(train_images, train_labels, noise_rate=0.5, split_per=0.9, random_seed=1, num_classes=10):
    clean_train_labels = train_labels[:, np.newaxis]
    noisy_labels, real_noise_rate, transition_matrix = utils.noisify_multiclass_symmetric(clean_train_labels, 
                                                noise=noise_rate, random_state=random_seed, nb_classes=num_classes)
    noisy_labels = noisy_labels.squeeze()
#    print(noisy_labels)
    num_samples = int(noisy_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples*split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = noisy_labels[train_set_index], noisy_labels[val_set_index]

    return train_set, val_set, train_labels, val_labels
