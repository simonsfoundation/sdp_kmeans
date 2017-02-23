from __future__ import absolute_import, print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import timeit
from data import real
from sdp_kmeans.sdp import sdp_kmeans_multilayer, cluster_sdp_burer_monteiro
from tests.utils import plot_matrix


dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'embedding/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def check_completely_positivity(sym_mat, Y):
    error = (np.linalg.norm(sym_mat - Y.dot(Y.T), 'fro')
             / np.linalg.norm(sym_mat, 'fro'))
    return error


def test_mnist(k, n_samples_range, rank_factors=[4, 8], digit=1):
    time_sdp = {}
    time_sdp_bm = dict([(rf, {}) for rf in rank_factors])
    rel_err_sdp_bm = dict([(rf, {}) for rf in rank_factors])

    for n_samples in n_samples_range:
        print('---')
        print('n_samples', n_samples)
        X = real.mnist(digit=digit, n_samples=n_samples)

        if n_samples < 1000:
            t = timeit.default_timer()
            Q_sdp = sdp_kmeans_multilayer(X, [k])
            t = timeit.default_timer() - t
            print('SDP', t)
            time_sdp[n_samples] = t

        for rf in rank_factors:
            np.random.seed(0)
            t = timeit.default_timer()
            Y = cluster_sdp_burer_monteiro(X, k, rank=k * rf)
            t = timeit.default_timer() - t
            time_sdp_bm[rf][n_samples] = t
            Q_bm = Y.dot(Y.T)
            if n_samples < 400:
                rel_err = (np.linalg.norm(Q_sdp[-1] - Q_bm, 'fro')
                           / np.linalg.norm(Q_sdp[-1], 'fro'))
                rel_err_sdp_bm[rf][n_samples] = rel_err
                print('BM rank-{}'.format(rf), t, rel_err)
            else:
                print('BM rank-{}'.format(rf), t)

    with open('mnist_times.pickle', 'wb') as file:
        pickle.dump(n_samples_range, file)
        pickle.dump(time_sdp, file)
        pickle.dump(time_sdp_bm, file)
        pickle.dump(rel_err_sdp_bm, file)

    # plt.savefig('{}{}_burer-monteiro.pdf'.format(dir_name, filename))


if __name__ == '__main__':
    n_samples_range = list(range(50, 1000, 50))
    n_samples_range += list(range(1000, 2000, 100))
    n_samples_range += list(range(2000, 10001, 1000))

    test_mnist(16, n_samples_range, rank_factors=[4, 8], digit=1)

    with open('mnist_times.pickle', 'rb') as file:
        n_samples_range = pickle.load(file)
        time_sdp = pickle.load(file)
        time_sdp_bm = pickle.load(file)
        rel_err_sdp_bm = pickle.load(file)

    print(time_sdp)
    print(time_sdp_bm)
    print(rel_err_sdp_bm)

    plt.show()
