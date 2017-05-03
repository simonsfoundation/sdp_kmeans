from __future__ import absolute_import, print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import timeit
import seaborn.apionly as sns
import sys
from data import real
from sdp_kmeans.sdp import sdp_kmeans, sdp_km_burer_monteiro
from tests.utils import Logger


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


def test_mnist(k, n_samples_range, rank_factors=[4, 8], digit=1,
               from_file=False):
    if not from_file:
        time_sdp = {}
        time_sdp_bm = dict([(rf, {}) for rf in rank_factors])
        rel_err_sdp_bm = dict([(rf, {}) for rf in rank_factors])

        for n_samples in n_samples_range:
            print('---')
            print('n_samples', n_samples)
            try:
                X = real.mnist(digit=digit, n_samples=n_samples)
            except ValueError:
                break

            if n_samples < 1000:
                t = timeit.default_timer()
                Q_sdp = sdp_kmeans(X, k, method='cvx')
                t = timeit.default_timer() - t
                print('SDP', t)
                time_sdp[n_samples] = t

            for rf in rank_factors:
                np.random.seed(0)
                t = timeit.default_timer()
                Y = sdp_km_burer_monteiro(X, k, rank=k * rf)
                t = timeit.default_timer() - t
                time_sdp_bm[rf][n_samples] = t
                Q_bm = Y.dot(Y.T)
                if n_samples < 1000:
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
    else:
        with open('mnist_times.pickle', 'rb') as file:
            n_samples_range = pickle.load(file)
            time_sdp = pickle.load(file)
            time_sdp_bm = pickle.load(file)
            rel_err_sdp_bm = pickle.load(file)

    sns.set_style('whitegrid')
    sns.set_color_codes()

    plt.figure()
    n_samples_range_active = [ns for ns in n_samples_range
                              if ns in time_sdp]
    plt.plot(n_samples_range_active,
             [time_sdp[ns] for ns in n_samples_range_active],
             linewidth=2,
             label=r'convex SDP solver ($K={0}$)'.format(k))

    for rf in time_sdp_bm:
        n_samples_range_active = [ns for ns in n_samples_range
                                  if ns in time_sdp_bm[rf]]
        plt.loglog(n_samples_range_active,
                 [time_sdp_bm[rf][ns] for ns in n_samples_range_active],
                 linewidth=2,
                 label=r'non-convex SDP solver ($K={0}, r={1}$)'.format(k, k * rf))

    plt.xlabel('Dataset size ($n$)', fontsize='x-large')
    plt.ylabel('Time (s)', fontsize='x-large')
    plt.xlim(min(n_samples_range_active) - 3, max(n_samples_range))
    plt.ylim(1e-2, 1e4)
    plt.yticks(10. ** np.arange(-1, 5))
    plt.legend(loc='lower right', fontsize='x-large')
    plt.savefig('{}mnist_sdp_bm_timing.pdf'.format(dir_name))


if __name__ == '__main__':
    logger = Logger(dir_name + 'test_clustering_real.txt')
    sys.stdout = logger

    n_samples_range = list(range(50, 1000, 50))
    n_samples_range += list(range(1000, 2000, 100))
    n_samples_range += list(range(2000, 10001, 1000))

    test_mnist(16, n_samples_range, rank_factors=[4, 8], digit=1,
               from_file=True)

    plt.show()

    sys.stdout = logger.stdout
    logger.close()
