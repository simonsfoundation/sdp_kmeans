from __future__ import absolute_import, print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import timeit
import seaborn.apionly as sns
import sys
from data import real
from sdp_kmeans import sdp_kmeans, sdp_km_burer_monteiro
from tests.utils import Logger

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'embedding/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def mnist_timing(k, n_samples_range, rank_factors=[4, 8], digit=1,
                  from_file=False):
    pickle_filename = 'mnist_k{}_times.pickle'.format(k)

    if not from_file:
        time_sdp_cvx = {}
        time_sdp_bm = dict([(rf, {}) for rf in rank_factors])
        rel_err_sdp_bm = dict([(rf, {}) for rf in rank_factors])
        time_sdp_cgm = {}
        rel_err_sdp_cgm = {}

        for n_samples in n_samples_range:
            print('---')
            print('n_samples', n_samples)
            try:
                X = real.mnist(digit=digit, n_samples=n_samples)
            except ValueError:
                break

            if n_samples <= 1000:
                t = timeit.default_timer()
                D, Q_cvx = sdp_kmeans(X, k, method='cvx')
                t = timeit.default_timer() - t
                print('SDP-CVX', t)
                time_sdp_cvx[n_samples] = t

            for rf in rank_factors:
                rank = k * rf
                np.random.seed(0)
                t = timeit.default_timer()
                Y = sdp_km_burer_monteiro(X, k, rank=rank)
                t = timeit.default_timer() - t
                time_sdp_bm[rf][n_samples] = t
                Q_bm = Y.dot(Y.T)
                if n_samples <= 1000:
                    rel_err = (np.linalg.norm(Q_cvx - Q_bm, 'fro')
                               / np.linalg.norm(Q_cvx, 'fro'))
                    rel_err_sdp_bm[rf][n_samples] = rel_err
                    print('BM rank-{}'.format(rank), t, rel_err)
                else:
                    print('BM rank-{}'.format(rank), t)

            t = timeit.default_timer()
            D, Q_cgm = sdp_kmeans(X, k, method='cgm')
            t = timeit.default_timer() - t
            time_sdp_cgm[n_samples] = t
            if n_samples <= 1000:
                rel_err = (np.linalg.norm(Q_cvx - Q_cgm, 'fro')
                           / np.linalg.norm(Q_cvx, 'fro'))
                rel_err_sdp_cgm[n_samples] = rel_err
                print('SDP_CGM', t, rel_err)
            else:
                print('SDP_CGM', t)

            with open(pickle_filename, 'wb') as file:
                pickle.dump(n_samples_range, file)
                pickle.dump(time_sdp_cvx, file)
                pickle.dump(time_sdp_bm, file)
                pickle.dump(rel_err_sdp_bm, file)
                pickle.dump(time_sdp_cgm, file)
                pickle.dump(rel_err_sdp_cgm, file)

    else:
        with open(pickle_filename, 'rb') as file:
            n_samples_range = pickle.load(file)
            time_sdp_cvx = pickle.load(file)
            time_sdp_bm = pickle.load(file)
            rel_err_sdp_bm = pickle.load(file)
            time_sdp_cgm = pickle.load(file)
            rel_err_sdp_cgm = pickle.load(file)

    sns.set_style('whitegrid')
    sns.set_color_codes()

    plt.figure()
    n_samples_range_active = [ns for ns in n_samples_range
                              if ns in time_sdp_cvx]
    plt.loglog(n_samples_range_active,
               [time_sdp_cvx[ns] for ns in n_samples_range_active],
               linewidth=2,
               label=r'standard SDP solver')

    for rf in time_sdp_bm:
        n_samples_range_active = [ns for ns in n_samples_range
                                  if ns in time_sdp_bm[rf]]
        plt.loglog(n_samples_range_active,
                   [time_sdp_bm[rf][ns] for ns in n_samples_range_active],
                   linewidth=2,
                   label=r'non-convex SDP solver ($r={}$)'.format(k * rf))

    n_samples_range_active = [ns for ns in n_samples_range
                              if ns in time_sdp_cgm]
    plt.loglog(n_samples_range_active,
               [time_sdp_cgm[ns] for ns in n_samples_range_active],
               linewidth=2,
               label=r'conditional gradient SDP solver')

    plt.xlabel('Dataset size ($n$)', fontsize='x-large')
    plt.ylabel('Time (s)', fontsize='x-large')
    plt.xlim(min(n_samples_range_active) - 3, max(n_samples_range))
    plt.ylim(1e-2, 1e4)
    plt.yticks(10. ** np.arange(-1, 5))
    plt.legend(loc='lower right', fontsize='x-large')
    plt.savefig('{}mnist_timing.pdf'.format(dir_name))


if __name__ == '__main__':
    # Beware: this test can take a really long time!

    logger = Logger(dir_name + 'test_clustering_real.txt')
    sys.stdout = logger

    n_samples_range = list(range(50, 1000, 50))
    n_samples_range += list(range(1000, 2000, 100))
    n_samples_range += list(range(2000, 10001, 1000))

    mnist_timing(16, n_samples_range, rank_factors=[4, 8], digit=1,
                 from_file=False)

    mnist_timing(200, [5000], rank_factors=[2, 4, 8], digit=0,
                 from_file=False)

    plt.show()

    sys.stdout = logger.stdout
    logger.close()
