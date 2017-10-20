from __future__ import absolute_import, print_function
import matplotlib.pyplot as plt
import os
import seaborn.apionly as sns
from sdp_kmeans import log_scale, sdp_kmeans
from data import toy
from tests.utils import plot_matrix, plot_data_clustered

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'clustering/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def test_multilayer(X, gt, n_clusters, filename):
    D, Q = sdp_kmeans(X, n_clusters)
    Q_log = log_scale(Q)

    sns.set_style('white')
    plt.figure(figsize=(12, 6), tight_layout=True)

    ax = plt.subplot(141)
    plot_data_clustered(X, gt, ax=ax)
    ax.set_title('Input dataset', fontsize='xx-large')

    titles = ['Input Gramian $\mathbf{{D}}$',
              '$\mathbf{{Q}}$ ($K={0}$)'.format(n_clusters),
              '$\mathbf{{Q}}$ (enhanced contrast)']
    for i, (M, t) in enumerate(zip([D, Q, Q_log], titles)):
        ax = plt.subplot(1, 4, i + 2)
        plot_matrix(M, ax=ax)
        ax.set_title(t, fontsize='xx-large')

    plt.tight_layout()
    plt.savefig('{}{}.pdf'.format(dir_name, filename))


if __name__ == '__main__':
    X, gt = toy.gaussian_blobs()
    test_multilayer(X, gt, 16, 'gaussian_blobs')

    X, gt = toy.circles()
    test_multilayer(X, gt, 16, 'circles')

    X, gt = toy.moons()
    test_multilayer(X, gt, 16, 'moons')

    X, gt = toy.double_swiss_roll()
    test_multilayer(X, gt, 64, 'double_swiss_roll')

    plt.show()
