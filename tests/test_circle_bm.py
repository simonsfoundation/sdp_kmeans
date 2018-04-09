from __future__ import absolute_import, print_function
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn.apionly as sns
from sdp_kmeans import sdp_km_burer_monteiro
from data import toy
from tests.utils import plot_matrix, plot_data_clustered

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'circle_bm/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def test_one_circle(n_clusters=16):
    X, gt = toy.circles(n_samples=200)
    X = X[gt == 0, :]
    gt = gt[gt == 0]

    Y = sdp_km_burer_monteiro(X, n_clusters, rank=n_clusters * 8)
    print(Y.shape)

    Q = Y.dot(Y.T)

    idx = np.argsort(np.argmax(Y, axis=0))
    print(idx.shape, idx)
    Y = Y[:, idx]

    sns.set_style('white')
    sns.set_color_codes()

    plt.figure(figsize=(12, 6), tight_layout=True)
    gs = gridspec.GridSpec(1, 3, width_ratios=(0.78, 0.78, 1))

    ax = plt.subplot(gs[0])
    plot_data_clustered(X, gt, ax=ax)
    ax.set_title('Input dataset', fontsize='xx-large')

    ax = plt.subplot(gs[1])
    plot_matrix(Q, ax=ax)
    ax.set_title('$\mathbf{{Q}}$', fontsize='xx-large')

    ax = plt.subplot(gs[2])
    plot_matrix(Y, ax=ax)
    ax.set_title('$\mathbf{{Y}}$', fontsize='xx-large')

    plt.savefig('{}{}.pdf'.format(dir_name, 'circle_bm'))


if __name__ == '__main__':
    test_one_circle()
    plt.show()
