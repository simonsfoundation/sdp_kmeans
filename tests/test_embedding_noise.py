from __future__ import absolute_import, print_function
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import seaborn.apionly as sns
from sdp_kmeans.embedding import sdp_kmeans_embedding
from data import toy
from tests.utils import plot_matrix, plot_data_embedded

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'embedding/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def test_toy_embedding(X, n_clusters, target_dim, filename):
    print('--------\n', filename)

    embedding, Ds = sdp_kmeans_embedding(X, n_clusters, target_dim,
                                         ret_sdp=True)

    sns.set_style('whitegrid')
    plt.figure(figsize=(16, 4), tight_layout=True)
    gs = gridspec.GridSpec(1, 4, width_ratios=(1.5, 1, 1, 1),
                           wspace=0.1)

    if X.shape[1] == 2:
        ax = plt.subplot(gs[0])
    if X.shape[1] == 3:
        ax = plt.subplot(gs[0], projection='3d')
    plot_data_embedded(X, ax=ax)

    for i, D_input in enumerate(Ds):
        ax = plt.subplot(gs[i+1])
        plot_matrix(D_input, ax=ax)
        if i == 0:
            ax.set_title('Input Gramian $\mathbf{{D}}$', fontsize='xx-large')
        else:
            title = '$\mathbf{{Q}}$ ($K={0}$)'.format(n_clusters)
            ax.set_title(title, fontsize='xx-large')

    ax = plt.subplot(gs[3])
    plot_data_embedded(embedding, ax=ax)

    plt.savefig('{}{}.pdf'.format(dir_name, filename))


def test_trefoil():
    for k in [10]:
        for std in np.arange(0, 0.21, 0.01):
            X = toy.trefoil_knot(n_samples=200)
            if std > 0:
                X += np.random.normal(scale=std, size=X.shape)
            filename = 'trefoil_knot_k{}_noise{:.2f}'.format(k, std)
            test_toy_embedding(X, k, 2, filename)


if __name__ == '__main__':
    test_trefoil()

    # plt.show()
