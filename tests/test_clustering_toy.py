from __future__ import absolute_import, print_function
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn.apionly as sns
from sdp_kmeans import connected_components, sdp_kmeans
from data import toy
from tests.utils import plot_matrix, plot_data_clustered

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'clustering/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def test_multilayer(X, gt, n_clusters, filename):
    D, Q = sdp_kmeans(X, [n_clusters])
    Q_labels, _ = connected_components(Q)

    sns.set_style('white')
    plt.figure(figsize=(12, 6), tight_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3])

    gs_in = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0],
                                             wspace=0, hspace=0,
                                             height_ratios=(0.5, 1, 0.5))
    ax = plt.subplot(gs_in[1, :])
    plot_data_clustered(X, gt, ax=ax)

    gs_in = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[1],
                                             wspace=.05, hspace=0)

    titles = ['Input Gramian',
              '$\mathbf{{Q}}$ ($K={0}$)'.format(n_clusters),
              'Connected components']
    for i, (M, t) in enumerate(zip([D, Q, Q_labels], titles)):
        reps = M.dot(X)

        ax = plt.subplot(gs_in[0, i])
        plot_matrix(M, ax=ax)
        ax.set_title(t, fontsize='xx-large')

        ax = plt.subplot(gs_in[1, i])
        plot_data_clustered(reps, gt, marker='x', ax=ax)

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
