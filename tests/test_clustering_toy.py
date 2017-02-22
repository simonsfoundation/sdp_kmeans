from __future__ import absolute_import, print_function
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
import seaborn.apionly as sns
from sdp_kmeans.sdp import sdp_kmeans_multilayer
from data import toy
from tests.utils import plot_matrix, plot_data_clustered

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'sdp_kmeans/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def test_multilayer(cluster_multilayer, X, gt, layer_sizes, filename,
                    figsize=(12, 5.5)):
    Ds = cluster_multilayer(X, layer_sizes)

    sns.set_style('white')
    plt.figure(figsize=figsize, tight_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, len(layer_sizes)])

    gs_in = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0],
                                             wspace=0, hspace=0,
                                             height_ratios=(0.5, 1, 0.5))
    ax = plt.subplot(gs_in[1, :])
    plot_data_clustered(X, gt, ax=ax)

    ax = plt.subplot(gs[1])
    ax.axis('off')
    gs_in = gridspec.GridSpecFromSubplotSpec(2, len(Ds), subplot_spec=gs[1],
                                             wspace=.05, hspace=0)

    for i, D_input in enumerate(Ds):
        reps = D_input.dot(X)

        ax = plt.subplot(gs_in[0, i])
        plot_matrix(D_input, ax=ax)
        if i == 0:
            ax.set_title('Original Gramian')
        else:
            ax.set_title('Layer {0}: '
                         '$\mathbf{{Q}}$ (K={1})'.format(i, layer_sizes[i-1]))

        ax = plt.subplot(gs_in[1, i])
        plot_data_clustered(reps, gt, marker='x', ax=ax)

    plt.savefig('{}{}.pdf'.format(dir_name, filename))


if __name__ == '__main__':
    X, gt = toy.gaussian_blobs()
    test_multilayer(sdp_kmeans_multilayer, X, gt, [16, 8, 6],
                    'gaussian_blobs')

    X, gt = toy.circles()
    test_multilayer(sdp_kmeans_multilayer, X, gt, [16, 8, 4, 2], 'circles',
                    figsize=(12, 4.2))

    X, gt = toy.moons()
    test_multilayer(sdp_kmeans_multilayer, X, gt, [16, 8, 4, 2], 'moons',
                    figsize=(12, 4.2))

    X, gt = toy.double_swiss_roll()
    test_multilayer(sdp_kmeans_multilayer, X, gt, [64, 32, 16, 8, 4, 2],
                    'double_swiss_roll', figsize=(16, 4.3))

    plt.show()
