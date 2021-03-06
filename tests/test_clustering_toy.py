import colorcet as cc
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sdp_kmeans import sdp_kmeans
from data import toy
from tests.utils import plot_matrix, plot_data_clustered

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'clustering/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def test_clustering(X, gt, n_clusters, filename):
    D, Q = sdp_kmeans(X, n_clusters)

    sns.set_style('white')
    plt.figure(figsize=(9, 5), tight_layout=True)

    ax = plt.subplot(131)
    plot_data_clustered(X, gt, ax=ax)
    ax.set_title('Input dataset', fontsize='xx-large')

    titles = ['Input Gramian $\mathbf{{D}}$',
              '$\mathbf{{Q}}$ ($K={0}$)'.format(n_clusters),
              '$\mathbf{{Q}}$ (enhanced contrast)']
    for i, (M, t) in enumerate(zip([D, Q], titles)):
        ax = plt.subplot(1, 3, i + 2)
        plot_matrix(M, ax=ax)
        ax.set_title(t, fontsize='xx-large')

    plt.tight_layout()
    plt.savefig('{}{}.pdf'.format(dir_name, filename))


if __name__ == '__main__':
    X, gt = toy.gaussian_blobs()
    test_clustering(X, gt, 6, 'gaussian_blobs')

    X, gt = toy.circles()
    test_clustering(X, gt, 16, 'circles')

    X, gt = toy.moons()
    test_clustering(X, gt, 16, 'moons')

    X, gt = toy.double_swiss_roll()
    test_clustering(X, gt, 64, 'double_swiss_roll')

    plt.show()
