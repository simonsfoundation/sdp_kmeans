from __future__ import absolute_import, print_function
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sdp_kmeans import dot_matrix, sdp_km
from data import toy
from tests.utils import plot_matrix, plot_data_clustered

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'multilayer/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def test_multilayer(X, gt, layer_sizes, filename, figsize=(15, 3.5)):
    sns.set_style('white')
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 6 + 2, wspace=.2)

    ax = plt.subplot(gs[0, 0])
    plot_data_clustered(X, gt, ax=ax)

    D = dot_matrix(X)

    ax = plt.subplot(gs[0, 1])
    plot_matrix(D, ax=ax, labels=gt)
    ax.set_title('Input Gramian', fontsize='xx-large')

    for i, k in enumerate(layer_sizes):
        Q = sdp_km(D, k, max_iters=100)

        ax = plt.subplot(gs[0, i + 2])
        plot_matrix(Q, ax=ax, labels=gt)
        ax.set_title(r'L{0}'
                     '\n'
                     r'$\mathbf{{Q}}$ ($K={1}$)'.format(i+1, k),
                     fontsize='xx-large')

        D = Q

    size_str = ''.join(['-{:d}'.format(k) for k in layer_sizes])
    plt.savefig('{}{}'.format(dir_name, filename) + size_str +'.pdf')


def main():
    X, gt = toy.circles()
    test_multilayer(X, gt, [8, 4, 2], 'circles')
    test_multilayer(X, gt, [12, 6, 2], 'circles')
    test_multilayer(X, gt, [16, 4, 2], 'circles')
    test_multilayer(X, gt, [16, 8, 4, 2], 'circles')
    test_multilayer(X, gt, [16, 12, 8, 6, 4, 2], 'circles')
    test_multilayer(X, gt, [32, 16, 8, 4, 2], 'circles')

    X, gt = toy.moons()
    test_multilayer(X, gt, [8, 4, 2], 'moons')
    test_multilayer(X, gt, [12, 6, 2], 'moons')
    test_multilayer(X, gt, [16, 4, 2], 'moons')
    test_multilayer(X, gt, [16, 8, 4, 2], 'moons')
    test_multilayer(X, gt, [16, 12, 8, 6, 4, 2], 'moons')
    test_multilayer(X, gt, [32, 16, 8, 4, 2], 'moons')
    test_multilayer(X, gt, [64, 32, 16, 8, 4, 2], 'moons')

    X, gt = toy.double_swiss_roll()
    test_multilayer(X, gt, [8, 4, 2], 'double_swiss_roll')
    test_multilayer(X, gt, [12, 6, 2], 'double_swiss_roll')
    test_multilayer(X, gt, [16, 4, 2], 'double_swiss_roll')
    test_multilayer(X, gt, [16, 8, 4, 2], 'double_swiss_roll')
    test_multilayer(X, gt, [16, 12, 8, 6, 4, 2], 'double_swiss_roll')
    test_multilayer(X, gt, [32, 16, 8, 4, 2], 'double_swiss_roll')
    test_multilayer(X, gt, [64, 32, 16, 8, 4, 2], 'double_swiss_roll')
    test_multilayer(X, gt, [64, 16, 4, 2], 'double_swiss_roll')
    test_multilayer(X, gt, [64, 8, 4, 2], 'double_swiss_roll')

    plt.show()


if __name__ == '__main__':
    main()