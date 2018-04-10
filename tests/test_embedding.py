from __future__ import absolute_import, print_function
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import seaborn.apionly as sns
from sdp_kmeans import sdp_kmeans_embedding
from data import toy, real
from tests.utils import plot_matrix, plot_data_embedded, plot_images_embedded

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'embedding/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def test_toy_embedding(X, n_clusters, target_dim, filename, palette='hls',
                       elev_azim=None):
    print('--------\n', filename)

    embedding, D, Q = sdp_kmeans_embedding(X, n_clusters, target_dim,
                                           ret_sdp=True)

    sns.set_style('whitegrid')
    plt.figure(figsize=(12, 6), tight_layout=True)
    gs = gridspec.GridSpec(1, 4, width_ratios=(1, 1, 1, 1))

    if X.shape[1] == 3:
        ax = plt.subplot(gs[0], projection='3d')
    else:
        ax = plt.subplot(gs[0])
    plot_data_embedded(X, ax=ax, palette=palette, elev_azim=elev_azim)
    ax.set_title('Input dataset', fontsize='xx-large')

    titles = ['Input Gramian $\mathbf{{D}}$',
              '$\mathbf{{Q}}$ ($K={0}$)'.format(n_clusters)]
    for i, (M, t) in enumerate(zip([D, Q], titles)):
        ax = plt.subplot(gs[i + 1])
        plot_matrix(M, ax=ax)
        ax.set_title(t, fontsize='xx-large')

    if target_dim == 2:
        ax = plt.subplot(gs[3])
    if target_dim == 3:
        ax = plt.subplot(gs[3], projection='3d')
    plot_data_embedded(embedding, ax=ax, palette=palette)
    ax.set_title('2D embedding', fontsize='xx-large')

    plt.savefig('{}{}.pdf'.format(dir_name, filename))

    return D, Q


def test_real_embedding(X, n_clusters, target_dim, img_getter, filename,
                        subsampling=10, zoom=.5, labels=None, palette='hls',
                        method='cvx'):
    print('--------\n', filename)

    embedding, D, Q = sdp_kmeans_embedding(X, n_clusters, target_dim,
                                           method=method, ret_sdp=True)

    sns.set_style('whitegrid')
    plt.figure(figsize=(12, 4), tight_layout=True)
    gs = gridspec.GridSpec(1, 3, wspace=0.)

    titles = ['Input Gramian $\mathbf{{D}}$',
              '$\mathbf{{Q}}$ ($K={0}$)'.format(n_clusters)]
    for i, (M, t) in enumerate(zip([D, Q], titles)):
        ax = plt.subplot(gs[i])
        plot_matrix(M, ax=ax)
        ax.set_title(t, fontsize='xx-large')

    ax = plt.subplot(gs[2])
    plot_images_embedded(embedding, img_getter, labels=labels,
                         subsampling=subsampling,
                         zoom=zoom, palette=palette, ax=ax)

    plt.savefig('{}{}.pdf'.format(dir_name, filename))

    plt.figure()
    plot_images_embedded(embedding, img_getter, labels=labels,
                         subsampling=subsampling,
                         zoom=zoom, palette=palette)
    plt.savefig('{}{}_embedding.pdf'.format(dir_name, filename))


def test_swiss_roll():
    X = toy.swiss_roll_3d(n_samples=200)
    test_toy_embedding(X, 32, 2, 'swiss_roll_3d', palette='Spectral',
                       elev_azim=(7, -80))


def test_square_grid():
    X = np.mgrid[0:16, 0:16]
    X = X.reshape((len(X), -1)).T

    name = 'square'
    D, Q = test_toy_embedding(X, 32, 2, name, palette='hls')

    def plot_mat_on_data(mat, sample):
        plot_data_embedded(X, palette='w')
        alpha = np.maximum(mat[sample], 0) / mat[sample].max()
        plot_data_embedded(X, palette='#FF0000', alpha=alpha)

    pdf_file_name = '{}{}_plot_{}_on_data_{}{}'

    plt.figure()
    plot_mat_on_data(D, 7 * 16 + 7)
    plt.savefig(pdf_file_name.format(dir_name, name, 'D', 'middle', '.pdf'))

    plt.figure()
    plot_mat_on_data(Q, 7 * 16 + 7)
    plt.savefig(pdf_file_name.format(dir_name, name, 'Q', 'middle', '.pdf'))

    plt.figure(figsize=(12, 3), tight_layout=True)
    gs = gridspec.GridSpec(1, 4, wspace=0.05)
    for i, s in enumerate([59, 84, 138, 163]):
        plt.subplot(gs[i])
        plot_mat_on_data(Q, s)
        plt.savefig(pdf_file_name.format(dir_name, name, 'Q', 'composite',
                                         '.png'))

    for s in range(len(X)):
        plt.figure()
        plot_mat_on_data(Q, s)
        plt.savefig(pdf_file_name.format(dir_name, name, 'Q', s, '.png'))
        plt.close()


def test_trefoil():
    X = toy.trefoil_knot(n_samples=200)
    test_toy_embedding(X, 16, 2, 'trefoil_knot')


def test_teapot():
    X = real.teapot()

    def teapot_img(k):
        return X[k, :].reshape((3, 101, 76)).T

    test_real_embedding(X, 20, 2, teapot_img, 'teapots')


def test_mnist(digit=1, n_samples=500, n_clusters=16, subsampling=5):
    X = real.mnist(digit=digit, n_samples=n_samples)
    print('Number of samples:', X.shape[0])

    def mnist_img(k):
        return 255. - X[k, :].reshape((28, 28))

    filename = 'mnist{}_n{}_k{}'.format(digit, n_samples, n_clusters)
    test_real_embedding(X, n_clusters, 2, mnist_img, filename,
                        subsampling=subsampling, zoom=0.3, palette='none')


def test_yale_faces(subjects=[1]):
    X, gt = real.yale_faces(subjects=subjects)

    def yale_img(k):
        return 255. - X[k, :].reshape((192, 168))

    filename = 'yale' + '-'.join([str(s) for s in subjects])
    test_real_embedding(X, 16, 2, yale_img, filename, subsampling=3, zoom=0.1,
                        labels=gt, palette='none')


if __name__ == '__main__':
    test_trefoil()
    test_teapot()
    for i in range(10):
        test_mnist(digit=i)
    test_yale_faces(subjects=[1])
    test_yale_faces(subjects=[1, 4])
    test_yale_faces(subjects=[1, 4, 5])
    test_yale_faces(subjects=[1, 4, 37])
    test_yale_faces(subjects=[1, 4, 5, 27])
    test_square_grid()

    # large-scale example, cannot use CVX
    for n_clusters in [32, 64, 96, 128]:
        test_mnist(digit=0, n_samples=0, n_clusters=n_clusters, method='cgm',
                   subsampling=40)

    plt.show()
