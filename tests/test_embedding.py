from __future__ import absolute_import, print_function
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    plt.figure(figsize=(16, 4), tight_layout=True)
    gs = gridspec.GridSpec(1, 4, width_ratios=(1.5, 1, 1, 1),
                           wspace=0.1)

    if X.shape[1] == 2:
        ax = plt.subplot(gs[0])
    if X.shape[1] == 3:
        ax = plt.subplot(gs[0], projection='3d')
    plot_data_embedded(X, ax=ax, palette=palette, elev_azim=elev_azim)

    titles = ['Input Gramian $\mathbf{{D}}$',
              '$\mathbf{{Q}}$ ($K={0}$)'.format(n_clusters)]
    for i, (M, t) in enumerate(zip([D, Q], titles)):
        ax = plt.subplot(gs[i+1])
        plot_matrix(M, ax=ax)
        ax.set_title(t, fontsize='xx-large')

    if target_dim == 2:
        ax = plt.subplot(gs[3])
    if target_dim == 3:
        ax = plt.subplot(gs[3], projection='3d')
    plot_data_embedded(embedding, ax=ax, palette=palette)

    plt.savefig('{}{}.pdf'.format(dir_name, filename))


def test_real_embedding(X, n_clusters, target_dim, img_getter, filename,
                        subsampling=10, zoom=.5, labels=None, palette='hls'):
    print('--------\n', filename)

    embedding, D, Q = sdp_kmeans_embedding(X, n_clusters, target_dim,
                                           ret_sdp=True)

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


def test_trefoil():
    X = toy.trefoil_knot(n_samples=200)
    test_toy_embedding(X, 16, 3, 'trefoil_knot')


def test_teapot():
    X = real.teapot()

    def teapot_img(k):
        return X[k, :].reshape((3, 101, 76)).T

    test_real_embedding(X, 20, 2, teapot_img, 'teapots')


def test_mnist(digit=1):
    X = real.mnist(digit=digit, n_samples=500)

    def mnist_img(k):
        return 255. - X[k, :].reshape((28, 28))

    filename = 'mnist{}'.format(digit)
    test_real_embedding(X, 16, 2, mnist_img, filename, subsampling=5, zoom=0.3,
                        palette='none')


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

    plt.show()
