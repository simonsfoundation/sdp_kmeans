from __future__ import absolute_import, print_function
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn.apionly as sns
import sklearn.cluster as sk_cluster
from sdp_kmeans.embedding import spectral_embedding
from sdp_kmeans.sdp import sdp_kmeans_multilayer
from data import toy, real
from tests.utils import plot_matrix, plot_data_clustered, plot_data_embedded,\
    plot_images_embedded

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'cluster-embedding/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def joint_clustering_embedding(X, layer_sizes, target_dim):
    Ds = sdp_kmeans_multilayer(X, layer_sizes)

    n_clusters = layer_sizes[-1]
    model = sk_cluster.KMeans(n_clusters=n_clusters, random_state=0)
    labels = model.fit_predict(Ds[-1].dot(X))

    embeddings = {}
    clusters = {}
    for k in range(n_clusters):
        mask = labels == k
        D_crop = Ds[1][mask, :][:, mask]
        embeddings[k] = spectral_embedding(D_crop, target_dim=target_dim)
        clusters[k] = mask

    return Ds, embeddings, clusters


def test_thin_lines(layer_sizes, target_dim):
    X, gt = toy.thin_lines(n_samples=200)
    filename = 'thin_lines'

    Ds, embeddings, clusters = joint_clustering_embedding(X, layer_sizes,
                                                          target_dim)
    n_clusters = layer_sizes[-1]

    if len(layer_sizes) > 1:
        figsize = (16, 6)
    else:
        figsize = None

    sns.set_style('white')
    plt.figure(figsize=figsize, tight_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, len(layer_sizes)])

    gs_in = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0],
                                             wspace=0, hspace=0,
                                             height_ratios=(0.5, 1, 0.5))
    ax = plt.subplot(gs_in[1, :])
    if X.shape[1] == 2:
        plot_data_clustered(X, gt, ax=ax)
    else:
        X_emb = spectral_embedding(X, target_dim=2)
        plot_data_clustered(X_emb, gt, ax=ax)

    ax = plt.subplot(gs[1])
    ax.axis('off')
    gs_in = gridspec.GridSpecFromSubplotSpec(2, len(Ds), subplot_spec=gs[1],
                                             hspace=0.1,
                                             height_ratios=(1, 1))

    for i, D_input in enumerate(Ds):
        reps = D_input.dot(X)

        ax = plt.subplot(gs_in[0, i])
        plot_matrix(D_input, ax=ax)
        if i == 0:
            ax.set_title('Original Gramian')
        else:
            ax.set_title('Layer {} (K={})'.format(i, layer_sizes[i-1]))

        ax = plt.subplot(gs_in[1, i])
        if X.shape[1] == 2:
            plot_data_clustered(reps, gt, marker='x', ax=ax)
        else:
            reps_emb = spectral_embedding(reps, target_dim=2)
            plot_data_clustered(reps_emb, gt, ax=ax)

    str_layer_sizes = '-'.join([str(ls) for ls in layer_sizes])
    plt.savefig('{}{}_solution_{}.pdf'.format(dir_name, filename,
                                              str_layer_sizes))

    if X.shape[1] == 2:
        reps = Ds[1].dot(X)
    else:
        reps = spectral_embedding(Ds[1].dot(X), target_dim=2)

    plt.figure()
    for k in range(n_clusters):
        emb = embeddings[k]
        mask = clusters[k]
        if target_dim == 1:
            emb = np.hstack((emb, np.zeros_like(emb)))
            emb *= 3
            emb += np.mean(reps[mask, :], axis=0)
        plot_data_embedded(emb)
    plt.axis('equal')

    plt.savefig('{}{}_embedding_{}.pdf'.format(dir_name, filename,
                                               str_layer_sizes))


def test_turntable(layer_sizes, target_dim):
    X, gt = real.turntable(objects=['Horse', 'Lamp'])
    filename = 'turntable'

    Ds, embeddings, clusters = joint_clustering_embedding(X, layer_sizes,
                                                          target_dim)
    n_clusters = layer_sizes[-1]

    if len(layer_sizes) > 1:
        figsize = (16, 5)
    else:
        figsize = None

    sns.set_style('white')
    plt.figure(figsize=figsize, tight_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, len(layer_sizes)])

    gs_in = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0],
                                             wspace=0, hspace=0,
                                             height_ratios=(0.5, 1, 0.5))
    ax = plt.subplot(gs_in[1, :])
    if X.shape[1] == 2:
        plot_data_clustered(X, gt, ax=ax)
    else:
        X_emb = spectral_embedding(X, target_dim=2)
        plot_data_clustered(X_emb, gt, ax=ax)

    ax = plt.subplot(gs[1])
    ax.axis('off')
    gs_in = gridspec.GridSpecFromSubplotSpec(2, len(Ds), subplot_spec=gs[1],
                                             hspace=0.1,
                                             height_ratios=(1, 1))

    for i, D_input in enumerate(Ds):
        reps = D_input.dot(X)

        ax = plt.subplot(gs_in[0, i])
        plot_matrix(D_input, ax=ax)
        if i == 0:
            ax.set_title('Original Gramian')
        else:
            ax.set_title('Layer {} (K={})'.format(i, layer_sizes[i-1]))

        ax = plt.subplot(gs_in[1, i])
        if X.shape[1] == 2:
            plot_data_clustered(reps, gt, marker='x', ax=ax)
        else:
            reps_emb = spectral_embedding(reps, target_dim=2)
            plot_data_clustered(reps_emb, gt, ax=ax)

    str_layer_sizes = '-'.join([str(ls) for ls in layer_sizes])
    plt.savefig('{}{}_solution_{}.pdf'.format(dir_name, filename,
                                              str_layer_sizes))

    for k in range(n_clusters):
        emb = embeddings[k]
        mask = clusters[k]
        X_mask = X[mask, :]

        def turnable_img(k):
            return X_mask[k, :].reshape((384, 512, 3))

        plt.figure()
        plot_images_embedded(emb, turnable_img, subsampling=5, zoom=0.07)

        str_format = '{}{}_embedding_{}_cluster{}.pdf'
        plt.savefig(str_format.format(dir_name, filename, str_layer_sizes, k))


if __name__ == '__main__':
    # test_thin_lines([32, 16, 8, 6], 1)
    test_turntable([32, 16, 8, 4, 2], 2)

    plt.show()
