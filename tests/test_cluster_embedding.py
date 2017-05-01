from __future__ import absolute_import, print_function
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn.apionly as sns
from sdp_kmeans import connected_components, sdp_kmeans, spectral_embedding
from data import toy, real
from tests.utils import plot_matrix, plot_data_clustered, plot_data_embedded,\
    plot_images_embedded

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'cluster-embedding/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def clustering_embedding(X, n_clusters, target_dim):
    D, Q = sdp_kmeans(X, n_clusters)
    Q_labels, clusters = connected_components(Q)

    embeddings = {}
    for k, mask in enumerate(clusters):
        D_crop = Q[mask, :][:, mask]
        embeddings[k] = spectral_embedding(D_crop, target_dim=target_dim)

    return D, Q, Q_labels, embeddings, clusters


def test_thin_lines(n_clusters, target_dim):
    X, gt = toy.thin_lines(n_samples=200)
    filename = 'thin_lines'

    D, Q, Q_labels, embeddings, clusters = clustering_embedding(X, n_clusters,
                                                                target_dim)

    sns.set_style('white')
    plt.figure(figsize=(10, 5.2), tight_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3])

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
    gs_in = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[1],
                                             wspace=0.05, hspace=0.05)

    titles = ['Input Gramian',
              '$\mathbf{{Q}}$ ($K={0}$)'.format(n_clusters),
              'Connected components']
    for i, (D_input, t) in enumerate(zip([D, Q, Q_labels], titles)):
        reps = D_input.dot(X)

        ax = plt.subplot(gs_in[0, i])
        plot_matrix(D_input, ax=ax)
        ax.set_title(t, fontsize='xx-large')

        ax = plt.subplot(gs_in[1, i])
        if X.shape[1] == 2:
            plot_data_clustered(reps, gt, marker='x', ax=ax)
        else:
            reps_emb = spectral_embedding(reps, target_dim=2)
            plot_data_clustered(reps_emb, gt, ax=ax)

    plt.savefig('{}{}_solution_{}.pdf'.format(dir_name, filename,
                                              n_clusters))

    if X.shape[1] == 2:
        reps = Q.dot(X)
    else:
        reps = spectral_embedding(Q.dot(X), target_dim=2)

    plt.figure()
    for k in range(len(clusters)):
        emb = embeddings[k]
        mask = clusters[k]
        if target_dim == 1:
            emb = np.hstack((emb, np.zeros_like(emb)))
            emb *= 3
            emb += np.mean(reps[mask, :], axis=0)
        plot_data_embedded(emb, palette='Spectral')
    plt.axis('equal')

    plt.savefig('{}{}_embedding_{}.pdf'.format(dir_name, filename,
                                               n_clusters))


def test_turntable(n_clusters, target_dim):
    X, gt = real.turntable(objects=['Horse', 'Lamp'])
    filename = 'turntable'

    D, Q, Q_labels, embeddings, clusters = clustering_embedding(X, n_clusters,
                                                                target_dim)

    sns.set_style('white')
    plt.figure(figsize=(10, 5.2), tight_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3])

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
    gs_in = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[1],
                                             wspace=0.05, hspace=0.05)

    titles = ['Input Gramian',
              '$\mathbf{{Q}}$ ($K={0}$)'.format(n_clusters),
              'Connected components']
    for i, (D_input, t) in enumerate(zip([D, Q, Q_labels], titles)):
        reps = D_input.dot(X)

        ax = plt.subplot(gs_in[0, i])
        plot_matrix(D_input, ax=ax)
        ax.set_title(t, fontsize='xx-large')

        ax = plt.subplot(gs_in[1, i])
        if X.shape[1] == 2:
            plot_data_clustered(reps, gt, marker='x', ax=ax)
        else:
            reps_emb = spectral_embedding(reps, target_dim=2)
            plot_data_clustered(reps_emb, gt, ax=ax)

    plt.savefig('{}{}_solution_{}.pdf'.format(dir_name, filename,
                                              n_clusters))

    for k in range(len(clusters)):
        emb = embeddings[k]
        mask = clusters[k]
        X_mask = X[mask, :]

        def turnable_img(k):
            return X_mask[k, :].reshape((384, 512, 3))

        plt.figure()
        plot_images_embedded(emb, turnable_img, subsampling=5, zoom=0.07)

        str_format = '{}{}_embedding_{}_cluster{}.pdf'
        plt.savefig(str_format.format(dir_name, filename, n_clusters, k))


if __name__ == '__main__':
    test_thin_lines(32, 1)
    test_turntable(35, 2)

    plt.show()
