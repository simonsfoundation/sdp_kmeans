from __future__ import absolute_import, print_function
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import linear_sum_assignment
import seaborn.apionly as sns
import sklearn.cluster as sk_cluster
import sklearn.metrics as sk_metrics
import sys
from sdp_kmeans.nmf import symnmf_admm
from sdp_kmeans.sdp import sdp_kmeans_multilayer
from data import real
from tests.utils import plot_confusion_matrix, line_plot_clustered,\
    plot_matrix, Logger

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'sdp_kmeans/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def confusion_matrix(gt, labels):
    conf_mat = sk_metrics.confusion_matrix(gt, labels)
    row_ind, col_ind = linear_sum_assignment(-conf_mat)
    conf_mat = conf_mat.take(row_ind, axis=0).take(col_ind, axis=1)
    return conf_mat


def test_multilayer(X, gt, layer_sizes, filename,
                    figsize=(12, 5.5)):
    print('--------\n', filename, '\n--------')
    str_layer_sizes = '-'.join([str(ls) for ls in layer_sizes])

    Ds = sdp_kmeans_multilayer(X, layer_sizes)

    factors_sdp = symnmf_admm(Ds[-1], k=layer_sizes[-1])
    labels_sdp = np.argmax(factors_sdp, axis=1)
    conf_mat_sdp = confusion_matrix(gt, labels_sdp)
    print('SDP K-means', layer_sizes)
    print('adjusted_mutual_info_score',
          sk_metrics.adjusted_mutual_info_score(gt, labels_sdp))
    print('confusion_matrix:\n', conf_mat_sdp)

    plot_confusion_matrix(conf_mat_sdp)
    plt.savefig('{}{}_{}_confmat_sdpkm.pdf'.format(dir_name, filename,
                                                   str_layer_sizes))

    model = sk_cluster.KMeans(n_clusters=layer_sizes[-1], random_state=0)
    labels_kmeans = model.fit_predict(X)
    D_kmeans = sum([np.outer(labels_kmeans == k, labels_kmeans == k)
                    for k in range(layer_sizes[-1])])
    conf_mat_kmeans = confusion_matrix(gt, labels_kmeans)
    print('K-means')
    print('adjusted_mutual_info_score',
          sk_metrics.adjusted_mutual_info_score(gt, labels_kmeans))
    print('confusion_matrix:\n', conf_mat_kmeans)

    plot_confusion_matrix(conf_mat_kmeans)
    plt.savefig('{}{}_{}_confmat_kmeans.pdf'.format(dir_name, filename,
                                                       str_layer_sizes))

    amis_max = 0
    dist_matrix = sk_metrics.pairwise_distances(X)
    for scale in np.linspace(np.maximum(dist_matrix.min(), 1e-6),
                                        dist_matrix.max(), 100):
        affinity = np.exp(-dist_matrix ** 2 / (2. * scale ** 2))
        model = sk_cluster.SpectralClustering(n_clusters=layer_sizes[-1],
                                              affinity='precomputed',
                                              random_state=0)
        labels_spectral = model.fit_predict(affinity)

        amis = sk_metrics.adjusted_mutual_info_score(gt, labels_spectral)
        if amis > amis_max:
            amis_max = amis
            conf_mat_spectral = confusion_matrix(gt, labels_spectral)
            D_spectral = sum([np.outer(labels_spectral == k,
                                       labels_spectral == k)
                              for k in range(layer_sizes[-1])])

    print('Spectral sdp_kmeans - scale={}'.format(scale))
    print('adjusted_mutual_info_score', amis_max)
    print('confusion_matrix:\n', conf_mat_spectral)

    plot_confusion_matrix(conf_mat_spectral)
    plt.savefig('{}{}_{}_confmat_spectral.pdf'.format(dir_name, filename,
                                                      str_layer_sizes))

    sns.set_style('white')
    plt.figure(figsize=figsize, tight_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, len(Ds) + 1])

    gs_in = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0],
                                             wspace=0, hspace=0,
                                             height_ratios=(0.5, 1, 0.5))
    ax = plt.subplot(gs_in[1, :])
    line_plot_clustered(X, gt, ax=ax)
    ax.set_adjustable('box')

    ax = plt.subplot(gs[1])
    ax.axis('off')
    gs_in = gridspec.GridSpecFromSubplotSpec(2, len(Ds) + 1, subplot_spec=gs[1],
                                             wspace=.1, hspace=0,
                                             height_ratios=(1, 0.6))

    for i, D_input in enumerate(Ds):
        reps = D_input.dot(X)

        ax = plt.subplot(gs_in[0, i])
        plot_matrix(D_input, ax=ax)
        if i == 0:
            ax.set_title('Original Gramian')
        else:
            ax.set_title('Layer {} (k={})'.format(i, layer_sizes[i-1]))

        ax = plt.subplot(gs_in[1, i])
        line_plot_clustered(reps, gt, ax=ax)

    ax = plt.subplot(gs_in[0, -1])
    plot_matrix(D_kmeans, ax=ax)
    ax.set_title('K-means (k={})'.format(layer_sizes[-1]))
    ax = plt.subplot(gs_in[1, -1])
    line_plot_clustered(D_kmeans.dot(X), gt, ax=ax)
    ax.set_adjustable('box')

    plt.savefig('{}{}_{}_solution_with_data.pdf'.format(dir_name, filename,
                                                        str_layer_sizes))

    sns.set_style('white')
    plt.figure(figsize=figsize, tight_layout=True)
    gs = gridspec.GridSpec(1, len(Ds) + 2)

    for i, D_input in enumerate(Ds):
        ax = plt.subplot(gs[i])
        plot_matrix(D_input, ax=ax)
        if i == 0:
            ax.set_title('Original Gramian')
        else:
            ax.set_title('Layer {} (k={})'.format(i, layer_sizes[i-1]))

    ax = plt.subplot(gs[-2])
    plot_matrix(D_kmeans, ax=ax)
    ax.set_title('K-means (k={})'.format(layer_sizes[-1]))

    ax = plt.subplot(gs[-1])
    plot_matrix(D_spectral, ax=ax)
    ax.set_title('Spectral (k={})'.format(layer_sizes[-1]))

    str_layer_sizes = '-'.join([str(ls) for ls in layer_sizes])
    plt.savefig('{}{}_{}_solution.pdf'.format(dir_name, filename,
                                              str_layer_sizes))


if __name__ == '__main__':
    logger = Logger(dir_name + 'test_clustering_real.txt')
    sys.stdout = logger

    X, gt = real.digits()
    test_multilayer(X, gt, [5, 4, 3], 'digits',
                    figsize=(12.5, 5.2))
    test_multilayer(X, gt, [3], 'digits',
                    figsize=(8, 5.2))

    X, gt = real.iris()
    test_multilayer(X, gt, [8, 4, 3, 3], 'iris',
                    figsize=(15.5, 5.2))
    test_multilayer(X, gt, [3], 'iris',
                    figsize=(8, 5.2))

    X, gt = real.wine()
    test_multilayer(X, gt, [3, 3, 3], 'wine',
                    figsize=(14.5, 5.2))
    test_multilayer(X, gt, [3], 'wine',
                    figsize=(14.5, 5.2))

    X, gt = real.breast()
    test_multilayer(X, gt, [10, 8, 4, 2], 'breast', figsize=(14.5, 5.2))
    test_multilayer(X, gt, [2], 'breast', figsize=(14.5, 5.2))

    plt.show()

    sys.stdout = logger.stdout
    logger.close()
