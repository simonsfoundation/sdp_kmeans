from __future__ import absolute_import, print_function
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
from scipy.stats import zscore
import seaborn as sns
import sklearn.manifold as skman
import sklearn.neighbors as skneigh
from sdp_kmeans import sdp_kmeans
from data import toy, real
from tests.utils import plot_matrix, plot_data_clustered

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'geodesics/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def all_paths_length(mat):
    graph = nx.from_numpy_array(mat)
    path_length = dict(nx.all_pairs_dijkstra_path_length(graph))
    geodesics = np.zeros(mat.shape)
    for u in range(mat.shape[0]):
        for v in range(mat.shape[1]):
            if v in path_length[u]:
                geodesics[u, v] = path_length[u][v]
            else:
                geodesics[u, v] = np.inf
    return geodesics


def run_geodesics_preservation(X, labels, orthogonal_noise=True, noise=0.05,
                               n_neighbors=10, keep_label=None, data_name=None):
    if keep_label is not None:
        X = X[labels == keep_label, :]

    X = zscore(X)

    model_nn = skneigh.NearestNeighbors(n_neighbors=n_neighbors)

    model_nn.fit(X)
    adj_gt = model_nn.kneighbors_graph(X=X, mode='distance')

    np.random.seed(0)
    if orthogonal_noise:
        X_noisy = np.hstack((X,
                             noise * np.random.randn(len(X), X.shape[1] * 5)))
    else:
        X_noisy = X + noise * np.random.randn(*X.shape)

    manifold_methods = {
        'LLE': skman.LocallyLinearEmbedding(method='standard', n_neighbors=5,
                                            random_state=0),
        'LLE-MOD': skman.LocallyLinearEmbedding(method='modified',
                                                n_neighbors=5, random_state=0),
        'LTSA': skman.LocallyLinearEmbedding(method='ltsa', n_neighbors=5,
                                             random_state=0),
        'ISOMAP': skman.Isomap(n_neighbors=5),
        'Spectral': skman.SpectralEmbedding(n_neighbors=5, random_state=0),
        'NO-EMB': None,
    }
    adj_manifold = {}
    for method in manifold_methods:
        if manifold_methods[method] is None:
            emb = X_noisy
        else:
            emb = manifold_methods[method].fit_transform(X_noisy)
        model_nn.fit(emb)
        adj = model_nn.kneighbors_graph(X=emb, mode='distance')
        adj_manifold[method] = adj.toarray()

    # plt.figure()
    # plot_data_clustered(X_noisy, labels)

    k = len(X) / n_neighbors
    Q_sdp = sdp_kmeans(X_noisy, k)[1]

    # plt.figure()
    # plot_matrix(Q_sdp)

    diag = np.diag(Q_sdp)
    adj_sdp = diag[:, np.newaxis] + diag[np.newaxis, :] - 2 * Q_sdp
    adj_sdp[adj_sdp < 0] = 0
    adj_sdp[Q_sdp < 1e-5] = 0
    adj_sdp = np.sqrt(adj_sdp)

    geodesics_gt = all_paths_length(adj_gt.toarray())
    geodesics_sdp = all_paths_length(adj_sdp)
    geodesics_manifold = dict([(method, all_paths_length(adj_manifold[method]))
                               for method in manifold_methods])

    percentiles = np.hstack((np.arange(1, 5, 1), np.arange(5, 55, 5)))

    intersections = dict([(method, {}) for method in manifold_methods])
    intersections['NOMAD'] = {}
    for method in intersections:
        for p in percentiles:
            intersections[method][p] = []

    for u in range(len(X)):
        dist_order_gt = np.argsort(geodesics_gt[u])
        dist_order = dict([(method, np.argsort(geodesics_manifold[method][u]))
                           for method in manifold_methods])
        dist_order['NOMAD'] = np.argsort(geodesics_sdp[u])

        for p in percentiles:
            size = np.maximum(int(np.floor(len(X) * p * 0.01)), 1)

            for method in intersections:
                n_matches = np.intersect1d(dist_order_gt[:size],
                                           dist_order[method][:size])
                intersections[method][p].append(len(n_matches) / size)

    df = pd.DataFrame(data=intersections)
    df_mean = df.applymap(np.mean)
    print(df_mean)
    df_mean['Data percentile'] = df_mean.index

    df_mean2 = df_mean.melt(id_vars='Data percentile', var_name='Method',
                            value_name='intersection ratio')

    with sns.axes_style('whitegrid'):
        hue_order = list(df.columns)
        hue_order.remove('NOMAD')
        hue_order.insert(0, 'NOMAD')
        hue_order.remove('ISOMAP')
        hue_order.insert(1, 'ISOMAP')

        dashes = dict(((cn, (2, 2)) for cn in df.columns))
        dashes['NOMAD'] = ''

        plt.figure()
        lp = sns.lineplot(x='Data percentile', y='intersection ratio',
                          data=df_mean2,
                          hue='Method', hue_order=hue_order,
                          style='Method', dashes=dashes)
        lp.axes.set_xlabel(lp.axes.get_xlabel(), fontsize='x-large')
        lp.axes.set_ylabel(lp.axes.get_ylabel(), fontsize='x-large')
        for tick in lp.axes.xaxis.get_major_ticks():
            tick.label.set_fontsize('x-large')
        for tick in lp.axes.yaxis.get_major_ticks():
            tick.label.set_fontsize('x-large')
        plt.legend(loc='lower right', fontsize='x-large')

        if data_name is not None:
            if orthogonal_noise:
                data_name += '_orthogonal_noise{:.2f}'.format(noise)
            else:
                data_name += '_isotropic_noise{:.2f}'.format(noise)
            if keep_label is not None:
                data_name += '_label{}'.format(keep_label)
            data_name = data_name.replace('.', '')
            plt.savefig(dir_name + data_name + '.pdf')
            plt.close()


def main():
    for noise in [0, 0.02, 0.05, 0.10]:
        X, gt = toy.circles(n_samples=400)
        run_geodesics_preservation(X, gt, orthogonal_noise=True, noise=noise,
                                   n_neighbors=10, keep_label=0,
                                   data_name='geodesics_circles')
        run_geodesics_preservation(X, gt, orthogonal_noise=False,  noise=noise,
                                   n_neighbors=10, keep_label=0,
                                   data_name='geodesics_circles')

        X, gt = toy.circles(n_samples=200)
        run_geodesics_preservation(X, gt, orthogonal_noise=True, noise=noise,
                                   n_neighbors=10, keep_label=None,
                                   data_name='geodesics_circles')
        run_geodesics_preservation(X, gt, orthogonal_noise=False, noise=noise,
                                   n_neighbors=10, keep_label=None,
                                   data_name='geodesics_circles')

        X, gt = toy.moons(n_samples=400, noise=None)
        run_geodesics_preservation(X, gt, orthogonal_noise=True, noise=noise,
                                   n_neighbors=10, keep_label=0,
                                   data_name='geodesics_moons')
        run_geodesics_preservation(X, gt, orthogonal_noise=False, noise=noise,
                                   n_neighbors=10, keep_label=0,
                                   data_name='geodesics_moons')

        X, gt = toy.moons(n_samples=200, noise=None)
        run_geodesics_preservation(X, gt, orthogonal_noise=True, noise=noise,
                                   n_neighbors=10, keep_label=None,
                                   data_name='geodesics_moons')
        run_geodesics_preservation(X, gt, orthogonal_noise=False, noise=noise,
                                   n_neighbors=10, keep_label=None,
                                   data_name='geodesics_moons')

        X = real.teapot()
        X = X.astype(np.float) / 255.
        gt = np.ones((len(X),))
        run_geodesics_preservation(X, gt, orthogonal_noise=True, noise=noise,
                                   n_neighbors=3, keep_label=None,
                                   data_name='geodesics_teapot')
        run_geodesics_preservation(X, gt, orthogonal_noise=False, noise=noise,
                                   n_neighbors=3, keep_label=None,
                                   data_name='geodesics_teapot')

    plt.show()


if __name__ == '__main__':
    main()
