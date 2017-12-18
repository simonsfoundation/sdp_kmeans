from __future__ import absolute_import, print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn.apionly as sns
from sdp_kmeans import sdp_kmeans, sdp_km_conditional_gradient
from data import real, toy
from tests.utils import plot_matrix, plot_data_clustered

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'cvx_vs_cgm/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def test_cvx_vs_cgm(X, gt, n_clusters, filename):
    D, Q_cvx = sdp_kmeans(X, n_clusters, method='cvx')

    out = sdp_km_conditional_gradient(D, n_clusters, stop_tol=1e-6, verbose=True)
    Q_cgm, cgm_rmse_list, cgm_obj_value_list = out

    cvx_rmse = np.sqrt(np.mean(Q_cvx[Q_cvx < 0] ** 2))
    cvx_error_list = [cvx_rmse] * len(cgm_obj_value_list)
    cvx_obj_value_list = [np.trace(D.dot(Q_cvx))] * len(cgm_obj_value_list)

    sns.set_style('white')
    plt.figure(figsize=(12, 6), tight_layout=True)

    ax = plt.subplot(141)
    plot_data_clustered(X, gt, ax=ax)
    ax.set_title('Input dataset', fontsize='xx-large')

    titles = ['Input Gramian $\mathbf{{D}}$',
              'Standard: $\mathbf{{Q}}$ ($K={0}$)'.format(n_clusters),
              'CGM: $\mathbf{{Q}}$ ($K={0}$)'.format(n_clusters)]
    for i, (M, t) in enumerate(zip([D, Q_cvx, Q_cgm], titles)):
        ax = plt.subplot(1, 4, i + 2)
        plot_matrix(M, ax=ax)
        ax.set_title(t, fontsize='xx-large')

    plt.tight_layout()
    plt.savefig('{}{}_cvx_vs_cgm.pdf'.format(dir_name, filename))

    _, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].loglog(cgm_rmse_list, linewidth=2)
    axes[0].semilogx(cvx_error_list, linestyle=':', linewidth=2)
    axes[0].set_ylabel('RMSE', fontsize='x-large')
    axes[0].set_xlabel('Iterations', fontsize='x-large')

    axes[1].semilogx(cgm_obj_value_list, linewidth=2)
    axes[1].semilogx(cvx_obj_value_list, linestyle=':', linewidth=2)
    axes[1].set_ylabel('Objective value', fontsize='x-large')
    axes[1].set_xlabel('Iterations', fontsize='x-large')

    plt.tight_layout()
    plt.savefig('{}{}_cgm_convergence.pdf'.format(dir_name, filename))


if __name__ == '__main__':
    X, gt = toy.gaussian_blobs()
    test_cvx_vs_cgm(X, gt, 16, 'gaussian_blobs')

    X, gt = toy.circles()
    test_cvx_vs_cgm(X, gt, 16, 'circles')

    X, gt = toy.moons()
    test_cvx_vs_cgm(X, gt, 16, 'moons')

    X, gt = toy.double_swiss_roll()
    test_cvx_vs_cgm(X, gt, 64, 'double_swiss_roll')

    X = toy.trefoil_knot(n_samples=200)
    gt = np.zeros((len(X),))
    test_cvx_vs_cgm(X, gt, 16, 'trefoil_knot')

    X = real.teapot()
    gt = np.zeros((len(X),))
    test_cvx_vs_cgm(X, gt, 20, 'trefoil_knot')

    plt.show()
