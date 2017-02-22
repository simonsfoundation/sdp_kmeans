from __future__ import absolute_import, print_function
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
from scipy.stats import circmean
import seaborn as sns
from sdp_kmeans.nmf import symnmf_admm
from sdp_kmeans.sdp import sdp_kmeans_multilayer, cluster_sdp_burer_monteiro
from data import real, toy
from tests.utils import plot_matrix, plot_data_clustered


dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'reconstruction/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def test_reconstruction(X, gt, layer_size, filename, from_file=False):
    Ds = sdp_kmeans_multilayer(X, [layer_size])

    if from_file:
        data = scipy.io.loadmat('{}{}.mat'.format(dir_name, filename))
        rec_errors = data['rec_errors']
        k_values = data['k_values']
    else:
        k_values = np.arange(200 + len(X)) + 1
        rec_errors = []
        for k in k_values:
            print(k)
            rec_errors_k = []
            for trials in range(50):
                Y = symnmf_admm(Ds[-1], k=k)
                rec_errors_k.append(check_completely_positivity(Ds[-1], Y))
            rec_errors.append(rec_errors_k)
        rec_errors = np.array(rec_errors)
        scipy.io.savemat('{}{}.mat'.format(dir_name, filename),
                         dict(rec_errors=rec_errors,
                              k_values=k_values))

    sns.set_style('white')

    plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(1, 3)

    ax = plt.subplot(gs[0])
    plot_data_clustered(X, gt, ax=ax)

    for i, D_input in enumerate(Ds):
        ax = plt.subplot(gs[i + 1])
        plot_matrix(D_input, ax=ax)
        if i == 0:
            ax.set_title('Original Gramian')
        else:
            ax.set_title('Layer {} (k={})'.format(i, layer_size))
    plt.savefig('{}{}_solution.pdf'.format(dir_name, filename))

    plt.figure(tight_layout=True)
    rec_errors = -np.log10(rec_errors)
    mean = np.mean(rec_errors, axis=1)
    std = np.std(rec_errors, axis=1)
    sns.set_palette('muted')
    plt.fill_between(np.squeeze(k_values), mean - 2 * std, mean + 2 * std,
                     alpha=0.3)
    plt.plot(np.squeeze(k_values), mean, linewidth=2)
    plt.plot([layer_size, layer_size], [mean.min(), mean.max()],
             linestyle='--', linewidth=2)
    plt.xlabel('$r$', size='x-large')
    plt.ylabel('Relative reconstruction error ($-log_{10}$)', size='x-large')
    plt.savefig('{}{}_curve.pdf'.format(dir_name, filename))


def test__circles_visualization(ranks):
    X, gt = toy.circles()
    Ds = sdp_kmeans_multilayer(X, [16])

    def get_y(rank):
        np.random.seed(0)
        Y = symnmf_admm(Ds[-1], k=rank, tol=1e-7)

        print('rank={}'.format(rank))
        print((np.linalg.norm(Ds[-1] - Y.dot(Y.T), 'fro')
               / np.linalg.norm(Ds[-1], 'fro')))

        def find_order(keys, cutoff=1e-2):
            Y0 = Y[gt == keys[0], :]
            Y1 = Y[gt == keys[1], :]

            cols0 = Y0.sum(axis=0) >= Y1.sum(axis=0)
            angles0 = np.arctan2(X[gt == 0, 1], X[gt == 0, 0]) + np.pi

            cutoff *= Y0.max()
            idx0, angmean0 = zip(*[(i, circmean(angles0[Y0[:, i] > cutoff],
                                                high=np.pi, low=-np.pi))
                                   for i in np.where(cols0)[0]])
            return [idx0[i] for i in np.argsort(angmean0)]

        order = find_order([0, 1]) + find_order([1, 0])
        return Y[:, order]

    plt.figure(tight_layout=True)
    for i, (r, Y) in enumerate([(r, get_y(r)) for r in ranks]):
        ax = plt.subplot(1, len(ranks), i + 1)
        plot_matrix(Y.T)
        ax.set_title(r'$\mathbf{{Y}}_+$ (r={0})'.format(r), fontsize='x-large')

    plt.savefig('{}circles_reconstruction_Ys_sorted.pdf'.format(dir_name))


def check_completely_positivity(sym_mat, Y):
    error = (np.linalg.norm(sym_mat - Y.dot(Y.T), 'fro')
             / np.linalg.norm(sym_mat, 'fro'))
    return error


def test_burer_monteiro(X, ranks, filename):
    Qs = sdp_kmeans_multilayer(X, [ranks[0]])

    plt.figure(figsize=(16, 4), tight_layout=True)
    plt.subplot(1, len(ranks) + 1, 1)
    plot_matrix(Qs[1])
    plt.title(r'$\mathbf{{Q}}_*$', fontsize='x-large')

    for i, r in enumerate(ranks):
        Y = cluster_sdp_burer_monteiro(X, ranks[0], rank=r)
        Q_nc = Y.dot(Y.T)
        err = np.linalg.norm(Qs[1] - Q_nc, 'fro') / np.linalg.norm(Qs[1], 'fro')

        plt.subplot(1, len(ranks) + 1, i + 2)
        plot_matrix(Q_nc)
        plt.title(r'$\mathbf{{Y}} \mathbf{{Y}}^{{\top}} (r={0})$'.format(r),
                  fontsize='x-large')
        plt.xlabel('{0:2.2f}%'.format(err * 100), fontsize='x-large')
        print('Rank', r, err)

    plt.savefig('{}{}_burer-monteiro.pdf'.format(dir_name, filename))


if __name__ == '__main__':
    X, gt = toy.circles()
    test_reconstruction(X, gt, 16, 'circles', from_file=True)

    X, gt = toy.double_swiss_roll()
    test_reconstruction(X, gt, 64, 'double-swiss-roll', from_file=True)

    X, gt = toy.circles()
    X = X[gt == 0]
    gt = gt[gt == 0]
    test_reconstruction(X, gt, 16, 'ring', from_file=False)

    test__circles_visualization([64, 200])

    X, _ = toy.circles()
    X = X[:100, :]
    test_burer_monteiro(X, [8, 16, 32, 64, 128, 256], 'circles')

    X = real.teapot()
    test_burer_monteiro(X, [20, 40, 60, 80, 100, 120], 'teapot')

    plt.show()
