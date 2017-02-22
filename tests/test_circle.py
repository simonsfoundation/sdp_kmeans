from __future__ import absolute_import, print_function
import cvxpy as cp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.linalg import dft
import seaborn.apionly as sns
from sdp_kmeans.sdp import sdp_kmeans_multilayer, dot_matrix
from data import toy
from tests.utils import plot_matrix, plot_data_clustered

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'sdp_kmeans/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def test_circles():
    layer_sizes = [16, 8, 4, 2]
    filename = 'circles_eigendecomposition'

    X, gt = toy.circles()
    Ds = sdp_kmeans_multilayer(X, layer_sizes)

    sns.set_style('white')
    sns.set_color_codes()

    plt.figure(figsize=(12, 4), tight_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, len(layer_sizes)])

    gs_in = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0],
                                             wspace=0, hspace=0,
                                             height_ratios=(0.5, 1, 0.5))
    ax = plt.subplot(gs_in[1, :])
    plot_data_clustered(X, gt, ax=ax)

    ax = plt.subplot(gs[1])
    ax.axis('off')
    gs_in = gridspec.GridSpecFromSubplotSpec(3, len(Ds), subplot_spec=gs[1],
                                             wspace=0.05, hspace=0.05,
                                             height_ratios=(1.2, 0.4, 0.4))

    for i, D_input in enumerate(Ds):
        ax = plt.subplot(gs_in[0, i])
        plot_matrix(D_input, ax=ax)
        if i == 0:
            ax.set_title('Original Gramian')
        else:
            ax.set_title('Layer {} (K={})'.format(i, layer_sizes[i-1]))

        eigvals, eigvecs = np.linalg.eigh(D_input)
        mask = eigvals >= (layer_sizes[i-1] * 1e-2)
        eigvecs = eigvecs[:, mask]
        Y = eigvecs.dot(np.diag(np.sqrt(eigvals[mask])))
        Y = Y[:, ::-1]
        eigvecs = eigvecs[:, ::-1]

        if i == 0:
            ax = plt.subplot(gs_in[1:3, i])
            ax.plot(Y[:, 0])
            ax.plot(Y[:, 1])
            ax.set_yticks([])
            ax.set_xticks([])
        else:
            s1 = np.sum(np.abs(eigvecs[:100, :]), axis=0)
            s2 = np.sum(np.abs(eigvecs[100:, :]), axis=0)

            ax = plt.subplot(gs_in[1, i])
            for idx in np.where(s1 > s2)[0]:
                ax.plot(Y[:, idx])
            ax.set_yticks([])
            ax.set_xticks([])
            ax = plt.subplot(gs_in[2, i])
            for idx in np.where(s1 < s2)[0]:
                ax.plot(Y[:, idx])
            ax.set_yticks([])
            ax.set_xticks([])

    plt.savefig('{}{}.pdf'.format(dir_name, filename))


def test_one_circle():
    X, gt = toy.circles(n_samples=200)
    X = X[gt == 0, :]
    gt = gt[gt == 0]

    k_range = np.arange(1, len(X) + 1)
    solutions = []
    for k in k_range:
        Ds = sdp_kmeans_multilayer(X, [k])
        solutions.append(Ds[-1])

    sns.set_style('white')
    sns.set_color_codes()

    plt.figure(figsize=(6, 6))
    plot_data_clustered(X, gt)

    plt.savefig('{}{}.pdf'.format(dir_name, 'circle'))

    plt.figure(figsize=(6, 6))
    n_diags = 10
    palette = sns.color_palette('hls', n_colors=n_diags + 1)
    for k, D in zip(k_range, solutions):
        D[D < 1e-3 * D.max()] = 0

        values = np.unique(D)
        try:
            abscissa = [k] * len(values)
        except TypeError:
            abscissa = k
        plt.scatter(abscissa, values, marker='o', s=5, c=palette[-1],
                   edgecolors='none')

        for i in range(n_diags):
            values = np.unique(np.diag(D, k=i))
            try:
                abscissa = [k] * len(values)
            except TypeError:
                abscissa = k
            plt.scatter(abscissa, values, marker='o', s=5, c=palette[i],
                       edgecolors='none')

    plt.xlabel(r'$K$', fontdict=dict(size='x-large'))
    plt.xlim(k_range.min() - 0.1, k_range.max() + 0.1)
    plt.ylim(-0.01, 1.01)
    plt.tick_params(axis='both', which='major', labelsize='x-large')

    plt.savefig('{}{}.pdf'.format(dir_name, 'circle_k-evolution'))
    plt.savefig('{}{}.png'.format(dir_name, 'circle_k-evolution'), dpi=300)

    plt.figure(figsize=(6, 6))
    sns.set_color_codes()

    number_of_eigvals = []
    F = dft(len(X), scale='sqrtn')
    for k, D in zip(k_range, solutions):
        eigvals = np.diag(F.dot(D).dot(np.conjugate(F).T))
        mask = eigvals >= (k * 1e-2)
        number_of_eigvals.append(np.sum(mask))

    plt.plot(k_range, number_of_eigvals, marker='o')

    plt.xlabel(r'$K$', fontdict=dict(size='x-large'))
    plt.xlim(k_range.min() - 2, k_range.max() + 2)
    plt.ylim(k_range.min() - 2, k_range.max() + 2)
    plt.tick_params(axis='both', which='major', labelsize='x-large')

    plt.savefig('{}{}.pdf'.format(dir_name, 'circle_k-evolution_eigs'))

    for k in [1, 12, 25, 100]:
        plt.figure(figsize=(6, 6))
        plot_matrix(solutions[k-1])
        plt.savefig('{}{}{}.pdf'.format(dir_name, 'circle_k', k))


def test_circle_sdp_lp():

    def circle_lp(X, k):
        D = dot_matrix(X)

        F = dft(len(X), scale='sqrtn')
        eigenvalues = F.dot(D).dot(np.conjugate(F).T)
        eigvals = np.diag(np.real(eigenvalues))

        def cos_vec(tau, n):
            p = np.arange(n)
            v = np.cos((2 * np.pi * p * tau) / n) / n
            return v[:, np.newaxis]

        n = len(X)

        q = cp.Variable(n, 1)
        objective = cp.Maximize(eigvals * q)
        constraints = [cp.sum_entries(q) == k,
                       q >= 0, q[0] == 1]
        for tau in range(n):
            constraints.append(q.T * cos_vec(tau, n) >= 0)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)
        q = np.asarray(q.value)

        Q_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Q_mat[i, j] = q.T.dot(cos_vec(i - j, n))

        return Q_mat, np.sort(np.squeeze(q))[::-1]

    X, gt = toy.circles(n_samples=200)
    X = X[gt == 0, :]

    k = 16
    D_sdp = sdp_kmeans_multilayer(X, [k])[1]
    D_lp, q = circle_lp(X, 16)

    eigvals, _ = np.linalg.eigh(D_sdp)
    eigvals = eigvals[::-1]

    sns.set_style('white')
    sns.set_color_codes()

    plt.figure(figsize=(8, 8), tight_layout=True)
    ax = plt.subplot(221)
    plot_matrix(D_sdp, ax=ax)
    plt.title('SDP solution', fontdict=dict(size='x-large'))
    ax = plt.subplot(222)
    plt.title('LP solution', fontdict=dict(size='x-large'))
    plot_matrix(D_lp, ax=ax)

    plt.subplot(223)
    plt.plot(eigvals, linewidth=3)
    plt.ylim(0, 1)
    plt.subplot(224)
    plt.plot(q, linewidth=3)
    plt.ylim(0, 1)

    plt.savefig('{}{}.pdf'.format(dir_name, 'circle_sdp_lp'))

    print(np.linalg.norm(eigvals - q) / np.linalg.norm(eigvals))
    print(np.linalg.norm(D_sdp - D_lp, 'fro') / np.linalg.norm(D_sdp, 'fro'))


if __name__ == '__main__':
    # test_circles()
    # test_one_circle()
    test_circle_sdp_lp()

    plt.show()
