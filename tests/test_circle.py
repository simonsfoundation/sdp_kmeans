from __future__ import absolute_import, print_function
import cvxpy as cp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.linalg import dft
import seaborn.apionly as sns
from sdp_kmeans import sdp_kmeans, dot_matrix
from data import toy
from tests.utils import plot_matrix, plot_data_clustered

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'circle/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def test_circles():
    n_clusters = 16
    filename = 'circles_eigendecomposition'

    X, gt = toy.circles()
    D, Q = sdp_kmeans(X, n_clusters)

    sns.set_style('white')
    sns.set_color_codes()

    plt.figure(figsize=(12, 2))
    gs = gridspec.GridSpec(2, 7, wspace=0.05, hspace=0.05,
                           height_ratios=(0.45, 0.45))

    ax = plt.subplot(gs[:, 0])
    plot_data_clustered(X, gt, ax=ax)

    titles = ['Input Gramian',
              '$\mathbf{{Q}}$ ($K={0}$)'.format(n_clusters)]
    for i, (M, t) in enumerate(zip([D, Q], titles)):
        ax = plt.subplot(gs[:, 2 * i + 1])
        plot_matrix(M, ax=ax)
        ax.set_title(t, fontsize='x-large')

        eigvals, eigvecs = np.linalg.eigh(M)
        mask = eigvals >= 1e-2
        eigvecs = eigvecs[:, mask]
        Y = eigvecs.dot(np.diag(np.sqrt(eigvals[mask])))
        Y = Y[:, ::-1]
        eigvecs = eigvecs[:, ::-1]

        if i == 0:
            ax = plt.subplot(gs[:, 2 * i + 2])
            ax.plot(Y[:, 0])
            ax.plot(Y[:, 1])
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_aspect(100)

        else:
            s1 = np.sum(np.abs(eigvecs[:100, :]), axis=0)
            s2 = np.sum(np.abs(eigvecs[100:, :]), axis=0)

            ax = plt.subplot(gs[0, 2 * i + 2])
            for idx in np.where(s1 > s2)[0]:
                ax.plot(Y[:, idx])
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_aspect(352, anchor='S')
            ax = plt.subplot(gs[1, 2 * i + 2])
            for idx in np.where(s1 < s2)[0]:
                ax.plot(Y[:, idx])
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_aspect(352, anchor='N')

            ax = plt.subplot(gs[0, 5])
            for idx in np.where(s1 > s2)[0][:3]:
                ax.plot(Y[:, idx])
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_aspect(352, anchor='S')
            ax = plt.subplot(gs[1, 5])
            for idx in np.where(s1 < s2)[0][:3]:
                ax.plot(Y[:, idx])
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_aspect(352, anchor='N')

    plt.savefig('{}{}.pdf'.format(dir_name, filename))


def test_one_circle():
    X, gt = toy.circles(n_samples=200)
    X = X[gt == 0, :]
    gt = gt[gt == 0]

    k_range = np.arange(1, len(X) + 1)
    solutions = []
    for k in k_range:
        D, Q = sdp_kmeans(X, k)
        solutions.append(Q)

    sns.set_style('white')
    sns.set_color_codes()

    plt.figure(figsize=(6, 6))
    plot_data_clustered(X, gt)

    plt.savefig('{}{}.pdf'.format(dir_name, 'circle'))

    plt.figure(figsize=(6, 6))
    n_diags = 10
    palette = sns.color_palette('hls', n_colors=n_diags + 1)
    for k, Q in zip(k_range, solutions):
        Q[Q < 1e-3 * Q.max()] = 0

        values = np.unique(Q)
        try:
            abscissa = [k] * len(values)
        except TypeError:
            abscissa = k
        plt.scatter(abscissa, values, marker='o', s=5, c=palette[-1],
                   edgecolors='none')

        for i in range(n_diags):
            values = np.unique(np.diag(Q, k=i))
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
    for k, Q in zip(k_range, solutions):
        eigvals = np.diag(F.dot(Q).dot(np.conjugate(F).T))
        mask = eigvals >= (k / (len(X) * 10))
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
    D_sdp = sdp_kmeans(X, k)[1]
    D_lp, q = circle_lp(X, 16)

    eigvals, _ = np.linalg.eigh(D_sdp)
    eigvals = eigvals[::-1]

    sns.set_style('white')
    sns.set_color_codes()

    plt.figure()
    plot_matrix(D_sdp)
    plt.title('SDP solution', fontsize='xx-large')
    plt.savefig('{}{}.pdf'.format(dir_name, 'circle_sdp'))

    plt.figure()
    plot_matrix(D_lp)
    plt.title('LP solution', fontsize='xx-large')
    plt.savefig('{}{}.pdf'.format(dir_name, 'circle_lp'))

    plt.figure(figsize=(8, 3))
    gs = gridspec.GridSpec(1, 2, wspace=0.3)

    ax = plt.subplot(gs[0])
    plt.plot(q, linewidth=3, label='LP')
    plt.plot(eigvals, linewidth=3, linestyle=':', label='SDP')
    plt.xlim(0, len(eigvals))
    plt.ylim(0, 1)
    plt.xlabel('Eigenvalues', fontsize='xx-large')
    plt.legend()
    ax.set_aspect(80, anchor='N')

    ax = plt.subplot(gs[1])
    plt.plot(np.zeros_like(eigvals), color='#636363', linewidth=1)
    plt.plot(eigvals - q, color='#4daf4a', linewidth=3)
    plt.xlim(0, len(eigvals))
    plt.xlabel('Eigenvalue differences', fontsize='xx-large')
    plt.legend()
    ax.set_aspect(24350, anchor='N')

    plt.savefig('{}{}.pdf'.format(dir_name, 'circle_sdp_lp'))

    print(np.linalg.norm(eigvals - q) / np.linalg.norm(eigvals))
    print(np.linalg.norm(D_sdp - D_lp, 'fro') / np.linalg.norm(D_sdp, 'fro'))


if __name__ == '__main__':
    test_circles()
    test_one_circle()
    test_circle_sdp_lp()

    plt.show()
