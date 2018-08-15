from __future__ import absolute_import, print_function
import cvxpy as cp
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from scipy.linalg import dft
import seaborn as sns
from sdp_kmeans import sdp_km, dot_matrix
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
    D = dot_matrix(X)
    Q = sdp_km(D, n_clusters, max_iters=10000, eps=1e-6)

    sns.set_style('white')
    sns.set_color_codes()

    plt.figure(figsize=(7, 4))
    gs = gridspec.GridSpec(3, 4, hspace=0.1, wspace=0.5,
                           height_ratios=(0.6, 0.45, 0.45))

    ax = plt.subplot(gs[:, 0])
    plot_data_clustered(X, gt, ax=ax)
    ax.set_title('Input dataset', fontsize='x-large')

    titles = ['Input Gramian',
              '$\mathbf{{Q}}$ ($K={0}$)'.format(n_clusters)]
    for i, (M, t) in enumerate(zip([D, Q], titles)):
        if i == 0:
            row_slice = 0
        else:
            row_slice = slice(1, 3)
        ax = plt.subplot(gs[row_slice, 1])
        plot_matrix(M, ax=ax, labels=gt)
        ax.set_title(t, fontsize='x-large')

        eigvals, eigvecs = np.linalg.eigh(M)
        mask = eigvals >= 1e-2
        eigvecs = eigvecs[:, mask]
        Y = eigvecs.dot(np.diag(np.sqrt(eigvals[mask])))
        Y = Y[:, ::-1]
        eigvecs = eigvecs[:, ::-1]

        if i == 0:
            ax = plt.subplot(gs[row_slice, 2])
            ax.plot(Y[:, 0])
            ax.plot(Y[:, 1])
            ax.set_xticks([])

        else:
            s1 = np.sum(np.abs(eigvecs[:len(X) // 2, :]), axis=0)
            s2 = np.sum(np.abs(eigvecs[len(X) // 2:, :]), axis=0)

            ax = plt.subplot(gs[1, 2])
            for idx in np.where(s1 > s2)[0]:
                ax.plot(Y[:, idx])
            ax.set_xticks([])

            ax = plt.subplot(gs[2, 2])
            for idx in np.where(s1 < s2)[0]:
                ax.plot(Y[:, idx])
            ax.set_xticks([])

            ax = plt.subplot(gs[1, 3])
            for idx in np.where(s1 > s2)[0][:3]:
                ax.plot(Y[:, idx])
            ax.set_xticks([])

            ax = plt.subplot(gs[2, 3])
            for idx in np.where(s1 < s2)[0][:3]:
                ax.plot(Y[:, idx])
            ax.set_xticks([])

    plt.savefig('{}{}.pdf'.format(dir_name, filename))


def test_one_circle(from_file=False):
    X, gt = toy.circles(n_samples=200)
    X = X[gt == 0, :]
    gt = gt[gt == 0]

    k_range = np.arange(1, len(X) + 1)

    if from_file:
        with open('solutions.pickle', 'rb') as f:
            solutions = pickle.load(f)
    else:
        solutions = []
        for k in k_range:
            print('K =', k)
            D = dot_matrix(X)
            Q = sdp_km(D, k, max_iters=10000, eps=1e-7)
            solutions.append(Q)

        with open('solutions.pickle', 'wb') as f:
            pickle.dump(solutions, f)

    sns.set_style('white')
    sns.set_color_codes()

    plt.figure(figsize=(6, 6))
    plot_data_clustered(X, gt)

    plt.savefig('{}{}.pdf'.format(dir_name, 'circle'))

    _, axes = plt.subplots(nrows=1, ncols=2, squeeze=True, figsize=(9, 7),
                           gridspec_kw={'width_ratios': (0.95, 0.05),
                                        'wspace': 0.04})
    n_diags = 10
    palette = sns.color_palette('hls', n_colors=n_diags + 1)
    for k, Q in zip(k_range, solutions):
        Q[Q < 1e-3 * Q.max()] = 0

        for i in range(n_diags):
            values = np.mean(np.unique(np.diag(Q, k=i)))
            try:
                abscissa = [k] * len(values)
            except TypeError:
                abscissa = k
            axes[0].scatter(abscissa, values, marker='o', s=5, c=palette[i],
                            edgecolors='none')

    axes[0].set_xlabel(r'$K$', fontdict=dict(size='x-large'))
    axes[0].set_ylabel(r'Value along $h$-diagonal', fontsize='x-large')
    axes[0].set_xticks(k_range[9::10] - 1)
    axes[0].set_xticklabels(k_range[9::10])
    axes[0].tick_params(axis='both', which='major', labelsize='x-large')

    cmap = mpl.colors.ListedColormap(palette)
    bounds = np.arange(n_diags+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    mpl.colorbar.ColorbarBase(axes[1], cmap=cmap,
                              norm=norm,
                              format=r'$h$=%d',
                              ticks=np.arange(n_diags)+0.5,
                              spacing='proportional',
                              orientation='vertical')
    axes[1].tick_params(axis='both', which='major', labelsize='x-large')

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
    plt.ylabel('Number of active eigenvalues', fontsize='x-large')
    plt.xlim(k_range.min() - 2, k_range.max() + 2)
    plt.ylim(k_range.min() - 2, k_range.max() + 2)
    plt.tick_params(axis='both', which='major', labelsize='x-large')

    plt.savefig('{}{}.pdf'.format(dir_name, 'circle_k-evolution_eigs'))

    all_eigvals = []
    F = dft(len(X), scale='sqrtn')
    for k, Q in zip(k_range, solutions):
        eigvals = np.diag(F.dot(Q).dot(np.conjugate(F).T))
        all_eigvals.append(eigvals)
    all_eigvals = np.vstack(all_eigvals)

    plt.figure(figsize=(6, 6))
    sns.stripplot(data=all_eigvals.T, size=2)
    plt.plot(k_range - 1, np.mean(all_eigvals, axis=1), color='#e41a1c',
             label='Mean')
    plt.plot(k_range - 1, np.median(all_eigvals, axis=1), color='k',
             label='Median')

    plt.xlabel(r'$K$', fontdict=dict(size='x-large'))
    plt.ylabel('Eigenvalues', fontsize='x-large')
    plt.legend(fontsize='x-large')
    plt.xticks(k_range[9::10] - 1, k_range[9::10])
    plt.tick_params(axis='both', which='major', labelsize='x-large')

    plt.savefig('{}{}.pdf'.format(dir_name, 'circle_k-evolution_eigs2'))

    # for k in [1, 12, 25, 100]:
    #     plt.figure(figsize=(6, 6))
    #     plot_matrix(solutions[k-1])
    #     plt.savefig('{}{}{}.pdf'.format(dir_name, 'circle_k', k))


def test_circle_sdp_lp():

    def circle_lp(X, k):
        D = dot_matrix(X)

        F = dft(len(X), scale='sqrtn')
        eigvals = np.diag(np.real(F.dot(D).dot(np.conjugate(F).T)))

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
        prob.solve(solver=cp.SCS, eps=1e-10, max_iters=100000)
        q = np.asarray(q.value)

        # from scipy.optimize import linprog
        #
        # A_eq = np.zeros((2, n))
        # A_eq[0, 0] = 1
        # A_eq[1, :] = 1
        # b_eq = np.array([1, k])
        # A_ub = np.vstack([-cos_vec(tau, n).T for tau in range(n)])
        # b_ub = np.zeros((n,))
        #
        # from scipy.io import savemat
        # savemat('sdp_lp.mat',
        #         dict(A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
        #              f=-eigvals))
        #
        # opt_res = linprog(-eigvals,
        #                   A_ub=A_ub, b_ub=b_ub,
        #                   A_eq=A_eq, b_eq=b_eq,
        #                   # bounds=(0, 1),
        #                   method='simplex',
        #                   # method='interior-point',
        #                   options={'disp': True, 'tol': 1e-10}
        #                   )
        #
        # print(opt_res.success, opt_res.status)
        # # print(opt_res.slack)
        # print(q[1])
        # print(opt_res.x)
        # print(prob.value)
        # print(-opt_res.fun)
        # print(np.allclose(q, opt_res.x))
        # print(np.allclose(np.sort(q), np.sort(opt_res.x)))
        # print('q0', opt_res.x[0])
        # print('SUM', np.sum(opt_res.x))
        # print('q min/max', np.min(opt_res.x), np.max(opt_res.x))
        # cos_cons = [opt_res.x.T.dot(cos_vec(tau, n)) for tau in range(n)]
        # print(np.min(cos_cons), np.max(cos_cons))
        # print(np.min(A_ub.dot(opt_res.x)), np.max(A_ub.dot(opt_res.x)))
        #
        # q = opt_res.x

        Q_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Q_mat[i, j] = q.T.dot(cos_vec(i - j, n))

        return Q_mat, np.sort(np.squeeze(q))[::-1]

    X, gt = toy.circles(n_samples=200)
    X = X[gt == 0, :]

    k = 16
    Q_lp, q = circle_lp(X, k)

    D = dot_matrix(X)
    Q_sdp = sdp_km(D, k)
    eigvals, _ = np.linalg.eigh(Q_sdp)
    eigvals = eigvals[::-1]

    sns.set_style('white')
    sns.set_color_codes()

    plt.figure()
    plot_matrix(Q_sdp)
    plt.title('SDP solution', fontsize='xx-large')
    plt.savefig('{}{}.pdf'.format(dir_name, 'circle_sdp'))

    plt.figure()
    plot_matrix(Q_lp)
    plt.title('LP solution', fontsize='xx-large')
    plt.savefig('{}{}.pdf'.format(dir_name, 'circle_lp'))

    plt.figure()
    plt.plot(q, linewidth=3, label='LP')
    plt.plot(eigvals, linewidth=3, linestyle=':', label='SDP')
    plt.xlim(0, len(eigvals))
    plt.xlabel('Eigenvalues', fontsize='xx-large')
    plt.legend()

    plt.savefig('{}{}.pdf'.format(dir_name, 'circle_sdp_lp'))

    print(np.linalg.norm(eigvals - q) / np.linalg.norm(eigvals))
    print(np.linalg.norm(Q_sdp - Q_lp, 'fro') / np.linalg.norm(Q_sdp, 'fro'))


if __name__ == '__main__':
    test_circles()
    test_one_circle(from_file=False)
    test_circle_sdp_lp()

    plt.show()
