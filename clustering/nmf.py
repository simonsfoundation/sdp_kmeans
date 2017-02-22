from __future__ import print_function, division, absolute_import
# import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
# from sklearn.decomposition.nmf import non_negative_factorization
# from scipy.sparse.linalg import eigsh
from clustering.utils import dot_matrix


# def cluster(X, n_clusters, build_gramian=True):
#     print(X.shape, n_clusters)
#     if build_gramian:
#         X, D = dot_matrix(X)
#     else:
#         D = X.copy()
#     D[D < 0] = 0
#
#     labels = symnmf_anls(D, n_clusters)
#     for i in range(n_clusters):
#         m = labels[:, i].max()
#         if m > 0:
#             labels[:, i] /= m
#
#     norms = np.linalg.norm(labels, axis=0)
#     print(norms)
#     norms[norms == 0] = 1
#     labels_norm = labels / norms
#     D_approx = labels_norm.dot(labels_norm.T)
#
#     return X, D, labels, D_approx


def symnmf_anls(A, k, H=None, maxiter=1e3, tol=1e-3, alpha=0,
                debug=0):
    #SYMNMF_ANLS ANLS algorithm for SymNMF
    #   [H, iter, obj] = symnmf_anls(A, k, params) optimizes
    #   the following formulation:
    #
    #   min_H f(H) = ||A - WH'||_F^2 + alpha * ||W-H||_F^2
    #   subject to W >= 0, H >= 0
    #
    #   where A is a n*n symmetric matrix,
    #         H is a n*k nonnegative matrix.
    #         (typically, k << n)
    #   'symnmf_anls' returns:
    #       H: The low-rank n*k matrix used to approximate A.
    #       iter: Number of iterations before termination.
    #       obj: Objective value f(H) at the final solution H.
    #            (is set to -1 when it is not actually computed)
    #
    #   The optional 'params' has the following fields:
    #       Hinit: Initialization of H. To avoid H=0 returned as
    #              solution, the default random 'Hinit' is
    #                  2 * full(sqrt(mean(mean(A)) / k)) * rand(n, k)
    #              to make sure that entries of 'Hinit' fall
    #              into the interval [0, 2*sqrt(m/k)],
    #              where 'm' is the average of all entries of A.
    #              User-defined 'Hinit' should follow this rule.
    #       maxiter: Maximum number of iteration allowed.
    #                (default is 10000)
    #       tol: The tolerance parameter 'mu' in the cited paper,
    #            to determine convergence and terminate the algorithm.
    #            (default is 1e-3)
    #       alpha: The parameter for penalty term in the above
    #              formulation. A negative 'alpha' means using
    #                  alpha = max(max(A))^2;
    #              in the algorithm. When alpha=0, this code is generally
    #              faster and will adjust the final W, H to be the same
    #              matrix; however, there is no theoretical guarantee
    #              to converge with alpha=0 so far.
    #              (default is -1)
    #       computeobj: A boolean variable indicating whether the
    #                   objective value f(H) at the final solution H
    #                   will be computed.
    #                   (default is true)
    #       debug: There are 3 levels of debug information output.
    #              debug=0: No output (default)
    #              debug=1: Output the initial and final norms of projected gradient
    #              debug=2: Output the norms of projected gradient
    #                       in each iteration
    #
    #   In the context of graph clustering, 'A' is a symmetric matrix containing
    #   similarity values between every pair of data points in a data set of size 'n'.
    #   The output 'H' is a clustering indicator matrix, and clustering assignments
    #   are indicated by the largest entry in each row of 'H'.
    #
    A_pos = A.copy()
    A_pos[A < 0] = 0

    n = A_pos.shape[0]
    if n != A_pos.shape[1]:
        raise ValueError('A must be a symmetric matrix!')

    if H is None:
        H = 2 * np.sqrt(np.abs(A_pos).mean() / k) * np.random.rand(n, k)
    if alpha < 0:
        alpha = A_pos.max() ** 2

    W = H.copy()
    I_k = alpha * np.eye(k)

    left = H.T.dot(H)
    right = A_pos.dot(H)
    for i in range(int(maxiter)):
        W = _nnls(left + I_k, (right + alpha * H).T).T
        left = W.T.dot(W)
        right = A_pos.dot(W)
        H = _nnls(left + I_k, (right + alpha * W).T).T
        temp = alpha * (H - W)
        gradH = H.dot(left) - right + temp
        left = H.T.dot(H)
        right = A_pos.dot(H)
        gradW = W.dot(left) - right - temp

        projnorm = np.sqrt(_grad_norm2(gradW, W) + _grad_norm2(gradH, H))
        if i == 0:
            initgrad = projnorm
            if debug:
                print('init grad norm', initgrad)

        # print(i, projnorm, tol * initgrad)

        if projnorm < tol * initgrad:
            if debug:
                print('final grad norm', projnorm)
            break
        else:
            if debug > 1:
                print('iter {}: grad norm {}'.format(i, projnorm))

    if alpha == 0:
        norms_W = np.linalg.norm(W)
        norms_H = np.linalg.norm(H)
        norms = np.sqrt(norms_W * norms_H)
        W *= norms / norms_W
        H *= norms / norms_H

    return H


def _grad_norm2(grad_mat, mat):
    mask = np.logical_or(grad_mat <= 0, mat > 0)
    return (grad_mat[mask] ** 2).sum()


def symnmf_admm(A, k, H=None, maxiter=1e3, tol=1e-5, sigma=1):
    A = A.copy()
    A[A < 0] = 0

    n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError('A must be a symmetric matrix!')

    if H is None:
        H = np.sqrt(A.mean() / k) * np.random.randn(n, k)
        np.abs(H, H)

    Gamma = np.zeros((n, k))
    step = 1

    norm_A = np.linalg.norm(A)

    error = []
    for i in range(int(maxiter)):
        temp = np.linalg.inv(H.T.dot(H) + sigma * np.identity(k))
        W = (A.dot(H) + sigma * H - Gamma).dot(temp)
        W = np.maximum(W, 0)
        temp = np.linalg.inv(W.T.dot(W) + sigma * np.identity(k))
        H = (A.T.dot(W) + sigma * W + Gamma).dot(temp)
        H = np.maximum(H, 0)
        Gamma += step * sigma * (W - H)

        # print(i, np.linalg.norm(A - W.dot(H.T)), np.linalg.norm(W - H) / np.linalg.norm(W))
        # print(i, np.linalg.norm(W), np.linalg.norm(H))
        # error.append(np.linalg.norm(A - W.dot(H)) / norm_A)
        error.append(np.linalg.norm(W - H) / np.linalg.norm(W))
        if i > 0 and np.abs(error[-1]) < tol:
            break

        W_old = W.copy()
        H_old = H.copy()

    return W


def _nnls(A, B):
    X = []
    for i in range(B.shape[1]):
        sol = scipy.optimize.nnls(A, B[:, i])
        X.append(sol[0])
    return np.array(X).T
