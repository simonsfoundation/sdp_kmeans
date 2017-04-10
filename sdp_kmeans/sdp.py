from __future__ import print_function, division, absolute_import
import cvxpy as cp
from functools import partial
import numpy as np
from scipy.optimize import minimize
from sdp_kmeans.nmf import symnmf_admm
from sdp_kmeans.utils import dot_matrix


def sdp_kmeans_multilayer(X, layer_sizes, method='cvx'):
    if method == 'cvx':
        solver = sdp_kmeans_multilayer_cvx
    elif method == 'bm':
        solver = sdp_kmeans_multilayer_bm
    else:
        raise ValueError('The method should be one of "cvx" and "bm"')

    return solver(X, layer_sizes)


def sdp_kmeans_multilayer_cvx(X, layer_sizes):
    Ds = [dot_matrix(X)]
    for size in layer_sizes:
        D_sdp = sdp_km(Ds[-1], size)
        Ds.append(D_sdp)
    return Ds


def sdp_km(D, n_clusters):
    Z = cp.Semidef(D.shape[0])
    ones = np.ones((D.shape[0], 1))
    objective = cp.Maximize(cp.trace(D * Z))
    constraints = [Z >= 0,
                   Z * ones == ones,
                   cp.trace(Z) == n_clusters]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)

    return np.asarray(Z.value)


def sdp_kmeans_multilayer_bm(X, layer_sizes):
    Ys = [X]
    for size in layer_sizes:
        Y = sdp_km_burer_monteiro(Ys[-1], size)
        Ys.append(Y)
    Ds = [Y.dot(Y.T) for Y in Ys]
    return Ds


def sdp_km_burer_monteiro(X, n_clusters, rank=None, maxiter=1e3, tol=1e-5):
    if rank is None:
        rank = 8 * n_clusters

    X_norm = X - np.mean(X, axis=0)
    cov = X_norm.T.dot(X_norm)
    X_norm /= np.trace(cov.dot(cov)) ** 0.25

    Y_shape = (len(X), rank)
    ones = np.ones((len(X), 1))

    def lagrangian(x, lambda1, lambda2, sigma1, sigma2):
        Y = x.reshape(Y_shape)

        YtX = Y.T.dot(X_norm)
        obj = -np.trace(YtX.dot(YtX.T))

        trYtY_minus_nclusters = np.trace(Y.T.dot(Y)) - n_clusters
        obj -= lambda1 * trYtY_minus_nclusters
        obj += .5 * sigma1 * trYtY_minus_nclusters ** 2

        YYt1_minus_1 = Y.dot(Y.T.dot(ones)) - ones
        obj -= lambda2.T.dot(YYt1_minus_1)[0, 0]
        obj += .5 * sigma2 * np.sum(YYt1_minus_1 ** 2)

        return obj

    def grad(x, lambda1, lambda2, sigma1, sigma2):
        Y = x.reshape(Y_shape)

        delta = -2 * X_norm.dot(X_norm.T.dot(Y))

        YtY = Y.T.dot(Y)
        delta -= 2 * (lambda1
                      -sigma1 * (np.trace(Y.T.dot(Y)) - n_clusters)) * Y

        delta -= ones.dot(lambda2.T.dot(Y)) + lambda2.dot(ones.T.dot(Y))
        # YYt = Y.dot(Y.T)
        # eet = ones.dot(ones.T)
        Yt1 = Y.T.dot(ones)
        delta += sigma2 * (Y.dot(Yt1).dot(Yt1.T) + ones.dot(Yt1.T).dot(YtY)
                           -2 * ones.dot(Yt1.T))

        return delta.flatten()

    Y = symnmf_admm(X_norm.dot(X_norm.T), rank)

    lambda1 = 0.
    lambda2 = np.zeros((len(X), 1))
    sigma1 = 1
    sigma2 = 1
    step = 1

    error = []
    for n_iter in range(int(maxiter)):
        fun = partial(lagrangian, lambda1=lambda1, lambda2=lambda2,
                      sigma1=sigma1, sigma2=sigma2)
        jac = partial(grad, lambda1=lambda1, lambda2=lambda2,
                      sigma1=sigma1, sigma2=sigma2)
        bounds = [(0, 1)] * np.prod(Y_shape)

        Y_old = Y.copy()
        res = minimize(fun, Y.flatten(), jac=jac, bounds=bounds,
                       method='L-BFGS-B',)
        Y = res.x.reshape(Y_shape)

        lambda1 -= step * sigma1 * (np.trace(Y.T.dot(Y)) - n_clusters)
        lambda2 -= step * sigma2 * (Y.dot(Y.T.dot(ones)) - ones)

        error.append(np.linalg.norm(Y - Y_old) / np.linalg.norm(Y_old))

        # YtX = Y.T.dot(X_norm)
        # print(n_iter, np.trace(YtX.dot(YtX.T)), error[-1],
        #       'const', np.min(Y.dot(Y.T)), np.trace(Y.T.dot(Y)),
        #       np.min(Y.dot(Y.T.dot(ones))),
        #       np.max(Y.dot(Y.T.dot(ones))),
        #       'cols',
        #       np.min(ones.T.dot(Y).dot(Y.T)),
        #       np.max(ones.T.dot(Y).dot(Y.T))
        #       )

        if error[-1] < tol:
            break

    return Y
