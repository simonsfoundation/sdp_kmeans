import numpy as np
from numpy.linalg import eigh, svd
from clustering.sdp import sdp_kmeans_multilayer


def sdp_kmeans_embedding(X, n_clusters, target_dim, ret_sdp=False):
    Ds = sdp_kmeans_multilayer(X, [n_clusters])

    Ds = [Ds[0], Ds[-1]]
    Y = spectral_embedding(Ds[-1], target_dim=target_dim, discard_first=True)
    if ret_sdp:
        return Y, Ds
    else:
        return Y


def spectral_embedding(mat, target_dim, discard_first=True):
    if discard_first:
        last = -1
    else:
        last = None
    if mat.shape[0] != mat.shape[1]:
        mat = mat.dot(mat.T)
    eigvals, eigvecs = eigh(mat)
    eigvecs = eigvecs[:, -(target_dim + 1):last]
    eigvals_crop = eigvals[-(target_dim+1):last]
    Y = eigvecs.dot(np.diag(np.sqrt(eigvals_crop)))
    Y = Y[:, ::-1]

    variance_explaned(eigvals, eigvals_crop)
    return Y


def variance_explaned(eigvals, eigvals_crop):
    eigvals_crop[eigvals_crop < 0] = 0
    eigvals[eigvals < 0] = 0
    var = np.sum(eigvals_crop) / np.sum(eigvals)
    print('Variance explained:', var)
    # print('Eigenvalues:', eigvals / np.sum(eigvals))

