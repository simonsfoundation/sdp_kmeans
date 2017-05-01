import numpy as np
from scipy.sparse.csgraph import connected_components as inner_conn_comp


def dot_matrix(X):
    X_norm = X - np.mean(X, axis=0)
    X_norm /= np.max(np.linalg.norm(X, axis=1))
    return X_norm.dot(X_norm.T)


def connected_components(sym_mat, thresh=1e-4):
    binary_mat = sym_mat > sym_mat.max() * thresh
    n_comp, labels = inner_conn_comp(binary_mat, directed=False,
                                     return_labels=True)
    Q_labels = 0
    clusters = []
    for i in range(n_comp):
        mask = labels == i
        Q_labels += np.outer(mask, mask)
        clusters.append(mask)

    return Q_labels, clusters
