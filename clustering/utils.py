import numpy as np


def dot_matrix(X):
    X_norm = X - np.mean(X, axis=0)
    X_norm /= np.max(np.linalg.norm(X, axis=1))
    return X_norm.dot(X_norm.T)


