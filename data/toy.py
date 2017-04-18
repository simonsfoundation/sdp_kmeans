from __future__ import print_function, division, absolute_import
import numpy as np
import sklearn.datasets as sk_datasets
from sklearn.utils import check_random_state


def gaussian_blobs(n_samples=200, return_centers=False):
    random_state = 0
    centers = [(-10, -10), (-10, 0), (0, -10)]
    centers.extend([(10, 10), (10, 0), (0, 10)])
    centers = np.array(centers)
    X, gt = sk_datasets.make_blobs(n_samples=n_samples, centers=centers,
                                   n_features=2, shuffle=False,
                                   random_state=random_state)
    if return_centers:
        return X, gt, centers
    else:
        return X, gt


def circles(n_samples=200, factor=0.5, noise=None, regular=True,
            random_state=0):
    def make_circles(n_samples=100, noise=None,
                     random_state=None, factor=.8):
        if regular:
            if factor > 1 or factor < 0:
                raise ValueError("'factor' has to be between 0 and 1.")

            generator = check_random_state(random_state)
            # so as not to have the first point = last point, we add
            # one and then remove it.
            linspace = np.linspace(0, 2 * np.pi, n_samples // 2 + 1)[:-1]
            outer_circ_x = np.cos(linspace)
            outer_circ_y = np.sin(linspace)
            inner_circ_x = outer_circ_x * factor
            inner_circ_y = outer_circ_y * factor

            X = np.vstack((np.hstack((outer_circ_x, inner_circ_x)),
                           np.hstack((outer_circ_y, inner_circ_y)))).T
            y = np.hstack([np.zeros(n_samples // 2, dtype=np.intp),
                           np.ones(n_samples // 2, dtype=np.intp)])

            if noise is not None:
                X += generator.normal(scale=noise, size=X.shape)

            return X, y
        else:
            return sk_datasets.make_circles(n_samples=n_samples,
                                            shuffle=False, noise=noise,
                                            random_state=random_state,
                                            factor=factor)

    X, gt = make_circles(n_samples=n_samples, factor=factor, noise=noise,
                         random_state=random_state)
    return X, gt


def moons():
    random_state = 0
    X, gt = sk_datasets.make_moons(n_samples=200, noise=.05,
                                   shuffle=False, random_state=random_state)
    return X, gt


def swiss_roll_2d(n_samples=100, noise=0.0, regular=True, random_state=None):
    if regular:
        generator = check_random_state(random_state)
        t = 1.5 * np.pi * (1 + 2 * np.linspace(0, 1, n_samples))
        t = t[np.newaxis, :]
        x = t * np.cos(t)
        y = t * np.sin(t)

        X = np.concatenate((x, y))
        X += noise * generator.randn(2, n_samples)
        X = X.T
        t = np.squeeze(t)

        return X, t
    else:
        X, t = sk_datasets.make_swiss_roll(n_samples=n_samples, noise=noise,
                                           random_state=random_state)
        X = X[:, [0, 2]]
        idx = np.argsort(t)
        return X[idx, :], t[idx]


def double_swiss_roll(n_samples=200, regular=True):
    random_state = 0
    X1, _ = swiss_roll_2d(n_samples=n_samples // 2, noise=.0, regular=regular,
                          random_state=random_state)

    theta = np.pi
    X2 = X1.dot(np.array([[np.cos(theta), np.sin(theta)],
                          [-np.sin(theta), np.cos(theta)]]))
    X = np.vstack((X1, X2))
    gt = np.zeros((len(X),))
    gt[len(X1):] = 1
    return X, gt


def swiss_roll_3d(n_samples=800):
    random_state = 0
    X, t = sk_datasets.make_swiss_roll(n_samples=n_samples,
                                       random_state=random_state)
    idx = np.argsort(t)
    X = X[idx, :]
    # X = np.roll(X, 1, axis=1)
    # X = X[:, :2]
    return X


def trefoil_knot(n_samples=400):
    t = np.linspace(0, 2 * np.pi, n_samples + 1)[:-1]
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    return np.hstack((x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]))


def thin_lines(n_samples=200):
    centers = [(-5, -5), (-5, 0), (0, -5)]
    centers.extend([(5, 5), (5, 0), (0, 5)])
    centers = np.array(centers)
    n_centers = len(centers)

    n_samples_per_center = [int(n_samples // n_centers)] * n_centers
    for i in range(n_samples % n_centers):
        n_samples_per_center[i] += 1

    angles = np.linspace(0, np.pi, n_centers)
    points = []
    gt = []
    for i, (n, phi, c) in enumerate(zip(n_samples_per_center, angles, centers)):
        tx = np.linspace(0, 3, n)
        # ty = np.zeros_like(tx)
        ty = 0.1 * np.random.randn(n)
        X = np.vstack((tx, ty)).T
        rotation = [[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]
        gt.append(i * np.ones((n,)))
        points.append(X.dot(rotation) + c)

    points = np.vstack(points)
    gt = np.hstack(gt)
    return points, gt
