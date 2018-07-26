import matplotlib.colors as mpl_colors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import warnings


def plot_confusion_matrix(conf_mat):
    cm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

    plt.matshow(cm, cmap='gray_r', vmin=0, vmax=1)

    # text portion
    ind_array = np.arange(cm.shape[0])
    x, y = np.meshgrid(ind_array, ind_array)

    for i, j in zip(x.flatten(), y.flatten()):
        c = 'k' if cm[i, j] <= 0.5 else 'w'
        plt.text(j, i, '{}'.format(conf_mat[i, j]), color=c, va='center', ha='center')

    plt.xticks([])
    plt.yticks([])


def plot_matrix(mat, cmap='gray_r', ax=None):
    if ax is None:
        ax = plt.gca()

    values = np.unique(mat)

    vmin = values.min()
    vmax = values.max()
    if ((vmax - vmin) / vmax) > 1e-3:
        values_mean = np.mean(values)
        values_std = np.std(values)
        vmin = np.maximum(vmin, values_mean - 10 * values_std)
        vmax = np.minimum(vmax, values_mean + 10 * values_std)

        ax.imshow(mat, interpolation='none', cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        mat = ((vmin + vmax) / 2) * np.ones_like(mat)
        ax.imshow(mat, interpolation='none', cmap=cmap)

    ax.imshow(mat, interpolation='none', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.grid(False)
    ax.tick_params(axis='both',
                   which='both',
                   bottom=False, top=False,
                   left=False, right=False,
                   labelbottom=False, labelleft=False)


def line_plot_clustered(X, gt, ax=None):
    if ax is None:
        ax = plt.gca()

    X = X.copy()
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0, ddof=1)

    keys = np.unique(gt)
    order = np.argsort([sum(gt == k) for k in keys])[::-1]
    keys = keys[order]

    for k, c in zip(keys, sns.color_palette('Set1', n_colors=len(keys))):
        mask = gt == k
        ax.plot(X[mask, :].T, color=c)
    ax.set_xlim(0, X.shape[1] - 1)
    if X.shape[1] < 10:
        xticks = range(X.shape[1])
    else:
        xticks = range(0, X.shape[1], 5)
    ax.set_xticks(xticks)
    ax.tick_params(axis='y',
                   which='both',
                   left=False, right=False,
                   labelleft=False)


def plot_data_clustered(X, gt, marker='o', ax=None):
    if ax is None:
        ax = plt.gca()

    keys = np.unique(gt)
    for k, c in zip(keys, sns.color_palette('Set1', n_colors=len(keys))):
        mask = gt == k
        ax.scatter(X[mask, 0], X[mask, 1], c=c, edgecolors=c,
                   marker=marker)

    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    X_max = X.max(axis=0)
    X_min = X.min(axis=0)
    range = (1.1 / 2.) * (X_max - X_min).max()
    center = (X_max + X_min) / 2.
    ax.set_xlim(xmin=center[0] - range, xmax=center[0] + range)
    ax.set_ylim(ymin=center[1] - range, ymax=center[1] + range)


formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, line=None:\
    formatwarning_orig(message, category, filename, lineno, line='')


def plot_data_embedded(X, palette='hls', marker='o', ax=None, elev_azim=None,
                       alpha=1):
    if X.shape[1] != 2 and X.shape[1] != 3:
        msg = 'Plotting first two dimensions out of {}.'.format(X.shape[1])
        warnings.warn(msg, category=RuntimeWarning)

        X = X[:, :2]

    _plot_data_embedded(X, palette=palette, marker=marker, ax=ax,
                        elev_azim=elev_azim, alpha=alpha)


def _plot_data_embedded(X, palette='hls', marker='o', ax=None, elev_azim=None,
                       alpha=1):
    if ax is None:
        ax = plt.gca()

    if palette == 'w':
        colors = [(1, 1, 1)] * len(X)
    elif palette == 'k':
        colors = [(0, 0, 0)] * len(X)
    elif palette == 'none':
        c = mpl_colors.to_rgb('#377eb8')
        colors = [c] * len(X)
    elif isinstance(palette, str) and palette[0] == '#':
        c = mpl_colors.to_rgb(palette)
        colors = [c] * len(X)
    else:
        colors = sns.color_palette(palette, n_colors=len(X))

    try:
        colors = [c + (a,) for c, a in zip(colors, alpha)]
        alpha = None
        edgecolors = 'k'
    except TypeError:
        edgecolors = colors

    X_max = X.max(axis=0)
    X_min = X.min(axis=0)
    range = (1.1 / 2.) * (X_max - X_min).max()
    center = (X_max + X_min) / 2.

    if X.shape[1] == 2:
        ax.scatter(X[:, 0], X[:, 1], c=colors, edgecolors=edgecolors,
                   marker=marker, alpha=alpha)

        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xlim(xmin=center[0] - range, xmax=center[0] + range)
        ax.set_ylim(ymin=center[1] - range, ymax=center[1] + range)

    elif X.shape[1] == 3:
        if elev_azim is not None:
            ax.view_init(elev=elev_azim[0], azim=elev_azim[1])

        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, edgecolors=colors,
                   marker=marker, alpha=alpha)

        ax.set_xlim(center[0] - range, center[0] + range)
        ax.set_ylim(center[1] - range, center[1] + range)
        ax.set_zlim(center[2] - range, center[2] + range)
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


def plot_images_embedded(embedding, img_getter, labels=None, palette='hls',
                         marker='o', ax=None, subsampling=10, zoom=.5):
    if ax is None:
        ax = plt.gca()

    plot_data_embedded(embedding, palette=palette, marker=marker, ax=ax)

    if labels is not None:
        frame_palette = sns.color_palette('Set1',
                                          n_colors=len(np.unique(labels)))

    for k in range(0, len(embedding), subsampling):
        pos = embedding[k, :]
        im = OffsetImage(img_getter(k), zoom=zoom)
        im.image.axes = ax

        if labels is not None:
            frameon = True
            bboxprops = dict(edgecolor=frame_palette[labels[k]], linewidth=3)
        else:
            frameon = False
            bboxprops = None

        ab = AnnotationBbox(im, pos,
                            xybox=(0., 0.),
                            xycoords='data',
                            boxcoords='offset points',
                            frameon=frameon, pad=0,
                            bboxprops=bboxprops)
        ax.add_artist(ab)


class Logger(object):
    def __init__(self, filename="Console.log"):
        self.stdout = sys.stdout
        self.log = open(filename, "w")

    def __del__(self):
        self.log.close()

    def close(self):
        self.log.close()

    def write(self, message):
        self.stdout.write(message)
        self.log.write(message)
        self.log.flush()

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)
