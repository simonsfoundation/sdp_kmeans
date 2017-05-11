import matplotlib.pyplot as plt
import numpy as np
import os
from data import toy

dir_name = '../results/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
dir_name += 'additional/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def distances2gramian():
    x, gt = toy.circles(n_samples=240)
    x = x[gt == 0, :]
    center = 60

    plt.figure(figsize=(13, 4), tight_layout=True)

    ax = plt.subplot(131)
    ax.set_title('Input dataset', fontsize='x-large')

    plt.scatter(x[::10, 0], x[::10, 1], c='k')
    for i, v in enumerate(x[::10, :]):
        if i == 6:
            color = '#e41a1c'
        else:
            color = '#377eb8'
        plt.arrow(0, 0, v[0], v[1], length_includes_head=True, head_width=0.07,
                  color=color)
        ax.annotate('{}'.format(i), xy=v, xytext=1.11 * v,
                    horizontalalignment='center', verticalalignment='center')

    ax.set_aspect('equal')
    ax.set_xlim(x[:, 0].min() - 0.2, x[:, 0].max() + 0.2)
    ax.set_ylim(x[:, 1].min() - 0.2, x[:, 1].max() + 0.2)

    ax = plt.subplot(132)
    ax.set_title('Euclidean distances to $\mathbf{{x}}_6$',
                 fontsize='x-large')
    dists = np.sqrt(np.sum((x - x[center]) ** 2, axis=1))

    plt.fill_between(range(len(x)), np.sqrt(2) * np.ones_like(dists), dists,
                     where=dists >= np.sqrt(2), facecolor='#fbb4ae',
                     alpha=0.5)
    plt.fill_between(range(len(x)), np.zeros_like(dists), dists,
                     where=dists < np.sqrt(2), facecolor='#ccebc5',
                     alpha=0.5)
    plt.plot([0, len(x)], [0, 0], 'k')
    plt.plot([0, len(x)], [np.sqrt(2)] * 2, 'k:')

    plt.stem(np.arange(0, len(dists), 10), dists[::10], basefmt='none')
    plot_info = plt.stem([center], [dists[center]], basefmt='none')
    plt.setp(plot_info.stemlines, 'color', '#e41a1c')
    plt.setp(plot_info.markerline, 'color', '#e41a1c', 'markerfacecolor', '#e41a1c')

    for i, d in enumerate(dists[::10]):
        ax.annotate('{}'.format(i), xy=(0, 0), xytext=(i * 10, d + 0.07),
                    horizontalalignment='center', verticalalignment='center')

    plt.xticks(np.linspace(0, len(dists), 3, endpoint=True),
               ['0', '$\pi$', '$2\pi$'])
    ax.set_ylim(ax.get_ylim()[0], dists.max() + 0.2)

    ax = plt.subplot(133)
    ax.set_title('Dot products with $\mathbf{{x}}_6$',
                 fontsize='x-large')
    xtx_center = x.dot(x[center, :])

    plt.fill_between(range(len(x)), np.zeros_like(xtx_center), xtx_center,
                     where=xtx_center >= 0, facecolor='#ccebc5',
                     alpha=0.5)
    plt.fill_between(range(len(x)), np.zeros_like(xtx_center), xtx_center,
                     where=xtx_center < 0, facecolor='#fbb4ae',
                     alpha=0.5)

    plt.plot([0, len(x)], [0, 0], 'k')

    plt.stem(np.arange(0, len(xtx_center), 10), xtx_center[::10], basefmt='none')
    plot_info = plt.stem([center], [xtx_center[center]], basefmt='none')
    plt.setp(plot_info.stemlines, 'color', '#e41a1c')
    plt.setp(plot_info.markerline, 'color', '#e41a1c', 'markerfacecolor', '#e41a1c')

    plt.xticks(np.arange(0, len(xtx_center), 10), np.arange(len(dists[::10])))

    for i, d in enumerate(xtx_center[::10]):
        if d >= 0:
            d += 0.07
        else:
            d -= 0.09
        ax.annotate('{}'.format(i), xy=(0, 0), xytext=(i * 10, d),
                    horizontalalignment='center', verticalalignment='center')

    plt.xticks(np.linspace(0, len(xtx_center), 3, endpoint=True),
               ['0', '$\pi$', '$2\pi$'])
    ax.set_ylim(xtx_center.min() - 0.2, xtx_center.max() + 0.2)

    plt.savefig(dir_name + 'distances2gramian.pdf')


if __name__ == '__main__':
    distances2gramian()
    plt.show()
