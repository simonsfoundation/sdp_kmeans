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
    x, gt = toy.circles(n_samples=200)
    x = x[gt == 0, :]
    center = 50

    plt.figure(figsize=(13, 4), tight_layout=True)

    ax = plt.subplot(131)
    ax.set_title('Input dataset', fontsize='x-large')

    plt.scatter(x[:, 0], x[:, 1], c='#377eb8')
    plt.scatter(x[center, 0], x[center, 1], c='#e41a1c')
    ax.annotate('$\mathbf{{x}}_0$', xy=x[center, :], xytext=0.85 * x[center, :],
                horizontalalignment='center', verticalalignment='center',
                fontsize='x-large')

    ax.set_aspect('equal')
    ax.set_xlim(x[:, 0].min() - 0.2, x[:, 0].max() + 0.2)
    ax.set_ylim(x[:, 1].min() - 0.2, x[:, 1].max() + 0.2)

    ax = plt.subplot(132)
    ax.set_title('Squared Euclidean distances to $\mathbf{{x}}_0$',
                 fontsize='x-large')
    dists = np.sum((x - x[center]) ** 2, axis=1)

    # plt.fill_between(range(len(x)), 2 * np.ones_like(dists), dists,
    #                  where=dists <= 2, facecolor='#ccebc5',
    #                  alpha=0.5)
    plt.plot(dists, linewidth=2)
    plt.plot([0, len(x)], [0, 0], 'k')
    plt.plot([0, len(x)], [2] * 2, ':', color='gray')
    plt.plot([0.25 * len(x)] * 2, [-0.2, 2], ':', color='gray')
    plt.plot([0.75 * len(x)] * 2, [-0.2, 2], ':', color='gray')
    plt.scatter(center, 0, c='#e41a1c', zorder=100)

    plt.xticks(np.linspace(0, len(x), 5, endpoint=True),
               ['0', r'$\frac{3}{4}\pi$', '$\pi$', r'$\frac{3}{2}\pi$',
                '$2\pi$'])
    ax.set_ylim(-0.2, dists.max() + 0.2)

    ax = plt.subplot(133)
    ax.set_title('Dot products with $\mathbf{{x}}_0$',
                 fontsize='x-large')
    xtx_center = x.dot(x[center, :])

    plt.plot([0, len(x)], [0, 0], 'k')
    plt.plot([0.25 * len(x)] * 2, [xtx_center.min() - 0.2, 0], ':', color='gray')
    plt.plot([0.75 * len(x)] * 2, [xtx_center.min() - 0.2, 0], ':', color='gray')
    plt.scatter(center, xtx_center[center], c='#e41a1c', zorder=100)

    # plt.fill_between(range(len(x)), np.zeros_like(xtx_center), xtx_center,
    #                  where=xtx_center >= 0, facecolor='#ccebc5',
    #                  alpha=0.5)
    plt.plot(xtx_center, linewidth=2)
    plt.plot(np.maximum(xtx_center, 0), '--', linewidth=4,
             label='NOMAD kernel function')

    plt.xticks(np.arange(0, len(xtx_center), 10), np.arange(len(dists[::10])))
    plt.xticks(np.linspace(0, len(x), 5, endpoint=True),
               ['0', r'$\frac{3}{4}\pi$', '$\pi$', r'$\frac{3}{2}\pi$',
                '$2\pi$'])
    ax.set_ylim(xtx_center.min() - 0.2, xtx_center.max() + 0.2)
    plt.legend()

    plt.savefig(dir_name + 'distances2gramian.pdf')


if __name__ == '__main__':
    distances2gramian()
    plt.show()
