from __future__ import print_function, division, absolute_import
import inspect
import numpy as np
import os
from PIL import Image
from scipy.io import loadmat, savemat
import sklearn.datasets as sk_datasets
from subprocess import Popen
from urllib.request import urlretrieve
from zipfile import ZipFile


def digits():
    data = sk_datasets.load_digits(n_class=3)
    X = data.data
    gt = data.target

    keep = X.max(axis=0) != X.min(axis=0)
    X = X[:, keep]
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0, ddof=1)

    idx = np.argsort(gt)
    X = X[idx, :]
    gt = gt[idx]
    return X, gt


def teapot():
    dir_name = os.path.dirname(inspect.getfile(teapot))
    filename = '{}/{}'.format(dir_name, 'teapots100.mat')
    if not os.path.exists(filename):
        urlretrieve('http://www.cs.columbia.edu/~jebara/4772/teapots100.mat',
                    filename)
    data = loadmat(filename)
    X = data['teapots'].T
    return X


def mnist(digit='all', n_samples=0, return_gt=False):
    mnist = sk_datasets.fetch_mldata('MNIST original')
    X = mnist.data
    gt = mnist.target

    if digit == 'all':  # keep all digits
        pass
    else:
        X = X[gt == digit, :]
        gt = gt[gt == digit]

    if n_samples > len(X):
        raise ValueError('Requesting {} samples'
                         'from {} datapoints'.format(n_samples, len(X)))
    if n_samples > 0:
        np.random.seed(0)
        selection = np.random.randint(len(X), size=n_samples)
        X = X[selection, :]
        gt = gt[selection]

        idx = np.argsort(gt)
        X = X[idx, :]
        gt = gt[idx]

    if return_gt:
        return X, gt
    else:
        return X


def iris():
    X, gt = sk_datasets.load_iris(return_X_y=True)
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0, ddof=1)
    return X, gt


def turntable(objects=['Carton', 'Car2']):
    dir_name = os.path.dirname(inspect.getfile(turntable))
    dir_name += '/Turntable'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    urlname = 'http://www.vision.caltech.edu/pmoreels/Datasets/' \
              'TurntableObjects/{}/14.6mmZoom/{}/'

    locations = ['ImageSet070504', 'ImageSet071204', 'ImageSet071404',
                 'ImageSet071604', 'ImageSet072104', 'ImageSet072204',
                 'ImageSet072304', 'ImageSet072604', 'ImageSet072704',
                 'ImageSet072904', 'ImageSet080304', 'ImageSet080404',
                 'ImageSet091504']

    for obj in objects:
        filename = '{}/{}'.format(dir_name, obj)
        if not os.path.exists(filename):
            os.mkdir(filename)

            for loc in locations:
                sp = Popen(['wget', '-nH', '--cut-dirs=5', '--mirror',
                            '--no-parent', urlname.format(loc, obj)],
                           stdout=open(os.devnull, 'wb'),
                           stderr=open(os.devnull, 'wb'),
                           cwd=dir_name)
                sp.wait()

    mat_filename = '{}/{}.mat'.format(dir_name, '-'.join(objects))
    if not os.path.exists(mat_filename):
        data_dict = {}
        for obj in objects:
            filename = '{}/{}/Bottom/'.format(dir_name, obj)
            angles = ['{:03d}'.format(a) for a in range(0, 360, 5)]
            images = [fn for fn in os.listdir(filename) for a in angles
                      if fn.startswith('img_1-{}_'.format(a)) and
                      fn.endswith('_0.JPG')]
            print(images)
            data = []
            for fn in images:
                im = Image.open(filename + fn)
                im = im.resize((im.width // 4, im.height // 4),
                               resample=Image.BILINEAR)
                data.append(np.array(im).flatten())
            data_dict[obj] = np.array(data)

        savemat(mat_filename, data_dict)
    else:
        data_dict = loadmat(mat_filename)

    X = np.vstack([data_dict[obj] for obj in objects])

    gt = np.zeros((len(X),), dtype=np.int)
    sizes = np.cumsum([0] + [len(data_dict[obj]) for obj in objects])
    for i in range(len(objects)):
        gt[sizes[i]:sizes[i + 1]] = i

    return X, gt


def yale_faces(subjects=[1]):
    """
    See
    http://vision.ucsd.edu/~iskwak/ExtYaleDatabase/Yale%20Face%20Database.htm
    for a description of the file format.
    """
    dir_name = os.path.dirname(inspect.getfile(yale_faces))
    filename = '{}/{}'.format(dir_name, 'CroppedYale.zip')
    if not os.path.exists(filename):
        urlretrieve('http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/'
                    'CroppedYale.zip', filename)

    subdir_name = '{}/{}'.format(dir_name, 'CroppedYale')
    if not os.path.exists(subdir_name):
        zip_ref = ZipFile(filename, 'r')
        zip_ref.extractall(dir_name)
        zip_ref.close()

    data_dict = {}
    for s in subjects:
        valid_dir = 'yaleB{:02d}'.format(s)
        valid_file = valid_dir + '_P00A'

        images = [f for f in os.listdir('{}/{}'.format(subdir_name, valid_dir))
                  if f.startswith(valid_file) and f.endswith('.pgm')]

        data = []
        for fn in images:
            im = Image.open('{}/{}/{}'.format(subdir_name, valid_dir, fn))
            data.append(np.array(im).flatten())
        data_dict[s] = np.array(data)

    X = np.vstack([data_dict[s] for s in subjects])

    gt = np.zeros((len(X),), dtype=np.int)
    sizes = np.cumsum([0] + [len(data_dict[obj]) for obj in subjects])
    for i in range(len(subjects)):
        gt[sizes[i]:sizes[i + 1]] = i

    return X, gt


def _uci_clustering(name, url=None, converters=None):
    if url is None:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/' \
              '{0}/{0}.data'

    dir_name = os.path.dirname(inspect.getfile(wine))
    local_filename = '{}/{}.data'.format(dir_name, name)
    if not os.path.exists(local_filename):
        urlretrieve(url.format(name), local_filename)

    try:
        mat = np.loadtxt(local_filename, delimiter=',', converters=converters)
    except ValueError:
        mat = np.loadtxt(local_filename, converters=converters)

    return mat


def wine():
    mat = _uci_clustering('wine')
    gt = mat[:, 0] - 1
    X = mat[:, 1:]
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0, ddof=1)
    return X, gt


def glass():
    mat = _uci_clustering('glass')
    gt = mat[:, -1] - 1
    X = mat[:, 1:-1]
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0, ddof=1)
    return X, gt


def seeds():
    mat = _uci_clustering('seeds', url='https://archive.ics.uci.edu/ml/'
                                       'machine-learning-databases/00236/'
                                       'seeds_dataset.txt')
    gt = mat[:, -1] - 1
    X = mat[:, :-1]
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0, ddof=1)
    return X, gt


def libras():
    mat = _uci_clustering('libras', url='https://archive.ics.uci.edu/ml/'
                                        'machine-learning-databases/libras/'
                                        'movement_libras.data')
    gt = mat[:, -1] - 1
    X = mat[:, :-1]
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0, ddof=1)
    return X, gt


def breast():
    def default_converter(s):
        try:
            return float(s)
        except ValueError:
            return -1

    mat = _uci_clustering('breast', url='https://archive.ics.uci.edu/ml/'
                                        'machine-learning-databases/'
                                        'breast-cancer-wisconsin/'
                                        'breast-cancer-wisconsin.data',
                          converters=dict([(i, default_converter)
                                          for i in range(11)]))

    gt = mat[:, -1]
    gt[gt == 2] = 0
    gt[gt == 4] = 1
    X = mat[:, 1:-1]
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0, ddof=1)

    idx = np.argsort(gt)
    X = X[idx, :]
    gt = gt[idx]
    return X, gt


if __name__ == '__main__':
    X, gt = breast()
    print(X.shape, gt.shape, gt.min(), gt.max())
