import os
import tarfile

import h5py
import numpy
import six
from six.moves import range, cPickle

from fuel.converters.base import fill_hdf5_file, check_exists

DISTRIBUTION_FILE = 'cifar-10-python.tar.gz'
SMALL = 0.001


@check_exists(required_files=[DISTRIBUTION_FILE])
def convert_cifar10(directory, output_directory,
                    output_filename='cifar10.hdf5'):
    """Converts the CIFAR-10 dataset to HDF5.

    Converts the CIFAR-10 dataset to an HDF5 dataset compatible with
    :class:`fuel.datasets.CIFAR10`. The converted dataset is saved as
    'cifar10.hdf5'.

    It assumes the existence of the following file:

    * `cifar-10-python.tar.gz`

    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_directory : str
        Directory in which to save the converted dataset.
    output_filename : str, optional
        Name of the saved dataset. Defaults to 'cifar10.hdf5'.

    Returns
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted dataset.

    """
    output_path = os.path.join(output_directory, output_filename)
    h5file = h5py.File(output_path, mode='w')
    input_file = os.path.join(directory, DISTRIBUTION_FILE)
    tar_file = tarfile.open(input_file, 'r:gz')

    train_batches = []
    for batch in range(1, 6):
        file = tar_file.extractfile(
            'cifar-10-batches-py/data_batch_%d' % batch)
        try:
            if six.PY3:
                array = cPickle.load(file, encoding='latin1')
            else:
                array = cPickle.load(file)
            train_batches.append(array)
        finally:
            file.close()

    train_features = numpy.concatenate(
        [batch['data'].reshape(batch['data'].shape[0], 3, 32, 32)
            for batch in train_batches])
    train_labels = numpy.concatenate(
        [numpy.array(batch['labels'], dtype=numpy.uint8)
            for batch in train_batches])
    train_labels = numpy.expand_dims(train_labels, 1)

    file = tar_file.extractfile('cifar-10-batches-py/test_batch')
    try:
        if six.PY3:
            test = cPickle.load(file, encoding='latin1')
        else:
            test = cPickle.load(file)
    finally:
        file.close()

    test_features = test['data'].reshape(test['data'].shape[0],
                                         3, 32, 32)
    test_labels = numpy.array(test['labels'], dtype=numpy.uint8)
    test_labels = numpy.expand_dims(test_labels, 1)

    data = (('train', 'features', train_features),
            ('train', 'targets', train_labels),
            ('test', 'features', test_features),
            ('test', 'targets', test_labels))
    fill_hdf5_file(h5file, data)
    h5file['features'].dims[0].label = 'batch'
    h5file['features'].dims[1].label = 'channel'
    h5file['features'].dims[2].label = 'height'
    h5file['features'].dims[3].label = 'width'
    h5file['targets'].dims[0].label = 'batch'
    h5file['targets'].dims[1].label = 'index'

    h5file.flush()
    h5file.close()

    return (output_path,)


def fill_subparser(subparser):
    """Sets up a subparser to convert the CIFAR10 dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `cifar10` command.

    """
    subparser.set_defaults(func=convert_cifar10)


@check_exists(required_files=[DISTRIBUTION_FILE])
def convert_cifar10_pca(directory, variance, output_file, patchsize=None):
    """Converts the CIFAR-10 dataset to HDF5.

    Converts the CIFAR-10 dataset to an HDF5 dataset compatible with
    :class:`fuel.datasets.CIFAR10`. The converted dataset is saved as
    'cifar10.hdf5'.

    It assumes the existence of the following file:

    * `cifar-10-python.tar.gz`

    Parameters
    ----------
    variance: float
        Variance kept with PCA.
    directory : str
        Directory in which input files reside.
    output_file : str
        Where to save the converted dataset.

    """

    rng = numpy.random.RandomState(1)

    h5file = h5py.File(output_file, mode='w')
    input_file = os.path.join(directory, DISTRIBUTION_FILE)
    tar_file = tarfile.open(input_file, 'r:gz')

    train_batches = []
    for batch in range(1, 6):
        file = tar_file.extractfile(
            'cifar-10-batches-py/data_batch_%d' % batch)
        try:
            if six.PY3:
                array = cPickle.load(file, encoding='latin1')
            else:
                array = cPickle.load(file)
            train_batches.append(array)
        finally:
            file.close()

    train_features = numpy.concatenate(
        [batch['data']/255. for batch in train_batches])
    train_labels = numpy.concatenate(
        [numpy.array(batch['labels'], dtype=numpy.uint8)
            for batch in train_batches])
    train_labels = numpy.expand_dims(train_labels, 1)

    file = tar_file.extractfile('cifar-10-batches-py/test_batch')
    try:
        if six.PY3:
            test = cPickle.load(file, encoding='latin1')
        else:
            test = cPickle.load(file)
    finally:
        file.close()

    test_features = test['data']/255.
    test_labels = numpy.array(test['labels'], dtype=numpy.uint8)
    test_labels = numpy.expand_dims(test_labels, 1)

    if patchsize is not None:
        train_features = numpy.concatenate(
                [crop_patches_color(im.reshape(3, 32, 32).transpose(1, 2, 0),
                    numpy.array([rng.randint(patchsize/2,
                                             32-patchsize/2, 40),
                    rng.randint(patchsize/2, 32-patchsize/2, 40)]).T,
                    patchsize) for im in train_features[:1000]])

        R = rng.permutation(train_features.shape[0])
        train_features = train_features[R, :]

        print train_labels.shape
        train_labels = numpy.concatenate(
                sum(([train_labels[i]]*40 for i in 
                    range(1000)), []))
        train_labels = train_labels[R]
        print train_labels.shape
    else:
        print train_features.shape
        train_features = train_features.reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1)).reshape(
                (train_features.shape[0], numpy.prod(train_features.shape[1:])))

        test_features = test_features.reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1)).reshape(
                (test_features.shape[0], numpy.prod(test_features.shape[1:])))

    print train_features.shape

    meanstd = train_features.std()
    train_features -= train_features.mean(1)[:,None]
    train_features /= train_features.std(1)[:,None] + 0.1 * meanstd
    train_features_mean_0 = train_features.mean(0)[None,:]
    train_features -= train_features_mean_0
    train_features_std_0 = train_features.std(0)[None,:] + 0.1 * meanstd
    train_features /= train_features_std_0

    if patchsize is None:
        test_features -= test_features.mean(1)[:,None]
        test_features /= test_features.std(1)[:,None] + 0.1 * meanstd
        test_features -= train_features_mean_0
        test_features /= train_features_std_0

    pca_backward, pca_forward, zca_backward, zca_forward = pca(train_features,
            variance, whiten=False)
    train_features_whitened = numpy.dot(train_features, pca_backward.T).astype("float32")
    if patchsize is None:
        test_features_whitened = numpy.dot(test_features, pca_backward.T).astype("float32")

    train_features = train_features_whitened
    if patchsize is None:
        test_features = test_features_whitened

    data = (('train', 'features', train_features),
            ('train', 'targets', train_labels))

    if patchsize is None:
        data += (('test', 'features', test_features),
                 ('test', 'targets', test_labels))

    fill_hdf5_file(h5file, data)
    
    h5file['features'].dims[0].label = 'batch'
    h5file['features'].dims[1].label = 'features'
    h5file['targets'].dims[0].label = 'batch'
    h5file['targets'].dims[1].label = 'index'

    h5file['pca_backward'] = pca_backward
    h5file['pca_forward'] = pca_forward
    h5file['zca_backward'] = zca_backward
    h5file['zca_forward'] = zca_forward

    h5file.flush()
    h5file.close()


def fill_subparser_pca_099(subparser):
    """Sets up a subparser to convert the CIFAR10 dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `cifar10` command.

    """
    def partial(directory, output_file):
        convert_cifar10_pca(directory, 0.99, output_file)

    subparser.set_defaults(func=partial)


def fill_subparser_pca_090(subparser):
    """Sets up a subparser to convert the CIFAR10 dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `cifar10` command.

    """
    def partial(directory, output_file):
        convert_cifar10_pca(directory, 0.90, output_file)

    subparser.set_defaults(func=partial)


def fill_subparser_pca_patches_12_090(subparser):
    """Sets up a subparser to convert the CIFAR10 dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `cifar10` command.

    """
    def partial(directory, output_file):
        convert_cifar10_pca(directory, 0.90, output_file, patchsize=12)

    subparser.set_defaults(func=partial)




def fill_subparser_pca_085(subparser):
    """Sets up a subparser to convert the CIFAR10 dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `cifar10` command.

    """
    def partial(directory, output_file):
        convert_cifar10_pca(directory, 0.85, output_file)

    subparser.set_defaults(func=partial)


def crop_patches_color(image, keypoints, patchsize):
    patches = numpy.zeros((len(keypoints), 3*patchsize**2))
    for i, k in enumerate(keypoints):
        patches[i, :] = image[k[0]-patchsize/2:k[0]+patchsize/2, k[1]-patchsize/2:k[1]+patchsize/2,:].flatten()
    return patches


def pca(data, whiten=True, use_gpu = True, batchsize=100, contrast_norm=True, dataset_norm=True, verbose=True):
    data = data.astype(np.float32)
    ncases = data.shape[0]
    nbatches = (ncases - 1) / batchsize + 1
    # contrast normalization

    if contrast_norm:
        if verbose:
            print 'using gpu'
            print 'performing contrast normalization'
        for bidx in range(nbatches):
            start = bidx * batchsize
            end = min((bidx + 1) * batchsize, ncases)
            if use_gpu:
                data[start:end] = theano_subtract_m1(data[start:end])
                data[start:end] = theano_divide_s1(data[start:end])
            else:
                data[start:end] -= data[start:end].mean(1)[:, None]
                s1 = data[start:end].std(1)[:, None]
                data[start:end] /= s1 + s1.mean()

    # normalization over dataset
    m0=0
    s0=1
    if dataset_norm:
        if verbose:
            print 'performing normalization over dataset'
        m0 = compute_mean0_batchwise(data, batchsize=batchsize, use_gpu=use_gpu, verbose=verbose)
        for bidx in range(nbatches):
            start = bidx * batchsize
            end = min((bidx + 1) * batchsize, ncases)
            if use_gpu:
                data[start:end] = theano_subtract_row(data[start:end], m0)
            else:
                data[start:end] -= m0

        s0 = compute_std0_batchwise(data, batchsize=batchsize, use_gpu=use_gpu, verbose=verbose)
        s0 += s0.mean()
        for bidx in range(nbatches):
            start = bidx * batchsize
            end = min((bidx + 1) * batchsize, ncases)
            if use_gpu:
                data[start:end] = theano_divide_row(data[start:end], s0)
            else:
                data[start:end] /= s0

    if verbose:
        print 'computing covariance matrix'
    covmat = compute_covmat_batchwise(data, use_gpu=use_gpu, batchsize=batchsize, verbose=verbose)
    if verbose:
        print 'performing eigenvalue decomposition'
    if whiten:
        V, W, var_fracs = _get_pca_params_from_covmat(covmat, verbose=verbose)
    else:
        V, W, var_fracs = _get_pca_nowhite_params_from_covmat(covmat, verbose=verbose)

    return V, W, m0, s0, var_fracs


#def pca(data, var_fraction, whiten=True):
#    """ principal components analysis of data (columnwise in array data), retaining as many components as required to retain var_fraction of the variance 
#    """
#    from numpy.linalg import eigh
#    u, v = eigh(numpy.cov(data, rowvar=0, bias=1))
#    v = v[:, numpy.argsort(u)[::-1]]
#    u.sort()
#    u = u[::-1]
#    u = u[u.cumsum()<=(u.sum()*var_fraction)]
#    numprincomps = u.shape[0]
#    u[u<SMALL] = SMALL
#    if whiten: 
#        backward_mapping = ((u**(-0.5))[:numprincomps][numpy.newaxis,:]*v[:,:numprincomps]).T
#        forward_mapping = (u**0.5)[:numprincomps][numpy.newaxis,:]*v[:,:numprincomps]
#    else: 
#        backward_mapping = v[:,:numprincomps].T
#        forward_mapping = v[:,:numprincomps]
#    return backward_mapping, forward_mapping, numpy.dot(v[:,:numprincomps], backward_mapping), numpy.dot(forward_mapping, v[:,:numprincomps].T)
