import os
import tarfile

import theano
import theano.tensor as T

# define some symbolic variables
theano_matrix1 = T.matrix(name='theano_matrix1')
theano_matrix2 = T.matrix(name='theano_matrix2')

# define some functions

# dot product/matrix product
theano_dot = theano.function([theano_matrix1, theano_matrix2], T.dot(theano_matrix1, theano_matrix2), name='theano_dot')

theano_scalar = T.fscalar(name='theano_scalar')
theano_scale = theano.function([theano_matrix1, theano_scalar], theano_matrix1 * theano_scalar, name='scale')

# elementwise product
theano_multiply = theano.function([theano_matrix1, theano_matrix2], theano_matrix1 * theano_matrix2, name='theano_multiply')

theano_row_vector = T.row(name='theano_row_vector')
theano_col_vector = T.col(name='theano_col_vector')

theano_subtract_row = theano.function([theano_matrix1, theano_row_vector], theano_matrix1 - theano_row_vector, name='theano_subtract_row')
theano_divide_row = theano.function([theano_matrix1, theano_row_vector], theano_matrix1 / theano_row_vector, name='theano_subtract_row')
theano_subtract_col = theano.function([theano_matrix1, theano_col_vector], theano_matrix1 - theano_col_vector, name='theano_subtract_col')
theano_divide_col = theano.function([theano_matrix1, theano_col_vector], theano_matrix1 / theano_col_vector, name='theano_subtract_col')

theano_var1 = theano.function([theano_matrix1], T.var(theano_matrix1, 1), name='theano_var1')
theano_mean0 = theano.function([theano_matrix1], T.mean(theano_matrix1, 0), name='theano_mean0')
theano_mean1 = theano.function([theano_matrix1], T.mean(theano_matrix1, 1), name='theano_mean1')

_theano_ssd0 = T.sum((theano_matrix1 - theano_row_vector)**2, 0)
theano_ssd0 = theano.function([theano_matrix1, theano_row_vector], _theano_ssd0, name='ssd0')
theano_sqrt = theano.function([theano_matrix1], T.sqrt(theano_matrix1), name='sqrt')


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
    #    subparser.set_defaults(func=convert_cifar10)
    return convert_cifar10


@check_exists(required_files=[DISTRIBUTION_FILE])
def convert_cifar10_pca(directory, output_directory, variance, output_filename='cifar10.hdf5', patchsize=None):
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

    pca_transf, pca_invtransf, m0, s0, var_fracs = pca(train_features,
            whiten=False, batchsize=10000)
    # need to convert to variance!!!
    nprinc = 781
    train_features = whiten(train_features, pca_transf, m0, s0, nprinc)

    if patchsize is None:
        test_features = whiten(test_features, pca_transf, m0, s0, nprinc)

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

    h5file['pca_transf'] = pca_transf
    h5file['pca_invtransf'] = pca_invtransf
#    h5file['zca_backward'] = zca_backward
#    h5file['zca_forward'] = zca_forward

    h5file.flush()
    h5file.close()

    return (output_path,)


def fill_subparser_pca_099(subparser):
    """Sets up a subparser to convert the CIFAR10 dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `cifar10` command.

    """
    def partial(directory, output_directory, output_filename='cifar10.hdf5'):
        return convert_cifar10_pca(directory, output_directory, 0.99, output_filename)

#    subparser.set_defaults(func=partial)
    return partial


def fill_subparser_pca_090(subparser):
    """Sets up a subparser to convert the CIFAR10 dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `cifar10` command.

    """
    def partial(directory, output_directory, output_filename='cifar10.hdf5'):
        return convert_cifar10_pca(directory, output_directory, 0.90, output_filename)

#    subparser.set_defaults(func=partial)
    return partial


def fill_subparser_pca_patches_12_090(subparser):
    """Sets up a subparser to convert the CIFAR10 dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `cifar10` command.

    """
    def partial(directory, output_directory, output_filename='cifar10.hdf5'):
        return convert_cifar10_pca(directory, output_directory, 0.90,
                output_filename, patchsize=12)

#    subparser.set_defaults(func=partial)
    return partial


def fill_subparser_pca_085(subparser):
    """Sets up a subparser to convert the CIFAR10 dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `cifar10` command.

    """
    def partial(directory, output_directory, output_filename='cifar10.hdf5'):
        return convert_cifar10_pca(directory, output_directory, 0.85,
                output_filename)

#    subparser.set_defaults(func=partial)
    return partial


def crop_patches_color(image, keypoints, patchsize):
    patches = numpy.zeros((len(keypoints), 3*patchsize**2))
    for i, k in enumerate(keypoints):
        patches[i, :] = image[k[0]-patchsize/2:k[0]+patchsize/2, k[1]-patchsize/2:k[1]+patchsize/2,:].flatten()
    return patches


def pca(data, whiten=True, use_gpu = True, batchsize=100, contrast_norm=True, dataset_norm=True, verbose=True):
    data = data.astype(numpy.float32)
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


_theano_std0_sum = T.sum((theano_matrix1 - theano_row_vector)**2 / (theano_scalar-1), # Bessel's correction
                         0)
theano_subtract_m1 = theano.function(
    inputs=[theano_matrix1],
    outputs=theano_matrix1 - T.mean(theano_matrix1, 1).dimshuffle(0, 'x'))

_s1 = T.std(theano_matrix1, 1).dimshuffle(0, 'x')
theano_divide_s1 = theano.function(
    inputs=[theano_matrix1],
    outputs=theano_matrix1 / (_s1 + _s1.mean()))


#NOTE: BE CAREFUL TO REUSE num_processed_cases only in the corresponding function!!!! bad example: feeding num_processed_cases returned by compute_mean0_iteratively to compute_cov_iteratively.
def compute_mean0_iteratively(batch, use_gpu=True, m0=None, nproccases=0):
    """Computes the mean over vertically stacked data points iteratively.

    This function can be used to compute the mean over data points online.

    Args:
        batch: a batch of data
        use_gpu: Indicates, whether theano should be used or not.
        m0: if this is not the first iteration, this should hold the result of
            the previous iteration
        num_processed_cases: this should hold the number of processed data
            points up to including the last iteration
    Returns:
        A tuple containing the mean over data points and the number of
        processed cases.
    """
    if m0 is None:
        if use_gpu:
            return theano_mean0(batch).reshape((1, -1)), batch.shape[0]
        else:
            return batch.mean(0).reshape((1, -1)), batch.shape[0]
    else:
        w0 = numpy.float32(nproccases) / (nproccases + batch.shape[0])
        nproccases += batch.shape[0]
        if use_gpu:
            return w0 * m0 + (1-w0) * theano_mean0(batch), nproccases
        else:
            return w0 * m0 + (1-w0) * batch.mean(0), nproccases

def compute_mean0_batchwise(data, batchsize=100, use_gpu=True, verbose=True):
    """Computes the mean over vertically stacked data points batchwise.

   This function partitions the data into batches and updates the mean batch
    after batch.

    Args:
        data: The vertically stacked data points
        batchsize: The number of data points in each batch
        use_gpu: Indicates, whether theano should be used or not.
        verbose: Set to True for debug output
    Returns:
        The mean over data points.
    """
    ncases, ndim = data.shape
    nbatches = (ncases - 1) / batchsize + 1
    m0 = numpy.zeros((1, ndim), dtype=theano.config.floatX)
    nproccases = 0
    for bidx in range(nbatches):
        if verbose:
            print "processing batch %d of %d" % (bidx+1, nbatches)
        start = bidx * batchsize
        end = min((bidx + 1) * batchsize, ncases)
        m0[:, :], nproccases = compute_mean0_iteratively(
            batch=data[start:end], use_gpu=use_gpu, m0=m0, nproccases=nproccases)
    return m0

def compute_std0_batchwise(data, use_gpu=True, batchsize=100, verbose=True):
    """Computes the standard deviation over stacked data points batchwise.

    This function partitions the data into batches, updates the variance
    batch after batch and then computes the square root of the variance.

    Args:
        data: The vertically stacked data points
        use_gpu: Indicates, whether theano should be used or not.
        batchsize: The number of data points in each batch
    Returns:
        The standard deviation over stacked data points.
    """
    ncases, ndim = data.shape
    nbatches = (ncases - 1) / batchsize + 1
    m0 = compute_mean0_batchwise(data, use_gpu=True, batchsize=batchsize)
    if use_gpu:
        s0 = theano.shared(numpy.zeros((1, ndim), dtype=theano.config.floatX), name='s0')
        s0_update_f = theano.function(inputs=[theano_matrix1, theano_row_vector, theano_scalar],
                                      outputs=[],
                                      updates={s0: s0 + _theano_std0_sum})
    else:
        s0 = numpy.zeros((1, data.shape[1]), dtype=theano.config.floatX)
    if data.shape[0] < batchsize:
        batchsize = data.shape[0]
    for bidx in range(nbatches):
        if verbose:
            print "processing batch %d of %d" % (bidx+1, nbatches)
        start = bidx * batchsize
        end = min((bidx + 1) * batchsize, ncases)
        if use_gpu:
            s0_update_f(data[start:end], m0, ncases)
        else:
            s0[:, :] += numpy.sum((data[start:end] - m0)**2 / (ncases-1), axis=0)

    if use_gpu:
        return numpy.sqrt(s0.get_value())
    else:
        return numpy.sqrt(s0)

def compute_covmat_batchwise(data, use_gpu=True, batchsize=100, verbose=True):
    """Computes the covariance matrix for vertically stacked data points.

    Args:
        data: The vertically stacked data points.
        use_gpu: Indicates, whether theano should be used or not.
        batchsize: The number of data points in each batch
        verbose: Set to True for debug output
    Returns:
        The covariance matrix for the data.
    """
    ncases, ndim = data.shape
    nbatches = (ncases - 1) / batchsize + 1
    if use_gpu:
        C = theano.shared(numpy.zeros((ndim, ndim),
                                   dtype=theano.config.floatX), name='C')
        update_C_f = theano.function(inputs=[theano_matrix1, theano_matrix2, theano_scalar],
                                      outputs=[],
                                      updates={
                                          C: C + T.dot(theano_matrix1, theano_matrix2) / theano_scalar
                                      })
    else:
        C = numpy.zeros((ndim, ndim), dtype=theano.config.floatX)
    for bidx in range(nbatches):
        if verbose:
            print "processing batch %d of %d" % (bidx+1, nbatches)
        start = bidx * batchsize
        end = min((bidx + 1) * batchsize, ncases)
        if use_gpu:
            update_C_f(data.T[:, start:end],
                       data[start:end, :],
                       numpy.float32(ncases))
        else:
            C += numpy.dot(data.T[:, start:end],
                       data[start:end, :]) / numpy.float32(ncases)
    if use_gpu:
        return C.get_value()
    else:
        return C

def _get_pca_params_from_covmat(C, verbose=True):
    """Computes pca parameters from the given data covariance matrix.

    Args:
        C: The data covariance matrix
        verbose: Set to True for debug output
    Returns:
        A tuple containing the PCA-Matrix, the inverse PCA Matrix and an array
        of fractions of variance (how much of the variance will be retained, if
        all rows after this one are dropped from V).
    """
    u, v = numpy.linalg.eigh(C)
    v = v[:, numpy.argsort(u)[::-1]]
    u.sort()
    u = u[::-1]
    u = u[u.cumsum()<u.sum()*(0.99999)] # throw away some eigenvalues for numerical stability
    var_fracs = u.cumsum()/u.sum()
    numprincomps = u.shape[0]
    V = ((u**(-0.5))[:numprincomps][numpy.newaxis, :]*v[:, :numprincomps]).T
    W = (u**0.5)[:numprincomps][numpy.newaxis, :]*v[:, :numprincomps]

    return V, W, var_fracs

def _get_pca_nowhite_params_from_covmat(C, verbose=True):
    """Computes pca parameters from the given data covariance matrix.

    Args:
        C: The data covariance matrix
        verbose: Set to True for debug output
    Returns:
        A tuple containing the PCA-Matrix, the inverse PCA Matrix and an array
        of fractions of variance (how much of the variance will be retained, if
        all rows after this one are dropped from V).
    """
    u, v = numpy.linalg.eigh(C)
    v = v[:, numpy.argsort(u)[::-1]]
    u.sort()
    u = u[::-1]
    u = u[u.cumsum()<u.sum()*(0.99999)] # throw away some eigenvalues for numerical stability
    var_fracs = u.cumsum()/u.sum()
    numprincomps = u.shape[0]
    V = v[:, :numprincomps].T
    W = v[:, :numprincomps]

    return V, W, var_fracs


def _get_zca_params_from_covmat(C, verbose=True):

    u, v = numpy.linalg.eigh(C)
    v = v[:, numpy.argsort(u)[::-1]]
    u.sort()
    u = u[::-1]
    u = u[u.cumsum()<u.sum()*(0.999)] # throw away some eigenvalues for numerical stability
    var_fracs = u.cumsum()/u.sum()
    numprincomps = u.shape[0]
    V = ((u**(-0.5))[:numprincomps][numpy.newaxis, :]*v[:, :numprincomps]).T
    V = (v[:, :numprincomps]).dot(V)

    W = (u**0.5)[:numprincomps][numpy.newaxis, :]*v[:, :numprincomps]
    W = W.dot(v[:, :numprincomps].T)

    return V, W

def pca(data, whiten=True, use_gpu = True, batchsize=100, contrast_norm=True, dataset_norm=True, verbose=True):
    data = data.astype(numpy.float32)
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

def zca(data, use_gpu = True, batchsize=100, verbose=True):

    data = data.astype(numpy.float32)
    ncases = data.shape[0]
    nbatches = (ncases - 1) / batchsize + 1
    # contrast normalization
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

    if verbose:
        print 'computing covariance matrix'
    covmat = compute_covmat_batchwise(data, use_gpu=use_gpu, batchsize=batchsize, verbose=verbose)
    if verbose:
        print 'performing eigenvalue decomposition'
    V, W = _get_zca_params_from_covmat(covmat, verbose=verbose)

    return V, W

def whiten(data, V, m0, s0, nprincomps, batchsize=500, contrast_norm=True, dataset_norm=True, use_gpu=True, verbose=True):
    data = data.astype(numpy.float32)
    ncases = data.shape[0]
    nbatches = (ncases - 1) / batchsize + 1

    data_white = numpy.zeros((ncases, nprincomps), dtype=numpy.float32)

    for bidx in range(nbatches):
        start = bidx * batchsize
        end = min((bidx + 1) * batchsize, ncases)
        if use_gpu:
            if contrast_norm:
                data[start:end] = theano_subtract_m1(data[start:end])
                data[start:end] = theano_divide_s1(data[start:end])
            if dataset_norm:
                data[start:end] = theano_subtract_row(data[start:end], m0)
                data[start:end] = theano_divide_row(data[start:end], s0)
            data_white[start:end] = theano_dot(data[start:end], V[:nprincomps].T)
        else:
            if contrast_norm:
                data[start:end] -= data[start:end].mean(1)[:, None]
                s1 = data[start:end].std(1)[:, None]
                data[start:end] /= s1 + s1.mean()
            if dataset_norm:
                data[start:end] -= m0
                data[start:end] /= s0
            data_white[start:end] = numpy.dot(data[start:end], V[:nprincomps].T)
    return data_white

def whitenX(data, V, m0, s0, nprincomps, batchsize=500, contrast_norm=True, dataset_norm=True, use_gpu=True, verbose=True):
    data = data.astype(numpy.float32)
    ncases = data.shape[0]
    nbatches = (ncases - 1) / batchsize + 1

    V = V[:nprincomps]
    V = numpy.concatenate((V,-1*V),0).T

    data_white = numpy.zeros((ncases, nprincomps*2), dtype=numpy.float32)

    for bidx in range(nbatches):
        start = bidx * batchsize
        end = min((bidx + 1) * batchsize, ncases)
        if use_gpu:
            if contrast_norm:
                data[start:end] = theano_subtract_m1(data[start:end])
                data[start:end] = theano_divide_s1(data[start:end])
            if dataset_norm:
                data[start:end] = theano_subtract_row(data[start:end], m0)
                data[start:end] = theano_divide_row(data[start:end], s0)
            data_white[start:end] = theano_dot(data[start:end], V)
        else:
            if contrast_norm:
                data[start:end] -= data[start:end].mean(1)[:, None]
                s1 = data[start:end].std(1)[:, None]
                data[start:end] /= s1 + s1.mean()
            if dataset_norm:
                data[start:end] -= m0
                data[start:end] /= s0
            data_white[start:end] = numpy.dot(data[start:end], V)
    return (data_white > 0.)*data_white


def zca_whiten(data, W, batchsize=500, use_gpu=True, verbose=True):
    data = data.astype(numpy.float32)
    ncases = data.shape[0]
    nbatches = (ncases - 1) / batchsize + 1

    data_white = numpy.zeros((ncases, data.shape[1]), dtype=numpy.float32)
    for bidx in range(nbatches):
        start = bidx * batchsize
        end = min((bidx + 1) * batchsize, ncases)
        if use_gpu:
            data[start:end] = theano_subtract_m1(data[start:end])
            data[start:end] = theano_divide_s1(data[start:end])
            data_white[start:end] = theano_dot(data[start:end], W)
        else:
            data[start:end] -= data[start:end].mean(1)[:, None]
            s1 = data[start:end].std(1)[:, None]
            data[start:end] /= s1 + s1.mean()
            data_white[start:end] = numpy.dot(data[start:end], W)
    return data_white    

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
