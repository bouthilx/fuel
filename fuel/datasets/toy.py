# -*- coding: utf-8 -*-

import logging

import numpy

from collections import OrderedDict

from fuel import config
from fuel.datasets import IndexableDataset


logger = logging.getLogger(__name__)


class Spiral(IndexableDataset):
    u"""Toy dataset containing points sampled from spirals on a 2d plane.

    The dataset contains 3 sources:

    * features -- the (x, y) position of the datapoints
    * position -- the relative position on the spiral arm
    * label -- the class labels (spiral arm)

    .. plot::

        from fuel.datasets.toy import Spiral

        ds = Spiral(classes=3)
        features, position, label = ds.get_data(None, slice(0, 500))

        plt.title("Datapoints drawn from Spiral(classes=3)")
        for l, m in enumerate(['o', '^', 'v']):
            mask = label == l
            plt.scatter(features[mask,0], features[mask,1],
                        c=position[mask], marker=m, label="label==%d"%l)
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.legend()
        plt.colorbar()
        plt.xlabel("features[:,0]")
        plt.ylabel("features[:,1]")
        plt.show()

    Parameters
    ----------
    num_examples : int
        Number of datapoints to create.
    classes : int
        Number of spiral arms.
    cycles : float
        Number of turns the arms take.
    noise : float
        Add normal distributed noise with standard deviation *noise*.

    """
    def __init__(self, num_examples=1000, classes=1, cycles=1., noise=0.0,
                 **kwargs):
        seed = kwargs.pop('seed', config.default_seed)
        rng = numpy.random.RandomState(seed)
        # Create dataset
        pos = rng.uniform(size=num_examples, low=0, high=cycles)
        label = rng.randint(size=num_examples, low=0, high=classes)
        radius = (2 * pos + 1) / 3.
        phase_offset = label * (2 * numpy.pi) / classes

        features = numpy.zeros(shape=(num_examples, 2), dtype='float32')

        features[:, 0] = radius * numpy.sin(2 * numpy.pi * pos + phase_offset)
        features[:, 1] = radius * numpy.cos(2 * numpy.pi * pos + phase_offset)
        features += noise * rng.normal(size=(num_examples, 2))

        data = OrderedDict([
            ('features', features),
            ('position', pos),
            ('label', label),
        ])

        super(Spiral, self).__init__(data, **kwargs)


class SwissRoll(IndexableDataset):
    """Dataset containing points from a 3-dimensional Swiss roll.

    The dataset contains 2 sources:

    * features -- the x, y and z position of the datapoints
    * position -- radial and z position on the manifold

    .. plot::

        from fuel.datasets.toy import SwissRoll
        import mpl_toolkits.mplot3d.axes3d as p3
        import numpy as np

        ds = SwissRoll()
        features, pos = ds.get_data(None, slice(0, 1000))

        color = pos[:,0]
        color -= color.min()
        color /= color.max()

        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.scatter(features[:,0], features[:,1], features[:,2],
                   'x', c=color)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.view_init(10., 10.)
        plt.show()

    Parameters
    ----------
    num_examples : int
        Number of datapoints to create.
    noise : float
        Add normal distributed noise with standard deviation *noise*.

    """
    def __init__(self, num_examples=1000, noise=0.0, **kwargs):
        cycles = 1.5
        seed = kwargs.pop('seed', config.default_seed)
        rng = numpy.random.RandomState(seed)
        pos = rng.uniform(size=num_examples, low=0, high=1)
        phi = cycles * numpy.pi * (1 + 2 * pos)
        radius = (1 + 2 * pos) / 3

        x = radius * numpy.cos(phi)
        y = radius * numpy.sin(phi)
        z = rng.uniform(size=num_examples, low=-1, high=1)

        features = numpy.zeros(shape=(num_examples, 3), dtype='float32')
        features[:, 0] = x
        features[:, 1] = y
        features[:, 2] = z
        features += noise * rng.normal(size=(num_examples, 3))

        position = numpy.zeros(shape=(num_examples, 2), dtype='float32')
        position[:, 0] = pos
        position[:, 1] = z

        data = OrderedDict([
            ('features', features),
            ('position', position),
        ])

        super(SwissRoll, self).__init__(data, **kwargs)


class Gaussians(IndexableDataset):
    u"""Toy dataset containing points sampled from spirals on a 2d plane.

    The dataset contains 3 sources:

    * features -- the (x, y) position of the datapoints
    * position -- the relative position on the spiral arm
    * label -- the class labels (spiral arm)

    .. plot::

        from pcas.datasets.gaussians import Gaussians

        ds = Gaussians(nb_of_modes=10, nb_of_dimensions=2, nb_of_points=1000)
        features, label = ds.get_data(None, slice(0, 500))

        plt.title("Datapoints drawn from Gaussians(10, 2, 1000)")
        for mode in xrange(options.nb_of_modes):
            mode_data_points = data_points[data_modes == mode]
            plt.scatter(mode_data_points[:, 0], mode_data_points[:, 1], s=2,
                        alpha=0.5)
        plt.show()

    Parameters
    ----------
    """
    def __init__(self, nb_of_modes, nb_of_dimensions, nb_of_points, rng=None,
                 **kwargs):
        # Create dataset
        features, modes = Gaussians.generate(
            nb_of_modes, nb_of_dimensions, nb_of_points, rng)

        data = OrderedDict([
            ('features', features),
            ('labels', modes),
        ])

        super(Gaussians, self).__init__(data, **kwargs)

    @staticmethod
    def generate(nb_of_modes, nb_of_dimensions, nb_of_points, rng=None):
        if rng is None:
            rng = numpy.random.RandomState((2017, 8, 10))

        nb_of_points_per_mode = int(nb_of_points / float(nb_of_modes))

        if nb_of_points_per_mode * nb_of_modes != nb_of_points:
            logger.warning("Will generate %d points instead of %d to have an "
                           "equal number of points for each mode." %
                           (nb_of_points_per_mode * nb_of_modes, nb_of_points))

        data_points = None
        data_modes = None
        for mode in xrange(nb_of_modes):
            mean = rng.uniform(-10, 10, size=nb_of_dimensions)
            mode_data_points = rng.normal(
                loc=mean,
                size=(nb_of_points_per_mode, nb_of_dimensions))
            if data_points is None:
                data_points = mode_data_points
                data_modes = numpy.zeros(nb_of_points_per_mode)
            else:
                data_points = numpy.concatenate((
                    data_points, mode_data_points))
                data_modes = numpy.concatenate((
                    data_modes, numpy.ones(nb_of_points_per_mode) * mode))

        idx = numpy.arange(data_points.shape[0])
        rng.shuffle(idx)
        data_points = data_points[idx]
        data_modes = data_modes[idx]

        return numpy.cast['float32'](data_points), data_modes
