import unittest

from theano.tensor.nnet.tests import test_neighbours
# We let that import do the init of the back-end if needed.
from .test_basic_ops import (mode_with_gpu,
                             mode_without_gpu)

from ..neighbours import GpuImages2Neibs


class T_GpuImages2Neibs(test_neighbours.T_Images2Neibs):
    mode = mode_with_gpu
    op = GpuImages2Neibs
    dtypes = ['int64', 'float32', 'float64']
