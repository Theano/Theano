import unittest

# We let that import do the init of the back-end if needed.
from .test_basic_ops import mode_with_gpu, GPUMixin

import theano.tensor.nnet.tests.test_neighbours
from ..neighbours import GpuImages2Neibs


class T_GpuImages2Neibs(GPUMixin,
                        theano.tensor.nnet.tests.test_neighbours.T_Images2Neibs):
    mode = mode_with_gpu
    op = GpuImages2Neibs
    dtypes = ['int64', 'float32', 'float64']


if __name__ == '__main__':
    unittest.main()
