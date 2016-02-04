# Skip test if cuda_ndarray is not available.
from __future__ import absolute_import, print_function, division
from nose.plugins.skip import SkipTest

import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

import theano.tensor.nnet.tests.test_neighbours
from theano.sandbox.cuda.neighbours import GpuImages2Neibs

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


class T_GpuImages2Neibs(theano.tensor.nnet.tests.test_neighbours.T_Images2Neibs):
    mode = mode_with_gpu
    op = GpuImages2Neibs
    dtypes = ['float32']

if __name__ == '__main__':
    unittest.main()
