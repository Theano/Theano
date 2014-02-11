# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest

import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

import theano.tensor.tests.test_extra_ops
from theano.sandbox.cuda.extra_ops import GpuCumsum

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')

from theano import tensor as T
import numpy as np
import theano
from theano import config
from theano.tensor.extra_ops import cumsum

class TestGpuCumsum(theano.tensor.tests.test_extra_ops.TestCumsumOp):
    mode = mode_with_gpu
    op = GpuCumsum
    dtypes = ['float32']

    def test_GpuCumsum(self):
        ### Test 1D case ###
        x = T.vector('x')
        f = theano.function([x], cumsum(x))

        # Even number of elements
        a = np.random.random((18,)).astype(config.floatX)
        assert np.allclose(np.cumsum(a), f(a))

        # Odd number of elements
        a = np.random.random((7,)).astype(config.floatX)
        assert np.allclose(np.cumsum(a), f(a))

        # Use multiple GPU threadblocks
        a = np.random.random((2048+1,)).astype(config.floatX)
        assert np.allclose(np.cumsum(a), f(a))

        # Use multiple GPU gridblocks
        a = np.random.random((2048*2048+1,)).astype(config.floatX)
        assert np.allclose(np.cumsum(a), f(a))

        # #x = T.tensor3('x')
        # #a = np.random.random((3, 5, 2)).astype(config.floatX)
        # x = T.vector('x')
        # a = np.random.random((6,)).astype(config.floatX)

        # f = theano.function([x], cumsum(x))
        # a = (np.ones(2048*2048+1)+1).astype(config.floatX)
        # print ""
        # print f(a)
        # print np.cumsum(a)
        # assert np.allclose(np.cumsum(a), f(a))  # Test axis=None
        # return

        # #for i in range(3000-1,3000*2):
        # #for i in range(3,2048*100,2048):
        # i = 145000;
        # while True:
        #     #a = np.random.random((i,)).astype(config.floatX)
        #     a = np.ones((i,), dtype=config.floatX)
        #     fa = f(a)
        #     npa = np.cumsum(a)

        #     if not np.allclose(npa, fa):
        #         print i, np.allclose(npa, fa)  # Test axis=None
        #         print fa
        #         print npa
        #         assert False

        #     if i % 1000 == 0:
        #        print i

        #     i += 1
