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

from mlpython.misc.utils import Timer

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
        print "\nEven number of elements"
        print f(a)
        print np.cumsum(a)
        assert np.allclose(np.cumsum(a), f(a))

        # Odd number of elements
        a = np.random.random((7,)).astype(config.floatX)
        print "\nOdd number of elements"
        print f(a)
        print np.cumsum(a)
        assert np.allclose(np.cumsum(a), f(a))

        # Use multiple GPU threadblocks
        a = np.random.random((2048+1,)).astype(config.floatX)
        print "\nUse multiple GPU threadblocks"
        print f(a)
        print np.cumsum(a)
        assert np.allclose(np.cumsum(a), f(a))

        # Use multiple GPU threadblocks
        a = np.random.random((2048*80+1,)).astype(config.floatX)
        print "\nUse multiple GPU threadblocks 2"
        print f(a)
        print np.cumsum(a)
        assert np.allclose(np.cumsum(a), f(a))

        # Use multiple GPU gridblocks
        #a = (np.random.random((2048*2048+1,)).astype(config.floatX) - 0.5) * 10.
        a = np.floor(np.random.random((2048*2048+1,)) * 10).astype(config.floatX)
        #a = np.ones((2048*2048+1,)).astype(config.floatX)
        print "\nUse multiple GPU gridblocks"
        print f(a)
        print np.cumsum(a)
        assert np.allclose(np.cumsum(a), f(a))

        # x = T.vector('x')
        # f = theano.function([x], cumsum(x))
        # a = (np.ones(0)+1).astype(config.floatX)
        # print ""
        # print f(a)
        # print np.cumsum(a)
        # assert np.allclose(np.cumsum(a), f(a))  # Test axis=None
        # return

        # #for i in range(3000-1,3000*2):
        # #for i in range(3,2048*100,2048):
        # i = 150000;
        # import time
        # start = time.time()
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
        #         print i
        #         print time.time() - start
        #         start = time.time()
        #         #i-=1000

        #     if i == 80000:
        #         f = theano.function([x], cumsum(x))

        #     i += 1
