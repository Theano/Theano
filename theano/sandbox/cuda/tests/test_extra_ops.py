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
from theano.tensor.extra_ops import cumsum, diff

from mlpython.misc.utils import Timer

class TestGpuCumsum(theano.tensor.tests.test_extra_ops.TestCumsumOp):
    mode = mode_with_gpu
    op = GpuCumsum
    dtypes = ['float32']

    def test_benchmark_1D_vs_2D(self):
        print "\nBenchmark:"

        from theano import sandbox, Out
        import time

        vlen = 40 * 1024 * 2048  # 10 x # cores x # threads per core
        iters = 25

        x = theano.shared(np.ones((vlen,), dtype=config.floatX), borrow=False)
        res = Out(sandbox.cuda.basic_ops.gpu_from_host(cumsum(x)), borrow=True)
        f = theano.function([], res)

        print f.maker.fgraph.toposort()
        t0 = time.time()
        for i in xrange(iters):
            r = f()
        t1 = time.time()
        
        print 'Looping %d times took' % iters, t1 - t0, 'seconds'
        print 'Result is', r
        print 'Numpy result is', np.asarray(r)
        

        # x = theano.shared(np.ones((1,vlen), dtype=config.floatX), borrow=True)
        # f = theano.function([], Out(sandbox.cuda.basic_ops.gpu_from_host(cumsum(x,axis=1)), borrow=True))

        # print f.maker.fgraph.toposort()
        # t0 = time.time()
        # for i in xrange(iters):
        #     r = f()
        # t1 = time.time()
        
        # print 'Looping %d times took' % iters, t1 - t0, 'seconds'
        # print 'Result is', r
        # print 'Numpy result is', np.asarray(r)

        # print 'Used the', config.device


    def test_GpuCumsum(self):
        ### Test 1D case ###
        x = T.vector('x')
        f = theano.function([x], cumsum(x))
        
        # Even number of elements
        a = np.random.random((18,)).astype(config.floatX)
        print f(a)
        print np.cumsum(a)
        assert np.allclose(np.cumsum(a), f(a))

        # Odd number of elements
        a = np.random.random((7,)).astype(config.floatX)
        assert np.allclose(np.cumsum(a), f(a))

        # Use multiple GPU threadblocks
        a = np.random.random((2048+2,)).astype(config.floatX)
        assert np.allclose(np.cumsum(a), f(a))

        # Use multiple GPU threadblocks
        a = np.random.random((2048*75+2,)).astype(config.floatX)
        assert np.allclose(np.cumsum(a), f(a))

        # Use multiple GPU gridblocks
        a = np.ones((2048*2048+2,)).astype(config.floatX)
        assert np.allclose(np.cumsum(a), f(a))


        # Extensive testing
        for i in xrange(int(1e3)*5):
            a = np.ones((i,), dtype=config.floatX)

            fa = f(a)
            npa = np.cumsum(a)

            if not np.allclose(npa, fa):
                print i, np.allclose(npa, fa)  # Test axis=None
                print fa
                print npa
                assert False

            if i % 1000 == 0:
                print i


        #for axis in xrange(2):
        for axis in xrange(2):
            ### Test 2D case - axis=1 ###
            x = T.matrix('x')
            f = theano.function([x], cumsum(x, axis=axis))
            
            # Even number of elements
            print "\n# Even number of elements (axis={0})".format(axis)
            a = np.random.random((18,18)).astype(config.floatX)
            assert np.allclose(np.cumsum(a, axis=axis), f(a))

            # Odd number of elements
            print "\n# Odd number of elements (axis={0})".format(axis)
            a = np.random.random((21,21)).astype(config.floatX)
            assert np.allclose(np.cumsum(a, axis=axis), f(a))

            # Use two GPU threadblocks
            print "\n# Use two GPU threadblocks (axis={0})".format(axis)
            a = np.random.random((2048+2,2048+2)).astype(config.floatX)
            assert np.allclose(np.cumsum(a, axis=axis), f(a))

            # Use multiple GPU threadblocks
            print "\n# Use multiple GPU threadblocks (axis={0})".format(axis)
            a = np.ones((10,2048*75+3)).astype(config.floatX)
            assert np.allclose(np.cumsum(a, axis=axis), f(a))
            
            a = np.ones((2048*75+3,10)).astype(config.floatX)
            assert np.allclose(np.cumsum(a, axis=axis), f(a))

            # Use multiple GPU gridblocks
            print "\n# Use multiple GPU gridblocks (axis={0})".format(axis)
            a = np.ones((11,2048*2048+3)).astype(config.floatX)
            assert np.allclose(np.cumsum(a, axis=axis), f(a))
            a = np.ones((2048*2048+3,11)).astype(config.floatX)
            assert np.allclose(np.cumsum(a, axis=axis), f(a))

            # Extensive testing for the first 10k sizes
            for i in xrange(int(1e3)*5):
                a = np.ones((11,i), dtype=config.floatX)
                fa = f(a)
                npa = np.cumsum(a, axis=axis)

                if not np.allclose(npa, fa):
                    print i, np.allclose(npa, fa)  # Test axis=None
                    print fa
                    print npa
                    assert False

                a = np.ones((i,11), dtype=config.floatX)
                fa = f(a)
                npa = np.cumsum(a, axis=axis)

                if not np.allclose(npa, fa):
                    print i, np.allclose(npa, fa)  # Test axis=None
                    print fa
                    print npa
                    assert False

                if i % 1000 == 0:
                    print i