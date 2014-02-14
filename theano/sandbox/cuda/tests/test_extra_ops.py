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
        
        # # Even number of elements
        # a = np.random.random((18,)).astype(config.floatX)
        # assert np.allclose(np.cumsum(a), f(a))

        # # Odd number of elements
        # a = np.random.random((7,)).astype(config.floatX)
        # assert np.allclose(np.cumsum(a), f(a))

        # # Use multiple GPU threadblocks
        # a = np.random.random((2048+2,)).astype(config.floatX)
        # assert np.allclose(np.cumsum(a), f(a))

        # # Use multiple GPU threadblocks
        # a = np.random.random((2048*75+2,)).astype(config.floatX)
        # assert np.allclose(np.cumsum(a), f(a))

        # # Use multiple GPU gridblocks
        # a = np.ones((2048*2048+2,)).astype(config.floatX)
        # assert np.allclose(np.cumsum(a), f(a))

        print "\nBenchmark:"

        import timeit as t
        #theano_time = t.timeit("np.ones((100,))", "import numpy as np", number=1000)

        stmt = "f(a)"
        setup = """
        import numpy as np
        import theano
        import theano.tensor as T
        from theano.tensor.extra_ops import cumsum
        from theano import config
        x = T.vector('x')
        f = theano.function([x], cumsum(x))
        a = np.ones((100000,), dtype=config.floatX)
        """.replace("        ", "")
        theano_time = t.timeit(stmt, setup, number=1000)
        print "Theano:\t", theano_time

        stmt = "np.cumsum(a)"
        setup = """
        import numpy as np
        from theano import config
        a = np.ones((100000,), dtype=config.floatX)
        """.replace("        ", "")
        numpy_time = t.timeit(stmt, setup, number=1000)
        print "Numpy:\t", numpy_time
        print "Speedup: {0}x".format(numpy_time/theano_time)


        # # Extensive testing
        # i = 0;
        # while True:
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

        #     i += 1


        # ### Test 2D case - axis=1 ###
        # x = T.matrix('x')
        # f = theano.function([x], cumsum(x, axis=1))
        
        # # # Even number of elements
        # # print "\n# Even number of elements"
        # # a = np.random.random((18,18)).astype(config.floatX)
        # # assert np.allclose(np.cumsum(a, axis=1), f(a))

        # # # Odd number of elements
        # # print "\n# Odd number of elements"
        # # assert np.allclose(np.cumsum(a, axis=1), f(a))

        # # # Use multiple GPU threadblocks
        # # print "\n# Use multiple GPU threadblocks"
        # # a = np.random.random((2048+2,2048+2)).astype(config.floatX)
        # # assert np.allclose(np.cumsum(a, axis=1), f(a))

        # # # Use multiple GPU threadblocks
        # # print "\n# Use multiple GPU threadblocks"
        # # a = np.ones((10,2048*75+3)).astype(config.floatX)
        # # assert np.allclose(np.cumsum(a, axis=1), f(a))

        # # # Use multiple GPU gridblocks
        # # print "\n# Use multiple GPU gridblocks"
        # # a = np.ones((11,2048*2048+3)).astype(config.floatX)
        # # assert np.allclose(np.cumsum(a, axis=1), f(a))

        # # Extensive testing
        # i = 19000;
        # while True:
        #     a = np.ones((11,i), dtype=config.floatX)
        #     fa = f(a)
        #     npa = np.cumsum(a, axis=1)

        #     if not np.allclose(npa, fa):
        #         print i, np.allclose(npa, fa)  # Test axis=None
        #         print fa
        #         print npa
        #         assert False

        #     if i % 1000 == 0:
        #         print i

        #     i += 1
