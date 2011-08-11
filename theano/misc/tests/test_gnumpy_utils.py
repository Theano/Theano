import theano
from theano.misc.gnumpy_utils import gnumpy_available

if not gnumpy_available:
    from nose.plugins.skip import SkipTest
    raise SkipTest("gnumpy not installed. Skip test of theano op with pycuda code.")

from theano.misc.gnumpy_utils import garray_to_cudandarray, cudandarray_to_garray

import gnumpy

def test(shape=(3,4,5)):
    """
Make sure that the gnumpy conversion is exact.
"""
    gpu = theano.sandbox.cuda.basic_ops.gpu_from_host
    U = gpu(theano.tensor.ftensor3('U'))
    ii = theano.function([U], gpu(U+1))


    A = gnumpy.rand(*shape)
    A_cnd = garray_to_cudandarray(A)
    B_cnd = ii(A_cnd)
    B = cudandarray_to_garray(B_cnd)
    from numpy import array
    B2 = array(B_cnd)

    u = (A+1).asarray()
    v = B.asarray()
    w = B2
    assert abs(u-v).max() == 0
    assert abs(u-w).max() == 0
