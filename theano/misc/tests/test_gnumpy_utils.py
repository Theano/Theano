import numpy

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
    assert A_cnd.shape == A.shape
#    assert A_cnd.dtype == A_gar.dtype # dtype always float32
#    assert A_cnd._strides == A_gar.strides, garray don't have strides
    B_cnd = ii(A_cnd)
    B = cudandarray_to_garray(B_cnd)
    assert A_cnd.shape == A.shape
    from numpy import array
    B2 = array(B_cnd)

    u = (A+1).asarray()
    v = B.asarray()
    w = B2
    assert abs(u-v).max() == 0
    assert abs(u-w).max() == 0

def test2(shape=(3,4,5)):
    """
Make sure that the gnumpy conversion is exact.
"""
    gpu = theano.sandbox.cuda.basic_ops.gpu_from_host
    U = gpu(theano.tensor.ftensor3('U'))
    ii = theano.function([U], gpu(U+1))


    A = numpy.random.rand(*shape).astype('float32')
    A_cnd = theano.sandbox.cuda.CudaNdarray(A)
    A_gar = cudandarray_to_garray(A_cnd)
    assert A_cnd.shape == A_gar.shape
#    assert A_cnd.dtype == A_gar.dtype # dtype always float32
#    assert A_cnd._strides == A_gar.strides, garray don't have strides

    B = garray_to_cudandarray(A_gar)
    from numpy import array
    B2 = array(B)

    assert A_cnd.shape == B.shape
#    assert A_cnd.dtype == B.dtype # dtype always float32
    assert A_cnd._strides == B._strides
    assert A_cnd.gpudata == B.gpudata
    v = numpy.asarray(B)
    assert abs(v-A).max() == 0
