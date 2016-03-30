from __future__ import absolute_import, print_function, division
import numpy

import theano
from theano.misc.gnumpy_utils import gnumpy_available

if not gnumpy_available:  # noqa
    from nose.plugins.skip import SkipTest
    raise SkipTest("gnumpy not installed. Skip test related to it.")

from theano.misc.gnumpy_utils import (garray_to_cudandarray,
                                      cudandarray_to_garray)

import gnumpy


def test(shape=(3, 4, 5)):
    """
    Make sure that the gnumpy conversion is exact from garray to
    CudaNdarray back to garray.
    """
    gpu = theano.sandbox.cuda.basic_ops.gpu_from_host
    U = gpu(theano.tensor.ftensor3('U'))
    ii = theano.function([U], gpu(U + 1))

    A = gnumpy.rand(*shape)
    A_cnd = garray_to_cudandarray(A)
    assert A_cnd.shape == A.shape
    # dtype always float32
    # garray don't have strides
    B_cnd = ii(A_cnd)
    B = cudandarray_to_garray(B_cnd)
    assert A_cnd.shape == A.shape
    from numpy import array

    u = (A + 1).asarray()
    v = B.asarray()
    w = array(B_cnd)
    assert (u == v).all()
    assert (u == w).all()


def test2(shape=(3, 4, 5)):
    """
    Make sure that the gnumpy conversion is exact from CudaNdarray to
    garray back to CudaNdarray.
    """
    gpu = theano.sandbox.cuda.basic_ops.gpu_from_host
    U = gpu(theano.tensor.ftensor3('U'))
    theano.function([U], gpu(U + 1))

    A = numpy.random.rand(*shape).astype('float32')
    A_cnd = theano.sandbox.cuda.CudaNdarray(A)
    A_gar = cudandarray_to_garray(A_cnd)
    assert A_cnd.shape == A_gar.shape
    # dtype always float32
    # garray don't have strides

    B = garray_to_cudandarray(A_gar)

    assert A_cnd.shape == B.shape
    # dtype always float32
    assert A_cnd._strides == B._strides
    assert A_cnd.gpudata == B.gpudata
    v = numpy.asarray(B)
    assert (v == A).all()


def test_broadcast_dims():
    """
    Test with some dimensions being 1.
    CudaNdarray use 0 for strides for those dimensions.
    """
    test((1, 2, 3))
    test((2, 1, 3))
    test((2, 3, 1))
    test2((1, 2, 3))
    test2((2, 1, 3))
    test2((2, 3, 1))
