from __future__ import absolute_import, print_function, division
import numpy

import theano.sandbox.cuda as cuda
import theano.misc.pycuda_init

if not theano.misc.pycuda_init.pycuda_available:  # noqa
    from nose.plugins.skip import SkipTest
    raise SkipTest("Pycuda not installed. Skip test of theano op with pycuda "
                   "code.")

if cuda.cuda_available is False:  # noqa
    from nose.plugins.skip import SkipTest
    raise SkipTest('Optional theano package cuda disabled')

from theano.misc.pycuda_utils import to_gpuarray, to_cudandarray
import pycuda.gpuarray


def test_to_gpuarray():
    cx = cuda.CudaNdarray.zeros((5, 4))

    px = to_gpuarray(cx)
    assert isinstance(px, pycuda.gpuarray.GPUArray)
    cx[0, 0] = numpy.asarray(1, dtype="float32")
    # Check that they share the same memory space
    assert px.gpudata == cx.gpudata
    assert numpy.asarray(cx[0, 0]) == 1

    assert numpy.allclose(numpy.asarray(cx), px.get())
    assert px.dtype == cx.dtype
    assert px.shape == cx.shape
    assert all(numpy.asarray(cx._strides) * 4 == px.strides)

    # Test when the CudaNdarray is strided
    cx = cx[::2, ::]
    px = to_gpuarray(cx, copyif=True)
    assert isinstance(px, pycuda.gpuarray.GPUArray)
    cx[0, 0] = numpy.asarray(2, dtype="float32")

    # Check that they do not share the same memory space
    assert px.gpudata != cx.gpudata
    assert numpy.asarray(cx[0, 0]) == 2
    assert not numpy.allclose(numpy.asarray(cx), px.get())

    assert px.dtype == cx.dtype
    assert px.shape == cx.shape
    assert not all(numpy.asarray(cx._strides) * 4 == px.strides)

    # Test that we return an error
    try:
        px = to_gpuarray(cx)
        assert False
    except ValueError:
        pass


def test_to_cudandarray():
    px = pycuda.gpuarray.zeros((3, 4, 5), 'float32')
    cx = to_cudandarray(px)
    assert isinstance(cx, cuda.CudaNdarray)
    assert numpy.allclose(px.get(),
                          numpy.asarray(cx))
    assert px.dtype == cx.dtype
    assert px.shape == cx.shape
    assert all(numpy.asarray(cx._strides) * 4 == px.strides)

    try:
        px = pycuda.gpuarray.zeros((3, 4, 5), 'float64')
        to_cudandarray(px)
        assert False
    except ValueError:
        pass

    try:
        to_cudandarray(numpy.zeros(4))
        assert False
    except ValueError:
        pass
