"""
This file is an example of view the memory allocated by pycuda in a GpuArray
in a CudaNdarray to be able to use it in Theano.

This also serve as a test for the function: cuda_ndarray.from_gpu_pointer
"""
from __future__ import absolute_import, print_function, division

import sys

import numpy as np

import theano
import theano.sandbox.cuda as cuda_ndarray
import theano.misc.pycuda_init

if not theano.misc.pycuda_init.pycuda_available:  # noqa
    from nose.plugins.skip import SkipTest
    raise SkipTest("Pycuda not installed."
                   " We skip tests of Theano Ops with pycuda code.")

if cuda_ndarray.cuda_available is False:  # noqa
    from nose.plugins.skip import SkipTest
    raise SkipTest('Optional theano package cuda disabled')

import pycuda
import pycuda.driver as drv
import pycuda.gpuarray


def test_pycuda_only():
    """Run pycuda only example to test that pycuda works."""
    from pycuda.compiler import SourceModule
    mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")

    multiply_them = mod.get_function("multiply_them")

    # Test with pycuda in/out of numpy.ndarray
    a = np.random.randn(100).astype(np.float32)
    b = np.random.randn(100).astype(np.float32)
    dest = np.zeros_like(a)
    multiply_them(
        drv.Out(dest), drv.In(a), drv.In(b),
        block=(400, 1, 1), grid=(1, 1))
    assert (dest == a * b).all()


def test_pycuda_theano():
    """Simple example with pycuda function and Theano CudaNdarray object."""
    from pycuda.compiler import SourceModule
    mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")

    multiply_them = mod.get_function("multiply_them")

    a = np.random.randn(100).astype(np.float32)
    b = np.random.randn(100).astype(np.float32)

    # Test with Theano object
    ga = cuda_ndarray.CudaNdarray(a)
    gb = cuda_ndarray.CudaNdarray(b)
    dest = cuda_ndarray.CudaNdarray.zeros(a.shape)
    multiply_them(dest, ga, gb,
                  block=(400, 1, 1), grid=(1, 1))
    assert (np.asarray(dest) == a * b).all()


def test_pycuda_memory_to_theano():
    # Test that we can use the GpuArray memory space in pycuda in a CudaNdarray
    y = pycuda.gpuarray.zeros((3, 4, 5), 'float32')
    print(sys.getrefcount(y))
    # This increase the ref count with never pycuda. Do pycuda also
    # cache ndarray?
    # print y.get()
    initial_refcount = sys.getrefcount(y)
    print("gpuarray ref count before creating a CudaNdarray", end=' ')
    print(sys.getrefcount(y))
    assert sys.getrefcount(y) == initial_refcount
    rand = np.random.randn(*y.shape).astype(np.float32)
    cuda_rand = cuda_ndarray.CudaNdarray(rand)

    strides = [1]
    for i in y.shape[::-1][:-1]:
        strides.append(strides[-1] * i)
    strides = tuple(strides[::-1])
    print('strides', strides)
    assert cuda_rand._strides == strides, (cuda_rand._strides, strides)

    # in pycuda trunk, y.ptr also works, which is a little cleaner
    y_ptr = int(y.gpudata)
    z = cuda_ndarray.from_gpu_pointer(y_ptr, y.shape, strides, y)
    print("gpuarray ref count after creating a CudaNdarray", sys.getrefcount(y))
    assert sys.getrefcount(y) == initial_refcount + 1
    assert (np.asarray(z) == 0).all()
    assert z.base is y

    # Test that we can take a view from this cuda view on pycuda memory
    zz = z.view()
    assert sys.getrefcount(y) == initial_refcount + 2
    assert zz.base is y
    del zz
    assert sys.getrefcount(y) == initial_refcount + 1

    cuda_ones = cuda_ndarray.CudaNdarray(np.asarray([[[1]]],
                                                    dtype='float32'))
    z += cuda_ones
    assert (np.asarray(z) == np.ones(y.shape)).all()
    assert (np.asarray(z) == 1).all()

    assert cuda_rand.shape == z.shape
    assert cuda_rand._strides == z._strides, (cuda_rand._strides, z._strides)
    assert (np.asarray(cuda_rand) == rand).all()
    z += cuda_rand
    assert (np.asarray(z) == (rand + 1)).all()

    # Check that the ref count to the gpuarray is right.
    del z
    print("gpuarray ref count after deleting the CudaNdarray", end=' ')
    print(sys.getrefcount(y))
    assert sys.getrefcount(y) == initial_refcount
