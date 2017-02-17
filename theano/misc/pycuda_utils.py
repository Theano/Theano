from __future__ import absolute_import, print_function, division
import pycuda.gpuarray

from theano.sandbox import cuda
if cuda.cuda_available is False:
    raise ImportError('Optional theano package cuda disabled')


def to_gpuarray(x, copyif=False):
    """ take a CudaNdarray and return a pycuda.gpuarray.GPUArray

    :type x: CudaNdarray
    :param x: The array to transform to pycuda.gpuarray.GPUArray.
    :type copyif: bool
    :param copyif: If False, raise an error if x is not c contiguous.
                   If it is c contiguous, we return a GPUArray that share
                   the same memory region as x.
                   If True, copy x if it is no c contiguous, so the return won't
                   shape the same memory region. If c contiguous, the return
                   will share the same memory region.

                   We need to do this as GPUArray don't fully support strided memory.

    :return type: pycuda.gpuarray.GPUArray
    """
    if not isinstance(x, cuda.CudaNdarray):
        raise ValueError("We can transfer only CudaNdarray to pycuda.gpuarray.GPUArray")
    else:
        # Check if it is c contiguous
        size = 1
        c_contiguous = True
        for i in range(x.ndim - 1, -1, -1):
            if x.shape[i] == 1:
                continue
            if x._strides[i] != size:
                c_contiguous = False
                break
            size *= x.shape[i]
        if not c_contiguous:
            if copyif:
                x = x.copy()
            else:
                raise ValueError("We were asked to not copy memory, but the memory is not c contiguous.")

        # Now x is always c contiguous
        px = pycuda.gpuarray.GPUArray(x.shape, x.dtype, base=x, gpudata=x.gpudata)
        return px


def to_cudandarray(x):
    """ take a pycuda.gpuarray.GPUArray and make a CudaNdarray that point to its memory

    :note: CudaNdarray support only float32, so only float32 GPUArray are accepted
    """
    if not isinstance(x, pycuda.gpuarray.GPUArray):
        raise ValueError("We can transfer only pycuda.gpuarray.GPUArray to CudaNdarray")
    elif x.dtype != "float32":
        raise ValueError("CudaNdarray support only float32")
    else:
        strides = [1]
        for i in x.shape[::-1][:-1]:
            strides.append(strides[-1] * i)
        strides = tuple(strides[::-1])
        ptr = int(x.gpudata)  # in pycuda trunk, y.ptr also works, which is a little cleaner
        z = cuda.from_gpu_pointer(ptr, x.shape, strides, x)
        return z
