import string

import numpy as np
import theano
import theano.tensor as T

from theano.sandbox.cuda import cuda_available, GpuOp

if cuda_available:
    from theano.sandbox.cuda import (basic_ops, CudaNdarrayType,
                                     CudaNdarray)
import theano.misc.pycuda_init
from theano.misc.pycuda_init import pycuda_available
if pycuda_available:
    import pycuda.gpuarray

try:
    import scikits.cuda
    from scikits.cuda import fft, cublas
    scikits.cuda.misc.init()
    scikits_cuda_available = True
except ImportError:
    scikits_cuda_available = False


def to_complex_gpuarray(x, copyif=False):
    """
    adapted version of theano.misc.pycuda_utils.to_gpuarray that takes
    an array with an extra trailing dimension of length 2 for
    real/imaginary parts, and turns it into a complex64 PyCUDA
    GPUArray.
    """
    if not isinstance(x, CudaNdarray):
        raise ValueError("We can transfer only CudaNdarray "
                         "to pycuda.gpuarray.GPUArray")
    else:
        # Check if trailing dimension has length 2
        assert x.shape[-1] == 2

        # check if dtype is float32
        assert x.dtype == 'float32'

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
                raise ValueError("We were asked to not copy memory, "
                                 "but the memory is not c contiguous.")

        # Now x is always c contiguous
        px = pycuda.gpuarray.GPUArray(x.shape[:-1], np.complex64, base=x,
                                      gpudata=x.gpudata)
        return px


def bptrs(a):
    """
    Pointer array when input represents a batch of matrices.

    taken from scikits.cuda tests/test_cublas.py
    """
    return pycuda.gpuarray.arange(a.ptr, a.ptr + a.shape[0] * a.strides[0],
                                  a.strides[0], dtype=cublas.ctypes.c_void_p)


def sc_complex_dot_batched(bx_gpu, by_gpu, bc_gpu, transa='N', transb='N',
                           handle=None):
    """
    uses cublasCgemmBatched to compute a bunch of complex dot products
    in parallel
    """
    if handle is None:
        handle = scikits.cuda.misc._global_cublas_handle

    assert len(bx_gpu.shape) == 3
    assert len(by_gpu.shape) == 3
    assert len(bc_gpu.shape) == 3
    assert bx_gpu.dtype == np.complex64
    assert by_gpu.dtype == np.complex64
    assert bc_gpu.dtype == np.complex64

    # Get the shapes of the arguments
    bx_shape = bx_gpu.shape
    by_shape = by_gpu.shape

    # Perform matrix multiplication for 2D arrays:
    alpha = np.complex64(1.0)
    beta = np.complex64(0.0)

    transa = string.lower(transa)
    transb = string.lower(transb)

    if transb in ['t', 'c']:
        N, m, k = by_shape
    elif transb in ['n']:
        N, k, m = by_shape
    else:
        raise ValueError('invalid value for transb')

    if transa in ['t', 'c']:
        N2, l, n = bx_shape
    elif transa in ['n']:
        N2, n, l = bx_shape
    else:
        raise ValueError('invalid value for transa')

    if l != k:
        raise ValueError('objects are not aligned')

    if N != N2:
        raise ValueError('batch sizes are not the same')

    if transb == 'n':
        lda = max(1, m)
    else:
        lda = max(1, k)

    if transa == 'n':
        ldb = max(1, k)
    else:
        ldb = max(1, n)

    ldc = max(1, m)

    # construct pointer arrays needed for cublasCgemmBatched
    bx_arr = bptrs(bx_gpu)
    by_arr = bptrs(by_gpu)
    bc_arr = bptrs(bc_gpu)

    cublas.cublasCgemmBatched(handle, transb, transa, m, n, k, alpha,
                              by_arr.gpudata, lda, bx_arr.gpudata, ldb,
                              beta, bc_arr.gpudata, ldc, N)


class FFTConv2D(GpuOp):
    def __init__(self, border_mode='valid', autopad=False):
        if border_mode not in ('valid', 'full'):
            raise ValueError('invalid border_mode')
        self.border_mode = border_mode
        self.autopad = autopad

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.autopad == other.autopad and
                self.border_mode == other.border_mode)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.autopad) ^ hash(self.border_mode)

    def make_node(self, input, filters):
        if not scikits_cuda_available:
            raise RuntimeError("scikits.cuda >= 0.5.0 is not available "
                               "but is required for this op")
        if not pycuda_available:
            raise RuntimeError("PyCUDA is not available "
                               "but is required for this op")
        _input = basic_ops.as_cuda_ndarray_variable(input)
        _filters = basic_ops.as_cuda_ndarray_variable(filters)

        if _input.ndim != 4:
            raise TypeError('ConvFFT2D requires input to be a 4D tensor, '
                            'recieved "%s" (%i dims)' %
                            (input, _input.ndim))
        if _filters.ndim != 4:
            raise TypeError('ConvFFT2D requires filters to be a 4D tensor, '
                            'recieved "%s" (%i dims)' %
                            (filters, _filters.ndim))

        out_type = CudaNdarrayType(broadcastable=[_input.broadcastable[0],
                                                  _filters.broadcastable[0],
                                                  False, False])

        return theano.Apply(self, [_input, _filters], [out_type()])

    def _dimshuffle(self, v, newdims):
        assert v.ndim == len(set(newdims))
        r = v.view()
        for i, d in enumerate(newdims):
            r._set_shape_i(i, v.shape[d])
            r._set_stride(i, v.strides[d])
        return r

    def _bcd(self, bx, by):
        input_shape_x = bx.shape
        input_shape_y = by.shape
        output_shape = (input_shape_x[0], input_shape_x[1],
                        input_shape_y[2], 2)

        bz = CudaNdarray.zeros(output_shape)

        sc_complex_dot_batched(to_complex_gpuarray(bx, copyif=True),
                               to_complex_gpuarray(by, copyif=True),
                               to_complex_gpuarray(bz))

        return bz


    def _do_fft(self, input, plans):
        from theano.misc.pycuda_utils import to_gpuarray
        input_shape = input.shape
        output_shape = (input_shape[0], input_shape[1], input_shape[2],
                        input_shape[3] // 2 + 1, 2)
        output = CudaNdarray.zeros(output_shape)

        if input_shape in plans:
            plan = plans[input_shape]
        else:
            plan = fft.Plan(input_shape[2:], np.float32, np.complex64,
                            batch=input_shape[0] * input_shape[1])
            plans[input_shape] = plan

        fft.fft(to_gpuarray(input), to_gpuarray(output), plan)
        return output

    def _do_ifft(self, input, plans):
        from theano.misc.pycuda_utils import to_gpuarray
        input_shape = input.shape
        output_shape = (input_shape[0], input_shape[1], input_shape[2],
                        (input_shape[3] - 1) * 2)
        if (input_shape, False) in plans:
            plan = plans[(input_shape, False)]
        else:
            plan = fft.Plan(output_shape[2:], np.complex64, np.float32,
                            batch=input_shape[0] * input_shape[1])
            plans[(input_shape, False)] = plan
        output = CudaNdarray.zeros(output_shape)

        fft.ifft(to_gpuarray(input), to_gpuarray(output), plan)
        return output

    def _mult_reduce(self, input_fft_v, filters_fft_v, input_shape,
                     filter_shape):
        b, ic, i0, i1, _ = input_shape
        oc = filter_shape[0]

        input_r = input_fft_v.reshape((b, ic, i0 * i1, 2))
        del input_fft_v
        filters_r = filters_fft_v.reshape((oc, ic, i0 * i1, 2))
        del filters_fft_v

        input_s = self._dimshuffle(input_r, (2, 0, 1, 3))
        del input_r
        filters_s = self._dimshuffle(filters_r, (2, 1, 0, 3))
        del filters_r

        output_s = self._bcd(input_s, filters_s)
        del input_s
        del filters_s

        output_r = self._dimshuffle(output_s, (1, 2, 0, 3))
        del output_s

        output = output_r.reshape((b, oc, i0, i1, 2))
        del output_r

        return output

    def perform(self, node, inp, out):
        input, filters = inp
        # we can't reuse the output
        out[0][0] = None

        b, ic, i0, i1 = input.shape
        oc, _, f0, f1 = filters.shape
        if self.border_mode == 'valid':
            o0 = i0
            o1 = i1
            if self.autopad and o1 % 2 == 1:
                input_padded = CudaNdarray.zeros((b, ic, o0, o1))
                input_padded[:, :, :i0, :i1] = input
            else:
                input_padded = input
        elif self.border_mode == 'full':
            o0 = i0 + 2 * (f0 - 1)
            o1 = i1 + 2 * (f1 - 1)
            if self.autopad and o1 % 2 == 1:
                o1 += 1

            # We line up the filters and the images in a way such that
            # the images intersect with the filters on one item. The
            # top-left item of the images is the bottom-right item of
            # the filters when we do the layout here.
            input_padded = CudaNdarray.zeros((b, ic, o0, o1))
            input_padded[:, :, (f0 - 1):(f0 - 1 + i0), (f1 - 1):(f1 - 1 + i1)] = input

        if o1 % 2 == 1:
            raise RuntimeError("final width is not a multiple of 2. Use the "
                               "autopad argument to add padding as necessary "
                               "or fix the shapes in your code.")

        filters_padded = CudaNdarray.zeros((oc, ic, o0, o1))
        filters_padded[:, :, :f0, :f1] = filters

        assert o1 % 2 == 0

        if getattr(node, '_fft_plans', None) is None:
            node._fft_plans = dict()

        input_fft_v = self._do_fft(input_padded, node._fft_plans)
        del input_padded
        filters_fft_v = self._do_fft(filters_padded, node._fft_plans)
        del filters_padded

        output_fft_s = self._mult_reduce(input_fft_v, filters_fft_v,
                                         input_fft_v.shape,
                                         filters_fft_v.shape)
        del input_fft_v
        del filters_fft_v

        output_circ = self._do_ifft(output_fft_s, node._fft_plans)
        del output_fft_s

        if self.border_mode == 'valid':
            output = output_circ[:, :,
                (f0 - 1):(f0 - 1 + i0 - f0 + 1),
                (f1 - 1):(f1 - 1 + i1 - f1 + 1)]
        elif self.border_mode == 'full':
            output = output_circ[:, :,
                (f0 - 1):(f0 - 1 + i0 + f0 - 1),
                (f1 - 1):(f1 - 1 + i1 + f1 - 1)]
        del output_circ

        output /= np.asarray((o0 * o1), dtype='float32')
        out[0][0] = output
