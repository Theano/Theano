from __future__ import absolute_import, print_function, division

import numpy as np
import theano
from theano import Op

from theano.gpuarray import (basic_ops, GpuArrayType)

try:
    import pygpu
    pygpu_available = True
except ImportError:
    pygpu_available = False

import pycuda.driver

try:
    import scikits.cuda
    from scikits.cuda import fft
    scikits_cuda_available = True
except (ImportError, Exception):
    scikits_cuda_available = False


class CuFFTOp(Op):
    """
    Performs a fast Fourier transform on the GPU using the scikits CUDA FFT
    through the gpuarray backend.

    The input must be a float32 variable of dimensions (m, n). It
    performs m 1-D FFTs of size n each.

    The output is a GpuArray of dimensions (m, n/2+1, 2). The output contains
    the n/2+1 non-trivial elements of the m real-valued FFTs. The real
    and imaginary parts stored as two float32 arrays, emulating complex64.
    Since theano does not support complex number operations, care must be
    taken to manually implement operators such as multiplication.

    The module provides the convenience function cufft(input).
    """

    __props__ = ()

    def output_type(self, inp):
        # add one extra dim for real/imag
        return GpuArrayType(inp.dtype,
                            broadcastable=[False] * (inp.type.ndim + 1),
                            context_name=inp.type.context_name)

    def make_node(self, inp):
        if not scikits_cuda_available:
            raise RuntimeError("scikits.cuda is needed for CuFFTOp")

        if not pygpu_available:
            raise RuntimeError("pygpu is needed for CuFFTOp")

        inp = basic_ops.gpu_contiguous(
            basic_ops.as_gpuarray_variable(inp,
                                           basic_ops.infer_context_name(inp)))

        assert inp.dtype == "float32"

        return theano.Apply(self, [inp], [self.output_type(inp)()])

    def make_thunk(self, node, storage_map, _, _2):

        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        # Initiliaze cuda context to the input's.
        with node.inputs[0].type.context:
            scikits.cuda.misc.init()

        plan_input_shape = [None]
        plan = [None]

        def thunk():
            input_shape = inputs[0][0].shape

            # construct output shape
            output_shape = list(input_shape)
            # DFT of real input is symmetric, no need to store
            # redundant coefficients
            output_shape[-1] = output_shape[-1] // 2 + 1
            # extra dimension with length 2 for real/imag
            output_shape += [2]
            output_shape = tuple(output_shape)

            z = outputs[0]

            # only allocate if there is no previous allocation of the
            # right size.
            if z[0] is None or z[0].shape != output_shape:
                z[0] = pygpu.zeros(output_shape, context=inputs[0][0].context,
                                   dtype='float32')

            input_pycuda = inputs[0][0]
            # I thought we'd need to change the type on output_pycuda
            # so it is complex64, but as it turns out scikits.cuda.fft
            # doesn't really care either way and treats the array as
            # if it is complex64 anyway.
            output_pycuda = z[0]

            with input_pycuda.context:
                # only initialise plan if necessary
                if plan[0] is None or plan_input_shape[0] != input_shape:
                    plan_input_shape[0] = input_shape
                    plan[0] = fft.Plan(input_shape[1:], np.float32, np.complex64,
                                       batch=input_shape[0])
                # Sync GPU variables before computation
                input_pycuda.sync()
                output_pycuda.sync()

                fft.fft(input_pycuda, output_pycuda, plan[0])
                # Sync results to ensure output contains completed computation
                pycuda.driver.Context.synchronize()

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk

cufft = CuFFTOp()
"""
Convenience function for CuFFTOp.

Parameters
----------
input
    Array of float32 of size (m, n), containing m inputs of length n.
"""


class CuIFFTOp(Op):
    """
    Performs an inverse fast Fourier transform on the GPU using the
    scikits CUDA FFT through the gpuarray backend.

    The input is a variable of dimensions (m, n/2+1, 2) with
    type float32 representing the n/2+1 non-trivial elements of m
    real-valued Fourier transforms of initial size n. The real and imaginary
    parts are stored as two float32 arrays, emulating complex64 given that
    Theano does not support complex numbers.

    The output is a float32 variable of dimensions (m, n) giving the m
    inverse FFTs. *The output is NOT normalized*. You can manualy divide
    by the size of the output array to normalize.

    The module provides the convenience function cuifft(input).
    """

    __props__ = ()

    def output_type(self, inp):
        # add one extra dim for real/imag
        return GpuArrayType(inp.dtype,
                            broadcastable=[False] * (inp.type.ndim - 1),
                            context_name=inp.type.context_name)

    def make_node(self, inp):
        if not scikits_cuda_available:
            raise RuntimeError("scikits.cuda is needed for CuFFTOp")

        if not pygpu_available:
            raise RuntimeError("pygpu is needed for CuFFTOp")
        # inp = as_gpuarray_variable(inp)
        inp = basic_ops.gpu_contiguous(
            basic_ops.as_gpuarray_variable(inp,
                                           basic_ops.infer_context_name(inp)))

        assert inp.dtype == "float32"

        return theano.Apply(self, [inp], [self.output_type(inp)()])

    def make_thunk(self, node, storage_map, _, _2):

        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        # Initiliaze cuda context to the input's.
        with node.inputs[0].type.context:
            scikits.cuda.misc.init()

        plan_input_shape = [None]
        plan = [None]

        def thunk():
            input_shape = inputs[0][0].shape

            # construct output shape
            # chop off the extra length-2 dimension for real/imag
            output_shape = list(input_shape[:-1])
            # restore full signal length
            output_shape[-1] = (output_shape[-1] - 1) * 2
            output_shape = tuple(output_shape)

            z = outputs[0]

            # only allocate if there is no previous allocation of the
            # right size.
            if z[0] is None or z[0].shape != output_shape:
                z[0] = pygpu.zeros(output_shape, context=inputs[0][0].context,
                                   dtype='float32')

            input_pycuda = inputs[0][0]
            # input_pycuda is a float32 array with an extra dimension,
            # but will be interpreted by scikits.cuda as a complex64
            # array instead.
            output_pycuda = z[0]

            with input_pycuda.context:
                # only initialise plan if necessary
                if plan[0] is None or plan_input_shape[0] != input_shape:
                    plan_input_shape[0] = input_shape
                    plan[0] = fft.Plan(output_shape[1:],
                                       np.complex64, np.float32,
                                       batch=output_shape[0])
                # Sync GPU variables before computation
                input_pycuda.sync()
                output_pycuda.sync()

                fft.ifft(input_pycuda, output_pycuda, plan[0])
                # Sync results to ensure output contains completed computation
                pycuda.driver.Context.synchronize()

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk

cuifft = CuIFFTOp()
"""
Convenience function for CuIFFTOp.

Parameters
----------
input
    Array of float32 of size (m, n/2+1, 2), containing m inputs with n/2+1
    non-trivial elements and real and imaginary parts stored as separate arrays.
"""
