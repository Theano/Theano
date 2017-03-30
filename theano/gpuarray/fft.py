from __future__ import absolute_import, print_function, division

import numpy as np
import theano
from theano import Op
import theano.tensor as T
from theano.gradient import DisconnectedType

from .basic_ops import (gpu_contiguous, as_gpuarray_variable,
                        infer_context_name)
from .type import GpuArrayType

import theano.tensor.fft
from .opt import register_opt, op_lifter, register_opt2

try:
    import pygpu
    pygpu_available = True
except ImportError:
    pygpu_available = False

try:
    import pycuda.driver
    pycuda_available = True
except ImportError:
    pycuda_available = False

try:
    import skcuda
    from skcuda import fft
    scikits_cuda_available = True
except (ImportError, Exception):
    scikits_cuda_available = False


class CuRFFTOp(Op):

    __props__ = ()

    def output_type(self, inp):
        # add one extra dim for real/imag
        return GpuArrayType(inp.dtype,
                            broadcastable=[False] * (inp.type.ndim + 1),
                            context_name=inp.type.context_name)

    def make_node(self, inp, s=None):
        # A shape parameter s can be provided as an input. For now this is used to
        # manage odd transform sizes.
        # Later this could be extended to handle padding and trunkation,
        # following numpy's interface. However, cuFFT expects array that match
        # the shape given to the plan, so padding will have to be done in the op.
        # The effect of padding on gradients has yet to be investigated.

        if not scikits_cuda_available:
            raise RuntimeError("skcuda is needed for CuFFTOp")

        if not pygpu_available:
            raise RuntimeError("pygpu is needed for CuFFTOp")

        if not pycuda_available:
            raise RuntimeError("pycuda is needed for CuFFTOp")

        inp = gpu_contiguous(as_gpuarray_variable(inp,
                                                  infer_context_name(inp)))

        # If no shape is provided as input, default to input data shape.
        if s is None:
            s = inp.shape[1:]
        s = T.as_tensor_variable(s)

        assert inp.dtype == "float32"
        assert s.ndim == 1
        assert s.dtype in theano.tensor.integer_dtypes

        return theano.Apply(self, [inp, s], [self.output_type(inp)()])

    def make_thunk(self, node, storage_map, _, _2, impl=None):

        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        # Initiliaze cuda context to the input's.
        with node.inputs[0].type.context:
            skcuda.misc.init()

        plan_input_shape = [None]
        plan = [None]

        def thunk():
            input_shape = inputs[0][0].shape
            s = inputs[1][0]

            # Since padding is not supported, assert s matches input shape.
            assert (input_shape[1:] == s).all()

            # construct output shape
            output_shape = [input_shape[0]] + list(s)
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
            # so it is complex64, but as it turns out skcuda.fft
            # doesn't really care either way and treats the array as
            # if it is complex64 anyway.
            output_pycuda = z[0]

            with input_pycuda.context:
                # only initialise plan if necessary
                if plan[0] is None or plan_input_shape[0] != input_shape:
                    plan_input_shape[0] = input_shape
                    plan[0] = fft.Plan(s, np.float32, np.complex64,
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

    def grad(self, inputs, output_grads):
        gout, = output_grads
        s = inputs[1]
        # Divide the last dimension of the output gradients by 2, they are
        # double-counted by the real-IFFT due to symmetry, except the first
        # and last elements (for even transforms) which are unique.
        idx = [slice(None)] * (gout.ndim - 2) \
            + [slice(1, (s[-1] // 2) + (s[-1] % 2))] + [slice(None)]
        gout = T.set_subtensor(gout[idx], gout[idx] * 0.5)
        return [cuirfft_op(gout, s), DisconnectedType()()]

    def connection_pattern(self, node):
        # Specificy that shape input parameter has no connection to graph and gradients.
        return [[True], [False]]

curfft_op = CuRFFTOp()


class CuIRFFTOp(Op):

    __props__ = ()

    def output_type(self, inp):
        # remove extra dim for real/imag
        return GpuArrayType(inp.dtype,
                            broadcastable=[False] * (inp.type.ndim - 1),
                            context_name=inp.type.context_name)

    def make_node(self, inp, s=None):
        # A shape parameter is expected as an input. For now this is used to
        # manage odd transform sizes.
        # Later this could be extended to handle padding and trunkation,
        # following numpy's interface. However, cuFFT expects array that match
        # the shape given to the plan, so padding will have to be done in the op.
        # The effect of padding on gradients has yet to be investigated.

        if not scikits_cuda_available:
            raise RuntimeError("skcuda is needed for CuIFFTOp")

        if not pygpu_available:
            raise RuntimeError("pygpu is needed for CuIFFTOp")

        if not pycuda_available:
            raise RuntimeError("pycuda is needed for CuIFFTOp")

        inp = gpu_contiguous(as_gpuarray_variable(inp,
                                                  infer_context_name(inp)))

        # If no shape is provided as input, calculate shape assuming even real transform.
        if s is None:
            s = inp.shape[1:-1]
            s = T.set_subtensor(s[-1], (s[-1] - 1) * 2)
        s = T.as_tensor_variable(s)

        assert inp.dtype == "float32"
        assert s.ndim == 1

        return theano.Apply(self, [inp, s], [self.output_type(inp)()])

    def make_thunk(self, node, storage_map, _, _2, impl=None):

        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        # Initiliaze cuda context to the input's.
        with node.inputs[0].type.context:
            skcuda.misc.init()

        plan_input_shape = [None]
        plan = [None]

        def thunk():
            input_shape = inputs[0][0].shape
            s = inputs[1][0]

            # Since padding is not supported, assert that last dimension corresponds to
            # input forward transform size.
            assert (input_shape[1:-2] == s[:-1]).all()
            assert ((input_shape[-2] - 1) * 2 + s[-1] % 2 == s[-1]).all()

            # construct output shape
            # chop off the extra length-2 dimension for real/imag
            output_shape = [input_shape[0]] + list(s)
            output_shape = tuple(output_shape)

            z = outputs[0]

            # only allocate if there is no previous allocation of the
            # right size.
            if z[0] is None or z[0].shape != output_shape:
                z[0] = pygpu.zeros(output_shape, context=inputs[0][0].context,
                                   dtype='float32')

            input_pycuda = inputs[0][0]
            # input_pycuda is a float32 array with an extra dimension,
            # but will be interpreted by skcuda as a complex64
            # array instead.
            output_pycuda = z[0]

            with input_pycuda.context:
                # only initialise plan if necessary
                if plan[0] is None or plan_input_shape[0] != input_shape:
                    plan_input_shape[0] = input_shape
                    plan[0] = fft.Plan(s, np.complex64, np.float32,
                                       batch=output_shape[0])

                # Sync GPU variables before computation
                input_pycuda.sync()
                output_pycuda.sync()

                fft.ifft(input_pycuda, output_pycuda, plan[0])
                # strangely enough, enabling rescaling here makes it run
                # very, very slowly, so do this rescaling manually
                # afterwards!

                # Sync results to ensure output contains completed computation
                pycuda.driver.Context.synchronize()

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk

    def grad(self, inputs, output_grads):
        gout, = output_grads
        s = inputs[1]
        gf = curfft_op(gout, s)
        # Multiply the last dimension of the gradient by 2, they represent
        # both positive and negative frequencies, except the first
        # and last elements (for even transforms) which are unique.
        idx = [slice(None)] * (gf.ndim - 2) \
            + [slice(1, (s[-1] // 2) + (s[-1] % 2))] + [slice(None)]
        gf = T.set_subtensor(gf[idx], gf[idx] * 2)
        return [gf, DisconnectedType()()]

    def connection_pattern(self, node):
        # Specificy that shape input parameter has no connection to graph and gradients.
        return [[True], [False]]

cuirfft_op = CuIRFFTOp()


def curfft(inp, norm=None):
    """
    Performs the fast Fourier transform of a real-valued input on the GPU.

    The input must be a real-valued float32 variable of dimensions (m, ..., n).
    It performs FFTs of size (..., n) on m batches.

    The output is a GpuArray of dimensions (m, ..., n//2+1, 2). The second to
    last dimension of the output contains the n//2+1 non-trivial elements of
    the real-valued FFTs. The real and imaginary parts are stored as a pair of
    float32 arrays.

    Parameters
    ----------
    inp
        Array of real-valued float32 of size (m, ..., n), containing m inputs of
        size (..., n).
    norm : {None, 'ortho', 'no_norm'}
        Normalization of transform. Following numpy, default *None* normalizes
        only the inverse transform by n, 'ortho' yields the unitary transform
        (:math:`1/\sqrt n` forward and inverse). In addition, 'no_norm' leaves
        the transform unnormalized.

    """

    s = inp.shape[1:]
    cond_norm = _unitary(norm)
    scaling = 1
    if cond_norm == "ortho":
        scaling = T.sqrt(s.prod().astype('float32'))

    return curfft_op(inp, s) / scaling


def cuirfft(inp, norm=None, is_odd=False):
    """
    Performs the inverse fast Fourier Transform with real-valued output on the GPU.

    The input is a variable of dimensions (m, ..., n//2+1, 2) with
    type float32 representing the non-trivial elements of m
    real-valued Fourier transforms of initial size (..., n). The real and
    imaginary parts are stored as a pair of float32 arrays.

    The output is a real-valued float32 variable of dimensions (m, ..., n)
    giving the m inverse FFTs.

    Parameters
    ----------
    inp
        Array of float32 of size (m, ..., n//2+1, 2), containing m inputs
        with n//2+1 non-trivial elements on the last dimension and real
        and imaginary parts stored as separate arrays.
    norm : {None, 'ortho', 'no_norm'}
        Normalization of transform. Following numpy, default *None* normalizes
        only the inverse transform by n, 'ortho' yields the unitary transform
        (:math:`1/\sqrt n` forward and inverse). In addition, 'no_norm' leaves
        the transform unnormalized.
    is_odd : {True, False}
        Set to True to get a real inverse transform output with an odd last dimension
        of length (N-1)*2 + 1 for an input last dimension of length N.

    """

    if is_odd not in (True, False):
        raise ValueError("Invalid value %s for id_odd, must be True or False" % is_odd)

    s = inp.shape[1:-1]
    if is_odd:
        s = T.set_subtensor(s[-1], (s[-1] - 1) * 2 + 1)
    else:
        s = T.set_subtensor(s[-1], (s[-1] - 1) * 2)

    cond_norm = _unitary(norm)
    scaling = 1
    if cond_norm is None:
        scaling = s.prod().astype('float32')
    elif cond_norm == "ortho":
        scaling = T.sqrt(s.prod().astype('float32'))

    return cuirfft_op(inp, s) / scaling


def _unitary(norm):
    if norm not in (None, "ortho", "no_norm"):
        raise ValueError("Invalid value %s for norm, must be None, 'ortho' or "
                         "'no norm'" % norm)
    return norm

if scikits_cuda_available:
    @register_opt('fast_compile')
    @op_lifter([theano.tensor.fft.RFFTOp])
    @register_opt2([theano.tensor.fft.RFFTOp], 'fast_compile')
    def local_gpua_curfft_op(op, ctx_name, inputs, outputs):
        return curfft_op

    @register_opt('fast_compile')
    @op_lifter([theano.tensor.fft.IRFFTOp])
    @register_opt2([theano.tensor.fft.IRFFTOp], 'fast_compile')
    def local_gpua_cuirfft_op(op, ctx_name, inputs, outputs):
        return cuirfft_op
