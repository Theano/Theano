from __future__ import absolute_import, print_function, division

import numpy as np
import theano
from theano import Op
import theano.tensor as T
from theano.gradient import DisconnectedType

from theano.gpuarray import (basic_ops, GpuArrayType)

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
    import scikits.cuda
    from scikits.cuda import fft
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

    def make_node(self, inp, s):
        if not scikits_cuda_available:
            raise RuntimeError("scikits.cuda is needed for CuFFTOp")

        if not pygpu_available:
            raise RuntimeError("pygpu is needed for CuFFTOp")

        if not pycuda_available:
            raise RuntimeError("pycuda is needed for CuFFTOp")

        inp = basic_ops.gpu_contiguous(
            basic_ops.as_gpuarray_variable(inp,
                                           basic_ops.infer_context_name(inp)))
        s = T.as_tensor_variable(s)

        assert inp.dtype == "float32"
        assert s.ndim == 1
        assert 'int' in s.dtype

        return theano.Apply(self, [inp, s], [self.output_type(inp)()])

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
            s = inputs[1][0]

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
            # so it is complex64, but as it turns out scikits.cuda.fft
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
        gout = T.set_subtensor(gout[idx], gout[idx]*0.5) 
        return [cuirfft_op(gout, s), DisconnectedType()()]
    
    def connection_pattern(self, node):
        return [[True],[False]]

curfft_op = CuRFFTOp()


class CuIRFFTOp(Op):

    __props__ = ()

    def output_type(self, inp):
        # add one extra dim for real/imag
        return GpuArrayType(inp.dtype,
                            broadcastable=[False] * (inp.type.ndim - 1),
                            context_name=inp.type.context_name)

    def make_node(self, inp, s):
        if not scikits_cuda_available:
            raise RuntimeError("scikits.cuda is needed for CuIFFTOp")

        if not pygpu_available:
            raise RuntimeError("pygpu is needed for CuIFFTOp")

        if not pycuda_available:
            raise RuntimeError("pycuda is needed for CuIFFTOp")

        inp = basic_ops.gpu_contiguous(
            basic_ops.as_gpuarray_variable(inp,
                                           basic_ops.infer_context_name(inp)))
        s = T.as_tensor_variable(s)

        assert inp.dtype == "float32"
        assert s.ndim == 1

        return theano.Apply(self, [inp, s], [self.output_type(inp)()])

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
            s = inputs[1][0]
            
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
            # but will be interpreted by scikits.cuda as a complex64
            # array instead.
            output_pycuda = z[0]

            with input_pycuda.context:
                # only initialise plan if necessary
                if plan[0] is None or plan_input_shape[0] != input_shape:
                    plan_input_shape[0] = input_shape
                    plan[0] = fft.Plan(s,np.complex64, np.float32,
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
        gf = T.set_subtensor(gf[idx], gf[idx]*2)
        return [gf, DisconnectedType()()]

    def connection_pattern(self, node):
        return [[True],[False]]

cuirfft_op = CuIRFFTOp()

def curfft(inp, norm=None):
    """
    Performs the fast Fourier transform of a real-valued output on the GPU 
    through the gpuarray backend.

    The input must be a real-valued float32 variable of dimensions (m, ..., n).
    It performs FFTs of size (..., n) on m batches.

    The output is a GpuArray of dimensions (m, ..., n//2+1, 2). The second to 
    last dimension of the output contains the n//2+1 non-trivial elements of
    the real-valued FFTs. The real and imaginary parts are stored as two
    float32 arrays, emulating complex64. Since theano does not support complex
    number operations, care must be taken to manually implement operators such
    as multiplication.

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
    if cond_norm is None or cond_norm == "no_norm":
        scaling = 1
    elif cond_norm == "ortho":
        scaling = T.sqrt(s.prod().astype('float32'))
    
    return curfft_op(inp, s) / scaling                                  

def cuirfft(inp, norm=None, is_odd=False):
    """
    Performs the real-valued output inverse Fourier Transform using the
    gpuarray backend.

    The input is a variable of dimensions (m, ..., n//2+1, 2) with
    type float32 representing the non-trivial elements of m
    real-valued Fourier transforms of initial size (..., n). The real and
    imaginary parts are stored as two float32 arrays, emulating complex64
    given that Theano does not support complex numbers.

    The output is a real-valued float32 variable of dimensions (m, ..., n)
    giving the m inverse FFTs. 

    Parameters
    ----------
    inp
        Array of float32 of size (m, ..., n//2+1, 2), containing m inputs 
        with n/2+1 non-trivial elements on the last dimension and real
        and imaginary parts stored as separate arrays.
    norm : {None, 'ortho', 'no_norm'}
        Normalization of transform. Following numpy, default *None* normalizes
        only the inverse transform by n, 'ortho' yields the unitary transform
        (:math:`1/\sqrt n` forward and inverse). In addition, 'no_norm' leaves
        the transform unnormalized.
        
    """

    s = inp.shape[1:-1]
    if is_odd:
        s = T.set_subtensor(s[-1], (s[-1] - 1) * 2 + 1)
    else:
        s = T.set_subtensor(s[-1], (s[-1] - 1) * 2)
            
    cond_norm = _unitary(norm)
    if cond_norm is None:
        scaling = s.prod().astype('float32')
    if cond_norm == "ortho":
        scaling = T.sqrt(s.prod().astype('float32'))
    if cond_norm == "no_norm":
        scaling = 1

    return cuirfft_op(inp, s) / scaling

def _unitary(norm):
    if norm not in (None, "ortho", "no_norm"):
        raise ValueError("Invalid value %s for norm, must be None, 'ortho' or "
                         "'no norm'" % norm)
    return norm
