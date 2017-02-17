from __future__ import absolute_import, print_function, division
import numpy as np
from theano import gof
import theano.tensor as T
from theano.gradient import DisconnectedType


class RFFTOp(gof.Op):

    __props__ = ()

    def output_type(self, inp):
        # add extra dim for real/imag
        return T.TensorType(inp.dtype,
                            broadcastable=[False] * (inp.type.ndim + 1))

    def make_node(self, a, s=None):
        a = T.as_tensor_variable(a)
        if a.ndim < 2:
            raise TypeError('%s: input must have dimension > 2, with first dimension batches' %
                            self.__class__.__name__)

        if s is None:
            s = a.shape[1:]
            s = T.as_tensor_variable(s)
        else:
            s = T.as_tensor_variable(s)
            if s.dtype not in T.integer_dtypes:
                raise TypeError('%s: length of the transformed axis must be'
                                ' of type integer' % self.__class__.__name__)
        return gof.Apply(self, [a, s], [self.output_type(a)()])

    def perform(self, node, inputs, output_storage):
        a = inputs[0]
        s = inputs[1]

        A = np.fft.rfftn(a, s=tuple(s))
        # Format output with two extra dimensions for real and imaginary
        # parts.
        out = np.zeros(A.shape + (2,), dtype=a.dtype)
        out[..., 0], out[..., 1] = np.real(A), np.imag(A)
        output_storage[0][0] = out

    def grad(self, inputs, output_grads):
        gout, = output_grads
        s = inputs[1]
        # Divide the last dimension of the output gradients by 2, they are
        # double-counted by the real-IFFT due to symmetry, except the first
        # and last elements (for even transforms) which are unique.
        idx = [slice(None)] * (gout.ndim - 2) \
            + [slice(1, (s[-1] // 2) + (s[-1] % 2))] + [slice(None)]
        gout = T.set_subtensor(gout[idx], gout[idx] * 0.5)
        return [irfft_op(gout, s), DisconnectedType()()]

    def connection_pattern(self, node):
        # Specificy that shape input parameter has no connection to graph and gradients.
        return [[True], [False]]

rfft_op = RFFTOp()


class IRFFTOp(gof.Op):

    __props__ = ()

    def output_type(self, inp):
        # remove extra dim for real/imag
        return T.TensorType(inp.dtype,
                            broadcastable=[False] * (inp.type.ndim - 1))

    def make_node(self, a, s=None):
        a = T.as_tensor_variable(a)
        if a.ndim < 3:
            raise TypeError('%s: input must have dimension >= 3,  with ' %
                            self.__class__.__name__ +
                            'first dimension batches and last real/imag parts')

        if s is None:
            s = a.shape[1:-1]
            s = T.set_subtensor(s[-1], (s[-1] - 1) * 2)
            s = T.as_tensor_variable(s)
        else:
            s = T.as_tensor_variable(s)
            if s.dtype not in T.integer_dtypes:
                raise TypeError('%s: length of the transformed axis must be'
                                ' of type integer' % self.__class__.__name__)
        return gof.Apply(self, [a, s], [self.output_type(a)()])

    def perform(self, node, inputs, output_storage):
        a = inputs[0]
        s = inputs[1]

        # Reconstruct complex array from two float dimensions
        inp = a[..., 0] + 1j * a[..., 1]
        out = np.fft.irfftn(inp, s=tuple(s))
        # Remove numpy's default normalization
        # Cast to input type (numpy outputs float64 by default)
        output_storage[0][0] = (out * s.prod()).astype(a.dtype)

    def grad(self, inputs, output_grads):
        gout, = output_grads
        s = inputs[1]
        gf = rfft_op(gout, s)
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

irfft_op = IRFFTOp()


def rfft(inp, norm=None):
    """
    Performs the fast Fourier transform of a real-valued input.

    The input must be a real-valued variable of dimensions (m, ..., n).
    It performs FFTs of size (..., n) on m batches.

    The output is a tensor of dimensions (m, ..., n//2+1, 2). The second to
    last dimension of the output contains the n//2+1 non-trivial elements of
    the real-valued FFTs. The real and imaginary parts are stored as a pair of
    float arrays.

    Parameters
    ----------
    inp
        Array of floats of size (m, ..., n), containing m inputs of
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
        scaling = T.sqrt(s.prod().astype(inp.dtype))

    return rfft_op(inp, s) / scaling


def irfft(inp, norm=None, is_odd=False):
    """
    Performs the inverse fast Fourier Transform with real-valued output.

    The input is a variable of dimensions (m, ..., n//2+1, 2)
    representing the non-trivial elements of m real-valued Fourier transforms
    of initial size (..., n). The real and imaginary parts are stored as a
    pair of float arrays.

    The output is a real-valued variable of dimensions (m, ..., n)
    giving the m inverse FFTs.

    Parameters
    ----------
    inp
        Array of size (m, ..., n//2+1, 2), containing m inputs
        with n//2+1 non-trivial elements on the last dimension and real
        and imaginary parts stored as separate real arrays.
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
    # Numpy's default normalization is 1/N on the inverse transform.
    if cond_norm is None:
        scaling = s.prod().astype(inp.dtype)
    elif cond_norm == "ortho":
        scaling = T.sqrt(s.prod().astype(inp.dtype))

    return irfft_op(inp, s) / scaling


def _unitary(norm):
    if norm not in (None, "ortho", "no_norm"):
        raise ValueError("Invalid value %s for norm, must be None, 'ortho' or "
                         "'no norm'" % norm)
    return norm
