"""
Abstract spatial transformer interface
"""
from __future__ import absolute_import, print_function, division

import theano
from theano import Op
from theano.scalar import as_scalar
from theano.tensor import as_tensor_variable
from theano.gof import Apply
from theano.gradient import grad_not_implemented


class AbstractSpatialTransformerOp(Op):
    def __init__(self, border_mode):
        self.border_mode = border_mode

    def make_node(self, inp, theta, out_height, out_width):
        _inp = as_tensor_variable(inp)
        _theta = as_tensor_variable(theta)

        if _inp.ndim != 4:
            raise TypeError('SpatialTransformerOp (make_node) requires input to '
                            'be a 4D tensor; received "%s" (%i dims)' %
                            (inp, _inp.ndim))

        assert _inp.type.dtype in ('float16', 'float32', 'float64')

        if _theta.ndim != 3:
            raise TypeError('SpatialTransformerOp (make_node) requires theta to '
                            'be a 3D tensor; received "%s" (%i dims)' %
                            (theta, _theta.ndim))

        assert _theta.type.dtype in ('float16', 'float32', 'float64')

        out = theano.tensor.tensor(dtype=_inp.type.dtype,
                                   broadcastable=_inp.broadcastable)

        return Apply(self, [_inp, _theta, out_height, out_width], [out])

    def grad(self, inputs, grads):
        inp, theta, out_height, out_width = inputs

        grad_i = AbstractSpatialTransformerGradIOp()(*inputs)
        grad_t = AbstractSpatialTransformerGradTOp()(*inputs)

        return [grad_i, grad_t,
                grad_not_implemented(self, 2, out_height),
                grad_not_implemented(self, 3, out_width)]


class AbstractSpatialTransformerGradIOp(Op):
    def make_node(self, inp, theta, out_height, out_width):
        _inp = as_tensor_variable(inp)
        _theta = as_tensor_variable(theta)
        _out_height = as_scalar(out_height)
        _out_width = as_scalar(out_width)

        out = theano.tensor.tensor(dtype=_inp.type.dtype,
                                   broadcastable=_inp.broadcastable)

        return Apply(self, [_inp, _theta, _out_height, _out_width], [out])


class AbstractSpatialTransformerGradTOp(Op):
    def make_node(self, inp, theta, out_height, out_width):
        _inp = as_tensor_variable(inp)
        _theta = as_tensor_variable(theta)
        _out_height = as_scalar(out_height)
        _out_width = as_scalar(out_width)

        out = theano.tensor.tensor(dtype=_inp.type.dtype,
                                   broadcastable=_inp.broadcastable)

        return Apply(self, [_inp, _theta, _out_height, _out_width], [out])
