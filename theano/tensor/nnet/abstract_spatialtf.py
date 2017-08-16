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
import numpy as np


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

    def debug_perform(self, node, inputs, output_storage):
        inp, theta, out_height, out_width = inputs
        out = output_storage[0]

        num_batch, num_channels, height, width = inp.shape

        # Apply affine transformation to sampling grid
        grid = self._sampling_grid(out_height, out_width)
        transformed_grid = np.dot(theta, grid)
        x_s = transformed_grid[:, 0]
        y_s = transformed_grid[:, 1]
        x_s_flat = x_s.flatten()
        y_s_flat = y_s.flatten()

        inputs_dim = np.transpose(inp, axes=(0, 2, 3, 1))
        inputs_transformed = self._interpolate(inputs_dim, x_s_flat, y_s_flat,
                                               out_height, out_width,
                                               node.op.border_mode)
        output = np.reshape(inputs_transformed, (num_batch, out_height, out_width, num_channels))
        out[0] = np.transpose(output, axes=(0, 3, 1, 2)).astype(inp.dtype)

    def _sampling_grid(self, height, width):
        # Create sampling grid
        x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
                               np.linspace(-1, 1, height))
        ones = np.ones(np.prod(x_t.shape))
        grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        return grid

    def _interpolate(self, im, x, y, out_height, out_width, border_mode):
        num_batch, height, width, channels = im.shape
        height_f = float(height)
        width_f = float(width)

        # Scale coordinates from [-1, 1] to [0, dimension - 1], where dimension
        # can be the width or height
        x = (x + 1) / 2 * (width_f - 1)
        y = (y + 1) / 2 * (height_f - 1)

        # Obtain indices of the 2x2 pixel neighborhood surrounding the coordinates;
        # we need those in floatX for interpolation and in int64 for indexing.
        x0_f = np.floor(x)
        y0_f = np.floor(y)
        x1_f = x0_f + 1
        y1_f = y0_f + 1

        # for indexing, we need to take care of the border mode for outside pixels
        if border_mode == 'nearest':
            x0 = np.clip(x0_f, 0, width_f - 1)
            x1 = np.clip(x1_f, 0, width_f - 1)
            y0 = np.clip(y0_f, 0, height_f - 1)
            y1 = np.clip(y1_f, 0, height_f - 1)
        elif border_mode == 'mirror':
            w = 2 * (width_f - 1)
            x0 = np.minimum(x0_f % w, -x0_f % w)
            x1 = np.minimum(x1_f % w, -x1_f % w)
            h = 2 * (height_f - 1)
            y0 = np.minimum(y0_f % h, -y0_f % h)
            y1 = np.minimum(y1_f % h, -y1_f % h)
        elif border_mode == 'wrap':
            x0 = np.mod(x0_f, width_f)
            x1 = np.mod(x1_f, width_f)
            y0 = np.mod(y0_f, height_f)
            y1 = np.mod(y1_f, height_f)
        else:
            raise ValueError("border_mode must be one of "
                             "'nearest', 'mirror', 'wrap'")
        x0, x1, y0, y1 = np.asarray([x0, x1, y0, y1]).astype('int64')
        # The input is [num_batch, height, width, channels]. We do the lookup in
        # the flattened input, i.e [num_batch*height*width, channels]. We need
        # to offset all indices to match the flat version
        dim2 = width
        dim1 = width * height
        base = np.repeat(np.arange(num_batch, dtype='int64') * dim1,
                         out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels for all samples
        im_flat = im.reshape((-1, channels))
        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]

        # Calculate interpolated values
        wa = ((x1_f - x) * (y1_f - y))[:, np.newaxis]
        wb = ((x1_f - x) * (y - y0_f))[:, np.newaxis]
        wc = ((x - x0_f) * (y1_f - y))[:, np.newaxis]
        wd = ((x - x0_f) * (y - y0_f))[:, np.newaxis]
        output = np.sum([wa * Ia, wb * Ib, wc * Ic, wd * Id], axis=0)
        return output


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
