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
import math


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
        grad_outputs = grads[0]

        grad_i, grad_g = AbstractSpatialTransformerGradIOp(self.border_mode)(inp, theta, grad_outputs, out_height, out_width)
        grad_t = AbstractSpatialTransformerGradTOp()(inp, theta, grad_g, out_height, out_width)

        return [grad_i, grad_t,
                grad_not_implemented(self, 2, out_height),
                grad_not_implemented(self, 3, out_width)]

    def perform(self, node, inputs, output_storage):
        inp, theta, out_height, out_width = inputs
        transformed_inputs = output_storage[0]

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
        transformed_inputs[0] = np.transpose(output, axes=(0, 3, 1, 2)).astype(inp.dtype)

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
    def __init__(self, border_mode):
        self.border_mode = border_mode

    def make_node(self, inp, theta, grad_outputs, out_height, out_width):
        _inp = as_tensor_variable(inp)
        _theta = as_tensor_variable(theta)
        _grad_outputs = as_tensor_variable(grad_outputs)
        _out_height = as_scalar(out_height)
        _out_width = as_scalar(out_width)

        grad_inp = theano.tensor.tensor(dtype=_inp.type.dtype,
                                        broadcastable=_inp.broadcastable)
        grad_grid = theano.tensor.tensor(dtype=_inp.type.dtype,
                                         broadcastable=_inp.broadcastable)

        return Apply(self, [_inp, _theta, _grad_outputs, _out_height, _out_width],
                     [grad_inp, grad_grid])

    def perform(self, node, inputs, output_storage):
        inp, theta, grad_outputs, out_height, out_width = inputs
        gradi_out = output_storage[0]
        gradg_out = output_storage[1]

        num_batch, num_channels, height, width = inp.shape

        grid = self.sampling_grid(out_height, out_width)
        transformed_grid = np.dot(theta, grid)
        x_s = transformed_grid[:, 0]
        y_s = transformed_grid[:, 1]
        x_s_flat = x_s.flatten()
        y_s_flat = y_s.flatten()

        inputs_transposed = np.transpose(inp, axes=(0, 2, 3, 1))
        grad_outputs_transposed = np.transpose(grad_outputs, axes=(0, 2, 3, 1))
        gradi, gradg = self.compute_grads(inputs_transposed, grad_outputs_transposed,
                                          x_s_flat, y_s_flat, out_height,
                                          out_width)

        grad_inputs = np.reshape(gradi, (num_batch, height, width, num_channels))
        grad_inputs = np.transpose(grad_inputs, axes=(0, 3, 1, 2)).astype(inp.dtype)

        grad_gradients = np.reshape(gradg, (num_batch, out_height, out_width, 2)).astype(inp.dtype)

        gradi_out[0] = grad_inputs
        gradg_out[0] = grad_gradients

    def sampling_grid(self, height, width):
        # Create sampling grid
        x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
                               np.linspace(-1, 1, height))
        ones = np.ones(np.prod(x_t.shape))
        grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        return grid

    def index_functor(self, num_batch, height, width, channels):
        def _clip(val, min_val, max_val):
            return max(min(val, max_val), min_val)

        def _index(b, h, w, c):
            _h = _clip(h, 0, height - 1)
            _w = _clip(w, 0, width - 1)
            idx = b * height * width * channels + _h * width * channels + _w * channels + c
            return idx
        return _index

    def compute_grads(self, _im, _grad_outputs, x, y, out_height, out_width):
        """
        Naive implementation of the algorithm to compute the gradients of the
        inputs and the grid from [1].

        [1] https://github.com/qassemoquab/stnbhwd/blob/master/generic/BilinearSamplerBHWD.c
        """
        num_batch, height, width, channels = _im.shape
        height_f = float(height)
        width_f = float(width)

        im = _im.flatten()
        grad_outputs = _grad_outputs.flatten()
        grad_inputs = np.zeros(_im.shape, dtype=_im.dtype).flatten()
        grad_grid = np.zeros((num_batch, out_height, out_width, 2), dtype=im.dtype).flatten()

        input_idx = self.index_functor(num_batch, height, width, channels)
        gout_idx = self.index_functor(num_batch, out_height, out_width, channels)
        grid_idx = self.index_functor(num_batch, out_height, out_width, 2)

        # Naive implementation of the algorithm to compute gradients of inputs and grid
        for b in range(num_batch):
            for h in range(out_height):
                for w in range(out_width):
                    xf = x[h * out_width + w]
                    yf = y[h * out_width + w]

                    # Scale x coordinate from [-1, 1] to [0, width-1]
                    xcoord = (xf + 1) * (width_f - 1) / 2
                    xInTopLeft = math.floor(xcoord)
                    xWeightTopLeft = float(1) - (xcoord - xInTopLeft)
                    # Scale y coordinate from [-1, 1] to [0, height-1]
                    ycoord = (yf + 1) * (height_f - 1) / 2
                    yInTopLeft = math.floor(ycoord)
                    yWeightTopLeft = float(1) - (ycoord - yInTopLeft)

                    topLeftDotProduct = float(0)
                    topRightDotProduct = float(0)
                    bottomLeftDotProduct = float(0)
                    bottomRightDotProduct = float(0)

                    inTopLeft = float(0)
                    inTopRight = float(0)
                    inBottomLeft = float(0)
                    inBottomRight = float(0)

                    for c in range(channels):
                        gradOutValue = grad_outputs[gout_idx(b, h, w, c)]
                        inTopLeft = im[input_idx(b, h, w, c)]
                        topLeftDotProduct += inTopLeft * gradOutValue
                        grad_inputs[input_idx(b, h, w, c)] += xWeightTopLeft * yWeightTopLeft * gradOutValue

                        inTopRight = im[input_idx(b, h, w + 1, c)]
                        topRightDotProduct += inTopRight * gradOutValue
                        grad_inputs[input_idx(b, h, w + 1, c)] += float(1 - xWeightTopLeft) * yWeightTopLeft * gradOutValue

                        inBottomLeft = im[input_idx(b, h + 1, w, c)]
                        bottomLeftDotProduct += inBottomLeft * gradOutValue
                        grad_inputs[input_idx(b, h + 1, w, c)] += xWeightTopLeft * float(1 - yWeightTopLeft) * gradOutValue

                        inBottomRight = im[input_idx(b, h + 1, w + 1, c)]
                        bottomRightDotProduct += inBottomRight * gradOutValue
                        grad_inputs[input_idx(b, h + 1, w + 1, c)] += float(1 - xWeightTopLeft) * float(1 - yWeightTopLeft) * gradOutValue

                    yf = -xWeightTopLeft * topLeftDotProduct
                    yf += xWeightTopLeft * bottomLeftDotProduct
                    yf -= float(1 - xWeightTopLeft) * topRightDotProduct
                    yf += float(1 - xWeightTopLeft) * bottomRightDotProduct

                    xf = -yWeightTopLeft * topLeftDotProduct
                    xf += yWeightTopLeft * topRightDotProduct
                    xf -= float(1 - yWeightTopLeft) * bottomLeftDotProduct
                    xf += float(1 - yWeightTopLeft) * bottomRightDotProduct

                    grad_grid[grid_idx(b, h, w, 0)] = yf * (height_f - 1) / 2
                    grad_grid[grid_idx(b, h, w, 1)] = xf * (width_f - 1) / 2
        return (grad_inputs, grad_grid)


class AbstractSpatialTransformerGradTOp(Op):
    def make_node(self, inp, theta, grad_grid, out_height, out_width):
        _inp = as_tensor_variable(inp)
        _theta = as_tensor_variable(theta)
        _grad_grid = as_tensor_variable(grad_grid)
        _out_height = as_scalar(out_height)
        _out_width = as_scalar(out_width)

        out = theano.tensor.tensor(dtype=_theta.type.dtype,
                                   broadcastable=_theta.broadcastable)

        return Apply(self, [_inp, _theta, _grad_grid, _out_height, _out_width], [out])

    def perform(self, node, inputs, output_storage):
        inp, theta, _grad_grid, out_height, out_width = inputs
        out = output_storage[0]

        num_batch, num_channels, height, width = inp.shape

        grid = self.sampling_grid(out_height, out_width)
        transposed_grid = np.transpose(grid, axes=(1, 0))
        batch_grid = np.asarray(num_batch * [transposed_grid])

        grad_grid = np.reshape(_grad_grid, (num_batch, out_height * out_width, 2))
        grad_grid_transposed = np.transpose(grad_grid, axes=(0, 2, 1))

        out[0] = np.asarray([np.matmul(grad_grid_transposed[i], batch_grid[i]) for i in range(num_batch)]).astype(theta.dtype)

    def sampling_grid(self, height, width):
        # Create sampling grid
        x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
                               np.linspace(-1, 1, height))
        ones = np.ones(np.prod(x_t.shape))
        grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        return grid
