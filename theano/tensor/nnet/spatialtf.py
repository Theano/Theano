"""
Concrete spatial transformer implementation
"""
from __future__ import absolute_import, print_function, division

import theano
from theano.gof.op import Op
from theano.scalar import as_scalar
from theano.tensor import as_tensor_variable
from theano.tensor.extra_ops import cpu_contiguous
from theano.gof import Apply
from theano.gradient import grad_not_implemented
import numpy as np


def sampling_grid(height, width):
    # Create sampling grid
    x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
                           np.linspace(-1, 1, height))
    ones = np.ones(np.prod(x_t.shape))
    grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    return grid


class TransformerGrid(Op):
    def make_node(self, theta, out_dims):
        _theta = cpu_contiguous(as_tensor_variable(theta))

        if _theta.ndim != 3:
            raise TypeError('SpatialTransformerGrid (make_node) requires theta to '
                            'be a 3D tensor; received "%s" (%i dims)' %
                            (theta, _theta.ndim))

        assert _theta.type.dtype in ('float16', 'float32', 'float64')

        _out_dims = cpu_contiguous(as_tensor_variable(out_dims))
        _out_dims = theano.tensor.basic.cast(out_dims, 'int64')

        _grid = theano.tensor.tensor(dtype=_theta.type.dtype,
                                     broadcastable=len(out_dims) * (False,))

        return Apply(self, [_theta, _out_dims], [_grid])

    def perform(self, node, inputs, output_storage):
        theta, out_dims = inputs

        assert len(out_dims) == 4
        # Theta should be in the format (batch_size, 2, 3)
        assert len(theta.shape) == 3
        assert (theta.shape[1] == 2) and (theta.shape[2] == 3)
        # Check if theta has the same batch size as out_dims
        assert theta.shape[0] == out_dims[0]

        num_batch = theta.shape[0]
        grid_out = output_storage[0]

        out_height, out_width = out_dims[2:]
        grid = sampling_grid(out_height, out_width)
        # Generate transformed grid with shape (num_batch, 2, out_height * out_width)
        transformed_grid = np.dot(theta, grid)
        # Dimshuffle grid into (2, num_batch, out_height * out_width)
        transposed_grid = np.transpose(transformed_grid, axes=(1, 0, 2))
        # Reshape into (2, num_batch, out_height, out_width)
        grid_out[0] = np.reshape(transposed_grid, (2, num_batch, out_height, out_width)).astype(theta.dtype)

    def grad(self, inputs, grads):
        theta, out_dims = inputs
        dgrid = grads[0]

        dtheta = TransformerGradT()(theta, dgrid)
        return [dtheta, grad_not_implemented(self, 1, out_dims)]


class TransformerSampler(Op):
    def __init__(self, border_mode='nearest'):
        self.border_mode = border_mode

    def make_node(self, inp, grid):
        _inp = cpu_contiguous(as_tensor_variable(inp))

        if _inp.ndim != 4:
            raise TypeError('SpatialTransformerSampler (make_node) requires input to '
                            'be a 4D tensor; received "%s" (%i dims)' %
                            (inp, _inp.ndim))

        assert _inp.type.dtype in ('float16', 'float32', 'float64')

        _grid = cpu_contiguous(as_tensor_variable(grid))

        if _grid.ndim != 4:
            raise TypeError('SpatialTransformerSampler (make_node) requires grid to '
                            'be a 4D tensor; received "%s" (%i dims)' %
                            (grid, _grid.ndim))

        out = theano.tensor.tensor(dtype=_inp.type.dtype,
                                   broadcastable=_inp.broadcastable)

        return Apply(self, [_inp, _grid], [out])

    def grad(self, inputs, grads):
        inp, grid = inputs
        grad_outputs = grads[0]

        grad_inp, grad_grid = TransformerGradI(self.border_mode)(inp, grid, grad_outputs)

        return [grad_inp, grad_grid]

    def perform(self, node, inputs, output_storage):
        inp, grid = inputs
        assert len(inp.shape) == 4
        assert len(grid.shape) == 4

        out = output_storage[0]

        out_height, out_width = grid.shape[2:]
        num_batch, num_channels, height, width = inp.shape
        border_mode = node.op.border_mode

        # Convert inp from NCHW to NHWC format
        inp_transposed = np.transpose(inp, axes=(0, 2, 3, 1))

        height_f, width_f = float(height), float(width)

        # Scale coordinates from [-1, 1] to [0, dimension -1], where dimension
        # can be the width or height
        x = grid[0, :].flatten()
        x = (x + 1) / 2 * (width_f - 1)

        y = grid[1, :].flatten()
        y = (y + 1) / 2 * (height_f - 1)

        # Obtain indices of the 2x2 pixel neighborhood surrounding the coordinates;
        # we need those in floatX for interpolation and in int64 for indexing.
        x0_f = np.floor(x)
        y0_f = np.floor(y)
        x1_f = x0_f + 1
        y1_f = y0_f + 1

        # For indexing, we need to take care of the border mode for outside pixels.
        x0, y0, x1, y1 = (None, None, None, None)
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

        # Compute offsets and indices
        height_stride = width
        batch_stride = width * height
        base = np.repeat(np.arange(num_batch, dtype='int64') * batch_stride,
                         out_height * out_width)
        base_y0 = base + y0 * height_stride
        base_y1 = base + y1 * height_stride
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1
        # Use indices to lookup pixels for all samples
        inp_flat = inp_transposed.reshape((-1, num_channels))
        Ia = inp_flat[idx_a]
        Ib = inp_flat[idx_b]
        Ic = inp_flat[idx_c]
        Id = inp_flat[idx_d]
        # Compute bilinear interpolation weights
        wa = ((x1_f - x) * (y1_f - y))[:, np.newaxis]
        wb = ((x1_f - x) * (y - y0_f))[:, np.newaxis]
        wc = ((x - x0_f) * (y1_f - y))[:, np.newaxis]
        wd = ((x - x0_f) * (y - y0_f))[:, np.newaxis]
        # Compute interpolated values
        transformed_inputs = np.sum([wa * Ia, wb * Ib, wc * Ic, wd * Id], axis=0)
        # Reshape flat array into NHWC format
        output = np.reshape(transformed_inputs, (num_batch, out_height, out_width, num_channels))
        # Dimshuffle NHWC tensor into NCHW format
        out[0] = np.transpose(output, axes=(0, 3, 1, 2)).astype(inp.dtype)


class TransformerGradI(Op):
    def __init__(self, border_mode):
        self.border_mode = border_mode

    def make_node(self, inp, grid, grad_outputs):
        _inp = cpu_contiguous(as_tensor_variable(inp))
        assert _inp.ndim == 4

        _grid = cpu_contiguous(as_tensor_variable(grid))
        assert _grid.ndim == 4

        _grad_outputs = cpu_contiguous(as_tensor_variable(grad_outputs))
        assert _grad_outputs.ndim == 4

        grad_inp = theano.tensor.tensor(dtype=_inp.type.dtype,
                                        broadcastable=_inp.broadcastable)

        grad_grid = theano.tensor.tensor(dtype=_inp.type.dtype,
                                         broadcastable=_inp.broadcastable)

        return Apply(self, [_inp, _grid, _grad_outputs], [grad_inp, grad_grid])

    def perform(self, node, inputs, output_storage):
        inp, grid, grad_outputs = inputs
        assert len(inp.shape) == 4
        assert len(grid.shape) == 4
        assert len(grad_outputs.shape) == 4

        grad_inp_out = output_storage[0]
        grad_grid_out = output_storage[1]

        out_height, out_width = grid.shape[2:]
        num_batch, num_channels, height, width = inp.shape
        border_mode = node.op.border_mode

        # Convert gradient of outputs' tensor from NCHW to NHWC format
        t_grad_outputs = np.transpose(grad_outputs, axes=(0, 2, 3, 1))

        height_f, width_f = float(height), float(width)

        # Scale coordinates from [-1, 1] to [0, dimension -1], where dimension
        # can be the width or height
        x = grid[0, :].flatten()
        x = (x + 1) / 2 * (width_f - 1)

        y = grid[1, :].flatten()
        y = (y + 1) / 2 * (height_f - 1)

        # Obtain indices of the 2x2 pixel neighborhood surrounding the coordinates;
        # we need those in floatX for interpolation and in int64 for indexing.
        x0_f = np.floor(x)
        y0_f = np.floor(y)
        x1_f = x0_f + 1
        y1_f = y0_f + 1

        # For indexing, we need to take care of the border mode for outside pixels.
        x0, y0, x1, y1 = (None, None, None, None)
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

        G_out = t_grad_outputs.reshape((-1, num_channels))

        # Compute offsets and indices for gradient of inputs
        height_stride = width
        batch_stride = width * height
        base = np.repeat(np.arange(num_batch, dtype='int64') * batch_stride,
                         out_height * out_width)
        base_y0 = base + y0 * height_stride
        base_y1 = base + y1 * height_stride
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1
        # Compute bilinear interpolation weights
        wa = ((x1_f - x) * (y1_f - y))[:, np.newaxis]
        wb = ((x1_f - x) * (y - y0_f))[:, np.newaxis]
        wc = ((x - x0_f) * (y1_f - y))[:, np.newaxis]
        wd = ((x - x0_f) * (y - y0_f))[:, np.newaxis]

        # Compute gradients of the inputs
        grad_inputs = np.zeros((num_batch * height * width, num_channels), dtype=inp.dtype)
        grad_inputs[idx_a] += wa * G_out
        grad_inputs[idx_b] += wb * G_out
        grad_inputs[idx_c] += wc * G_out
        grad_inputs[idx_d] += wd * G_out
        # Reshape gradients to NHWC tensor format
        grad_inputs = np.reshape(grad_inputs, (num_batch, height, width, num_channels))
        # Dimshuffle tensor from NHWC to NCHW format
        grad_inp_out[0] = np.transpose(grad_inputs, axes=(0, 3, 1, 2))

        # Compute gradients of the sampling grid
        grad_gradients = np.zeros((2, num_batch * out_height * out_width), dtype=grid.dtype)

        inp_transposed = np.transpose(inp, axes=(0, 2, 3, 1))
        inp_flat = inp_transposed.reshape((-1, num_channels))

        # shapes: (n * h * w, c) * (n * h * w, c)
        tl = inp_flat[idx_a] * G_out
        tr = inp_flat[idx_b] * G_out
        bl = inp_flat[idx_c] * G_out
        br = inp_flat[idx_d] * G_out

        xw_top_left = (x1_f - x)[:, np.newaxis]  # shape (n * h * w, 1)
        xw_bottom_left = (x - x0_f)[:, np.newaxis]  # shape (n * h * w, 1)

        yw_top_left = (y1_f - y)[:, np.newaxis]  # shape (n * h * w, 1)
        yw_bottom_left = (y - y0_f)[:, np.newaxis]  # shape (n * h * w, 1)

        # resulting shape: (n * h * w, c)
        xf = - yw_top_left * tl + yw_top_left * tr - yw_bottom_left * bl + yw_bottom_left * br
        yf = - xw_top_left * tl + xw_bottom_left * bl - xw_bottom_left * tr + xw_bottom_left * br

        # Sum over feature maps of xf and yf
        grad_gradients[0, :] = np.sum(xf, axis=1)  # * (width_f - 1) / 2
        grad_gradients[1, :] = np.sum(yf, axis=1)  # * (height_f - 1) / 2

        grad_grid_out[0] = np.reshape(grad_gradients, grid.shape)


class TransformerGradT(Op):
    def make_node(self, theta, grad_grid):
        _theta = as_tensor_variable(theta)
        _grad_grid = as_tensor_variable(grad_grid)

        out = theano.tensor.tensor(dtype=_theta.type.dtype,
                                   broadcastable=_theta.broadcastable)

        return Apply(self, [_theta, _grad_grid], [out])

    def perform(self, node, inputs, output_storage):
        theta, grad_grid = inputs
        out = output_storage[0]

        num_batch = theta.shape[0]
        out_height, out_width = grad_grid.shape[2:]

        grid = sampling_grid(out_height, out_width)
        # (3, h * w) -> (h * w, 3)
        transposed_grid = np.transpose(grid, axes=(1, 0))
        # repeat sampling grid, for all images in the batch, i.e. (n, h * w, 3)
        batch_grid = np.asarray(num_batch * [transposed_grid], dtype=theta.dtype)

        # reshape gradients of grid from (n, h, w, 2) -> (n, h * w, 2)
        _grad_grid_transposed = np.transpose(grad_grid, axes=(1, 0, 2, 3))
        # (n, h * w, 2) -> (n, 2, h * w)
        _grad_grid = np.reshape(_grad_grid_transposed, (num_batch, 2, out_height * out_width))

        out[0] = np.asarray([np.matmul(_grad_grid[i], batch_grid[i]) for i in range(num_batch)]).astype(theta.dtype)


def spatialtf(img, theta, scale_width=1, scale_height=1, border_mode='nearest'):
    """
    Spatial transformer (by Jaderberg et. al).

    Parameters
    ----------
    img : tensor
        Images to which the transformations will be applied. The implementation
        assumes the tensor is in NCHW format, where N is the number of images,
        C is the number of color channels, H is the height of the inputs, and
        W is width of the inputs.
    theta : tensor
        Affine transformation tensor containing one affine transformation
        matrix per image. ``theta`` is usually generated by the localization
        network.
    scale_height: float
        A float specifying the scaling factor for the height of the output
        image. A value of 1 will keep the original height of the input. Values
        larger than 1 will upsample the input. Values below 1 will downsample
        the input.
    scale_width: float
        A float specifying the scaling factor for the width of the output
        image. A value of 1 will keep the original width of the input. Values
        larger than 1 will upsample the input. Values below 1 will downsample
        the input.

    Returns
    -------
    out : tensor
        Transformed images with width and height properly scaled.

    Notes
    -----
    Currently, cuDNN only supports 2D transformations with 2x3 affine
    transformation matrices.

    Bilinear interpolation is the only grid sampler method available.
    """
    img = as_tensor_variable(img)
    assert img.ndim == 4

    theta = as_tensor_variable(theta)
    assert theta.ndim == 3

    num_batch, num_channels, height, width = img.shape
    out_height = theano.tensor.cast(theano.tensor.ceil(scale_height * height), 'int64')
    out_width = theano.tensor.cast(theano.tensor.ceil(scale_width * width), 'int64')

    out_dims = (num_batch, num_channels, out_height, out_width)
    out_dims = tuple([as_scalar(v).astype('int64') for v in out_dims])

    grid = TransformerGrid()(theta, out_dims)
    sampler = TransformerSampler()(img, grid)
    return sampler
