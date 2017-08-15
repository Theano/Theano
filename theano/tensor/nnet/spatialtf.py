"""
Concrete spatial transformer implementation
"""
from __future__ import absolute_import, print_function, division

import theano
from theano.tensor import as_tensor_variable
# Instantiate abstract_spatialtf here, so the user just have to import this module
from .abstract_spatialtf import AbstractSpatialTransformerOp


def spatialtf(inp, theta, scale_height=1, scale_width=1, border_mode='nearest'):
    inp = as_tensor_variable(inp)
    assert inp.ndim == 4

    height, width = inp.shape[2:]
    out_height = theano.tensor.cast(scale_height * height, 'int64')
    out_width = theano.tensor.cast(scale_width * width, 'int64')

    return AbstractSpatialTransformerOp(border_mode)(inp, theta, out_height, out_width)


def spatialtf_cpu(inp, theta, out_height, out_width, border_mode='nearest'):
    # Assumes tensor is in NCHW format
    num_batch, num_channels, height, width = inp.shape
    theta = theano.tensor.reshape(theta, (-1, 2, 3))

    # grid of (x_t, y_t, 1), eq (1) in ref [1]
    out_height = theano.tensor.cast(out_height, 'int64')
    out_width = theano.tensor.cast(out_width, 'int64')
    grid = _meshgrid(out_height, out_width)
    # transform a x (x_t, y_t, 1)^t -> (x_s, y_s)
    t_g = theano.tensor.dot(theta, grid)
    x_s = t_g[:, 0]
    y_s = t_g[:, 1]
    x_s_flat = x_s.flatten()
    y_s_flat = y_s.flatten()

    # dimshuffle input to (bs, height, width, channels)
    inputs_dim = inp.dimshuffle(0, 2, 3, 1)
    inputs_transformed = _interpolate(inputs_dim, x_s_flat, y_s_flat,
                                      out_height, out_width, border_mode)

    output = theano.tensor.reshape(inputs_transformed, (num_batch, out_height, out_width, num_channels))
    output = output.dimshuffle(0, 3, 1, 2)  # dimshuffle to conv format
    return output


def _interpolate(im, x, y, out_height, out_width, border_mode):
    # *_f are floats
    num_batch, height, width, channels = im.shape
    height_f = theano.tensor.cast(height, theano.config.floatX)
    width_f = theano.tensor.cast(width, theano.config.floatX)

    # scale coordinates from [-1, 1] to [0, dimension - 1], where dimension
    # can be the width or height
    x = (x + 1) / 2 * (width_f - 1)
    y = (y + 1) / 2 * (height_f - 1)

    # obtain indices of the 2x2 pixel neighborhood surrounding the coordinates;
    # we need those in floatX for interpolation and in int64 for indexing.
    x0_f = theano.tensor.floor(x)
    y0_f = theano.tensor.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1

    # for indexing, we need to take care of the border mode for outside pixels.
    if border_mode == 'nearest':
        x0 = theano.tensor.clip(x0_f, 0, width_f - 1)
        x1 = theano.tensor.clip(x1_f, 0, width_f - 1)
        y0 = theano.tensor.clip(y0_f, 0, height_f - 1)
        y1 = theano.tensor.clip(y1_f, 0, height_f - 1)
    elif border_mode == 'mirror':
        w = 2 * (width_f - 1)
        x0 = theano.tensor.minimum(x0_f % w, -x0_f % w)
        x1 = theano.tensor.minimum(x1_f % w, -x1_f % w)
        h = 2 * (height_f - 1)
        y0 = theano.tensor.minimum(y0_f % h, -y0_f % h)
        y1 = theano.tensor.minimum(y1_f % h, -y1_f % h)
    elif border_mode == 'wrap':
        x0 = theano.tensor.mod(x0_f, width_f)
        x1 = theano.tensor.mod(x1_f, width_f)
        y0 = theano.tensor.mod(y0_f, height_f)
        y1 = theano.tensor.mod(y1_f, height_f)
    else:
        raise ValueError("border_mode must be one of "
                         "'nearest', 'mirror', 'wrap'")
    x0, x1, y0, y1 = (theano.tensor.cast(v, 'int64') for v in (x0, x1, y0, y1))

    # The input is [num_batch, height, width, channels]. We do the lookup in
    # the flattened input, i.e [num_batch*height*width, channels]. We need
    # to offset all indices to match the flat version
    dim2 = width
    dim1 = width * height
    base = theano.tensor.repeat(
        theano.tensor.arange(num_batch, dtype='int64') * dim1, out_height * out_width)
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

    # calculate interpolated values
    wa = ((x1_f - x) * (y1_f - y)).dimshuffle(0, 'x')
    wb = ((x1_f - x) * (y - y0_f)).dimshuffle(0, 'x')
    wc = ((x - x0_f) * (y1_f - y)).dimshuffle(0, 'x')
    wd = ((x - x0_f) * (y - y0_f)).dimshuffle(0, 'x')
    output = theano.tensor.sum([wa * Ia, wb * Ib, wc * Ic, wd * Id], axis=0)
    return output


def _linspace(start, stop, num):
    # Theano linspace. Behaves similar to np.linspace
    start = theano.tensor.cast(start, theano.config.floatX)
    stop = theano.tensor.cast(stop, theano.config.floatX)
    num = theano.tensor.cast(num, theano.config.floatX)
    step = (stop - start) / (num - 1)
    return theano.tensor.arange(num, dtype=theano.config.floatX) * step + start


def _meshgrid(height, width):
    # This function is the grid generator from eq. (1) in reference [1].
    # It is equivalent to the following numpy code:
    #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
    #                         np.linspace(-1, 1, height))
    #  ones = np.ones(np.prod(x_t.shape))
    #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    # It is implemented in Theano instead to support symbolic grid sizes.
    # Note: If the image size is known at layer construction time, we could
    # compute the meshgrid offline in numpy instead of doing it dynamically
    # in Theano. However, it hardly affected performance when we tried.
    x_t = theano.tensor.dot(theano.tensor.ones((height, 1)),
                            _linspace(-1.0, 1.0, width).dimshuffle('x', 0))
    y_t = theano.tensor.dot(_linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
                            theano.tensor.ones((1, width)))

    x_t_flat = x_t.reshape((1, -1))
    y_t_flat = y_t.reshape((1, -1))
    ones = theano.tensor.ones_like(x_t_flat)
    grid = theano.tensor.concatenate([x_t_flat, y_t_flat, ones], axis=0)
    return grid


def spatialtf_gradi_cpu(inp, theta, out_height, out_width):
    raise NotImplemented('CPU spatial transformer gradient of inputs not yet implemented.')


def spatialtf_gradt_cpu(inp, theta, out_height, out_width):
    raise NotImplemented('CPU spatial transformer gradient of transformation not yet implemented.')
