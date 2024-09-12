from __future__ import absolute_import, print_function, division
import theano
import theano.ifelse
import theano.tensor as T
from theano.scalar import as_scalar


def theano_linspace(start, stop, num, dtype):
    # Theano linspace. Behaves similar to np.linspace
    start = T.cast(start, dtype)
    stop = T.cast(stop, dtype)
    num = T.cast(num, dtype)
    # if num is 1, we set step to 0 as it does not matter. Else we compute step.
    step = theano.ifelse.ifelse(theano.tensor.eq(num, 1),
                                T.cast(0, dtype),
                                (stop - start) / (num - 1))
    return T.arange(num, dtype=dtype) * step + start


def theano_spatialtf_sampling_grid(height, width, dtype):
    # Grid generator.
    # It is equivalent to the following numpy code:
    #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
    #                         np.linspace(-1, 1, height))
    #  ones = np.ones(np.prod(x_t.shape))
    #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

    # Returns a tensor with shape (3, height * width)

    x_t = T.dot(T.ones((height, 1), dtype),
                theano_linspace(-1.0, 1.0, width, dtype).dimshuffle('x', 0))
    y_t = T.dot(theano_linspace(-1.0, 1.0, height, dtype).dimshuffle(0, 'x'),
                T.ones((1, width), dtype))

    x_t_flat = x_t.reshape((1, -1))
    y_t_flat = y_t.reshape((1, -1))
    ones = T.ones_like(x_t_flat)
    grid = T.concatenate([x_t_flat, y_t_flat, ones], axis=0)
    return grid


def theano_spatialtf_upscale_values(v, limit):
    """
    Scale value v from frame [-1; 1] to frame [0; limit - 1].
    (v may be not in interval [-1; 1]).
    """
    return ((v + 1.0) * (limit - 1.0)) / 2.0


def theano_spatialtf_bilinear_kernel(values):
    """
    Bilinear sampling kernel.
    """
    return T.maximum(1.0 - abs(values), T.zeros(values.shape, values.dtype))


def theano_spatialtf_grid(theta, out_dims):
    """
    Generates grid coordinates for spatial transformer network.
    """
    theta = T.as_tensor_variable(theta)
    out_dims = T.as_tensor_variable(out_dims)

    assert theta.ndim == 3
    assert theta.type.dtype in ('float16', 'float32', 'float64')
    assert out_dims.ndim == 1
    assert out_dims.dtype in theano.tensor.basic.integer_dtypes

    # TODO: # Only 2D images are currently supported.

    theta = T.reshape(theta, (-1, 2, 3))

    # Generate grid.
    out_height = T.cast(out_dims[2], 'int64')
    out_width = T.cast(out_dims[3], 'int64')
    mesh_grid = theano_spatialtf_sampling_grid(out_height, out_width, theta.dtype)
    # theta has shape (B, 2, 3)
    # mesh_grid has shape (3, H*W)
    # Product has shape (B, 2, H*W).
    # Final output should have shape (B, H, W, 2)
    grid = T.dot(theta, mesh_grid).transpose(0, 2, 1)
    return T.reshape(grid, (-1, out_height, out_width, 2))


def theano_spatialtf_from_grid(inp, grid):
    inp = T.as_tensor_variable(inp)
    grid = T.as_tensor_variable(grid)

    assert inp.ndim == 4
    assert grid.ndim == 4
    assert inp.dtype in ('float16', 'float32', 'float64')
    assert grid.dtype in ('float16', 'float32', 'float64')

    num_batch, num_channels, height, width = inp.shape
    # TODO: assert num_batch == grid.shape[0]
    # TODO: assert grid.shape[3] == 2

    out_height = grid.shape[1]
    out_width = grid.shape[2]

    Q = out_height * out_width
    # We reshape grid to (B, H * W, 2)
    grid_reshaped = T.reshape(grid, (-1, out_height * out_width, 2))
    # We get x and y coordinates from grid: shape (B, H * W, 1), and we upscale them.
    all_x, all_y = T.split(grid_reshaped, [1, 1], 2, axis=2)
    all_x_scaled = theano_spatialtf_upscale_values(all_x, width)
    all_y_scaled = theano_spatialtf_upscale_values(all_y, height)
    # We prepare x, y to compute x - m, y -n,
    # for every 0 <= m < M, 0 <= n < N
    all_neg_ones = -T.ones((num_batch, Q, 1), grid.dtype)
    all_x_neg_ones = T.concatenate((all_x_scaled, all_neg_ones), axis=2)
    all_y_neg_ones = T.concatenate((all_y_scaled, all_neg_ones), axis=2)
    M1 = T.concatenate((T.ones((1, width), grid.dtype),
                        T.arange(width, dtype=grid.dtype).reshape((1, width))),
                       axis=0)
    N1 = T.concatenate((T.ones((1, height), grid.dtype),
                        T.arange(height, dtype=grid.dtype).reshape((1, height))),
                       axis=0)
    # Now we compute x - m, y - n.
    all_kxm = theano_spatialtf_bilinear_kernel(T.dot(all_x_neg_ones, M1)).reshape((num_batch, Q, 1, width))
    all_kyn = theano_spatialtf_bilinear_kernel(T.dot(all_y_neg_ones, N1)).reshape((num_batch, Q, height, 1))
    # We combine both values to get (x - m)(y - n) for every (x, y), m and n.
    # all_kyx should have shape (B, Q, N, M)
    all_kyx, _ = theano.scan(fn=lambda a, b: T.batched_dot(a, b), sequences=[all_kyn, all_kxm], n_steps=num_batch)
    # Now we compute output: every (x - m)(y - n) * corresponding input[m, n] for every channel.
    b_q_nm = all_kyx.reshape((num_batch, Q, height * width))
    b_nm_c = inp.reshape((num_batch, num_channels, height * width)).transpose(0, 2, 1)
    b_q_c = T.batched_dot(b_q_nm, b_nm_c)
    # We reshape output to shape (B, C, H, W)
    output = b_q_c.transpose(0, 2, 1).reshape((num_batch, num_channels, out_height, out_width))
    return output


def theano_spatialtf(inp, theta, scale_width=1, scale_height=1):
    """
    Symbolic Spatial Transformer implementation using Theano variables
    Jaderberg et. al: http://arxiv.org/abs/1506.02025
    """
    inp = T.as_tensor_variable(inp)
    theta = T.as_tensor_variable(theta)
    scale_width = as_scalar(scale_width)
    scale_height = as_scalar(scale_height)
    num_batch, num_channels, height, width = inp.shape

    out_height = T.cast(T.ceil(height * scale_height), 'int64')
    out_width = T.cast(T.ceil(width * scale_width), 'int64')
    grid = theano_spatialtf_grid(theta, [num_batch, num_channels, out_height, out_width])
    output = theano_spatialtf_from_grid(inp, grid)
    return output
