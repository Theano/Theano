from __future__ import absolute_import, print_function, division
import unittest

import numpy as np
import theano
from theano import tensor as T
from theano import config
from theano.tests import unittest_tools as utt
from theano.gpuarray.tests.config import mode_with_gpu, mode_without_gpu
from theano.tensor.nnet.spatialtf import spatialtf
from theano.gpuarray.dnn import dnn_spatialtf


def spatialtf_symb(inp, theta, scale_width, scale_height, border_mode='nearest'):
    """
    Symbolic Spatial Transformer implementation using Theano from Lasagne
    Original author: skaae (https://github.com/skaae)
    """
    def _interpolate(im, x, y, out_height, out_width, border_mode):
        # *_f are floats
        num_batch, height, width, channels = im.shape
        height_f = T.cast(height, theano.config.floatX)
        width_f = T.cast(width, theano.config.floatX)

        # scale coordinates from [-1, 1] to [0, dimension - 1], where dimension
        # can be the width or height
        x = (x + 1) / 2 * (width_f - 1)
        y = (y + 1) / 2 * (height_f - 1)

        # obtain indices of the 2x2 pixel neighborhood surrounding the coordinates;
        # we need those in floatX for interpolation and in int64 for indexing.
        x0_f = T.floor(x)
        y0_f = T.floor(y)
        x1_f = x0_f + 1
        y1_f = y0_f + 1

        # for indexing, we need to take care of the border mode for outside pixels.
        if border_mode == 'nearest':
            x0 = T.clip(x0_f, 0, width_f - 1)
            x1 = T.clip(x1_f, 0, width_f - 1)
            y0 = T.clip(y0_f, 0, height_f - 1)
            y1 = T.clip(y1_f, 0, height_f - 1)
        elif border_mode == 'mirror':
            w = 2 * (width_f - 1)
            x0 = T.minimum(x0_f % w, -x0_f % w)
            x1 = T.minimum(x1_f % w, -x1_f % w)
            h = 2 * (height_f - 1)
            y0 = T.minimum(y0_f % h, -y0_f % h)
            y1 = T.minimum(y1_f % h, -y1_f % h)
        elif border_mode == 'wrap':
            x0 = T.mod(x0_f, width_f)
            x1 = T.mod(x1_f, width_f)
            y0 = T.mod(y0_f, height_f)
            y1 = T.mod(y1_f, height_f)
        else:
            raise ValueError("border_mode must be one of "
                             "'nearest', 'mirror', 'wrap'")
        x0, x1, y0, y1 = (T.cast(v, 'int64') for v in (x0, x1, y0, y1))

        # The input is [num_batch, height, width, channels]. We do the lookup in
        # the flattened input, i.e [num_batch*height*width, channels]. We need
        # to offset all indices to match the flat version
        dim2 = width
        dim1 = width * height
        base = T.repeat(
            T.arange(num_batch, dtype='int64') * dim1, out_height * out_width)
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
        output = T.sum([wa * Ia, wb * Ib, wc * Ic, wd * Id], axis=0)
        return output

    def _linspace(start, stop, num):
        # Theano linspace. Behaves similar to np.linspace
        start = T.cast(start, theano.config.floatX)
        stop = T.cast(stop, theano.config.floatX)
        num = T.cast(num, theano.config.floatX)
        step = (stop - start) / (num - 1)
        return T.arange(num, dtype=theano.config.floatX) * step + start

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
        x_t = T.dot(T.ones((height, 1)),
                    _linspace(-1.0, 1.0, width).dimshuffle('x', 0))
        y_t = T.dot(_linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
                    T.ones((1, width)))

        x_t_flat = x_t.reshape((1, -1))
        y_t_flat = y_t.reshape((1, -1))
        ones = T.ones_like(x_t_flat)
        grid = T.concatenate([x_t_flat, y_t_flat, ones], axis=0)
        return grid

    num_batch, num_channels, height, width = inp.shape
    theta = T.reshape(theta, (-1, 2, 3))

    # grid of (x_t, y_t, 1), eq (1) in ref [1]
    out_height = T.cast(T.ceil(height * scale_height), 'int64')
    out_width = T.cast(T.ceil(width * scale_width), 'int64')
    grid = _meshgrid(out_height, out_width)
    # transform a x (x_t, y_t, 1)^t -> (x_s, y_s)
    t_g = T.dot(theta, grid)
    x_s = t_g[:, 0]
    y_s = t_g[:, 1]
    x_s_flat = x_s.flatten()
    y_s_flat = y_s.flatten()

    # dimshuffle input to  (bs, height, width, channels)
    input_dim = inp.dimshuffle(0, 2, 3, 1)
    input_transformed = _interpolate(
        input_dim, x_s_flat, y_s_flat,
        out_height, out_width, border_mode)

    output = T.reshape(
        input_transformed, (num_batch, out_height, out_width, num_channels))
    output = output.dimshuffle(0, 3, 1, 2)  # dimshuffle to conv format
    return output


class TestTransformer(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def getInputs(self):
        inp_shape = (np.random.randint(1, 11), 1, 32, 32)
        num_images, num_channels, height, width = inp_shape

        inp = np.random.random(inp_shape).astype(config.floatX)

        transform = [[-1, 0, 0],
                     [0, -1, 0]]
        theta = np.asarray(num_images * [transform], dtype=config.floatX)
        scale_height = np.random.random()
        scale_width = np.random.random()

        return (inp, theta, scale_width, scale_height)

    def test_symb(self):
        """
        Compare CPU implementation with a symbolic one
        """
        inp, theta, scale_width, scale_height = self.getInputs()

        # Setup symbolic variables
        t_inp = T.tensor4('inp')
        t_theta = T.tensor3('theta')
        t_scale_height = T.scalar('scale_height')
        t_scale_width = T.scalar('scalar_width')

        symb_out_op = spatialtf_symb(t_inp, t_theta, t_scale_width, t_scale_height)
        symb_out_fn = theano.function([t_inp, t_theta, t_scale_width, t_scale_height],
                                      symb_out_op, mode=mode_without_gpu)
        symb_out = symb_out_fn(inp, theta, scale_width, scale_height)

        cpu_out_op = spatialtf(t_inp, t_theta, t_scale_width, t_scale_height)
        cpu_out_fn = theano.function([t_inp, t_theta, t_scale_width, t_scale_height],
                                     cpu_out_op, mode=mode_without_gpu)
        cpu_out = cpu_out_fn(inp, theta, scale_width, scale_height)

        # Check if results are approx. equal
        utt.assert_allclose(symb_out, cpu_out)

    def test_gpu(self):
        inp, theta, scale_width, scale_height = self.getInputs()
        # Setup symbolic variables
        t_inp = T.tensor4('inp')
        t_theta = T.tensor3('theta')
        t_scale_height = T.scalar('scale_height')
        t_scale_width = T.scalar('scalar_width')
        # Setup spatial transformer on the CPU
        op_cpu = spatialtf(t_inp, t_theta, t_scale_width, t_scale_height)
        fn_cpu = theano.function([t_inp, t_theta, t_scale_width, t_scale_height],
                                 op_cpu, mode=mode_without_gpu)
        # Evaluate CPU function
        out_cpu = fn_cpu(inp, theta, scale_width, scale_height)
        # Setup spatial transformer on the GPU using cuDNN
        op_gpu = dnn_spatialtf(t_inp, t_theta, t_scale_width, t_scale_height)
        fn_gpu = theano.function([t_inp, t_theta, t_scale_width, t_scale_height],
                                 op_gpu, mode=mode_with_gpu)
        # Evaluate GPU function
        out_gpu = fn_gpu(inp, theta, scale_width, scale_height)
        # Check results
        utt.assert_allclose(out_gpu, out_cpu)

    def test_gradi(self):
        inp_shape = (np.random.randint(1, 11), 1, 32, 32)
        num_images, num_channels, height, width = inp_shape

        inp = np.random.random(inp_shape).astype(config.floatX)
        transform = [[-1, 0, 0],
                     [0, -1, 0]]
        theta = np.asarray(num_images * [transform], dtype=config.floatX)
        scale_height = 0.5
        scale_width = 0.25

        def grad_inp_functor(inputs):
            out = spatialtf(inputs, theta, scale_width, scale_height)
            return out

        # Check results
        utt.verify_grad(grad_inp_functor, [inp])
