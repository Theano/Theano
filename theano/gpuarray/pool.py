from __future__ import absolute_import, print_function, division
import os.path

import theano
from theano import Apply
from theano.tensor.basic import as_tensor_variable
from theano.tensor.signal.pool import Pool

from .basic_ops import (CGpuKernelBase, infer_context_name,
                        as_gpuarray_variable, gpu_contiguous)

try:
    import pygpu
except ImportError as e:
    # To make sure theano is importable
    pass


class GpuPool(CGpuKernelBase):
    """
    Implement the max and average pooling on the gpu.

    """
    __props__ = ('ignore_border', 'mode', 'ndim')

    def __init__(self, ignore_border, mode='max', ndim=2):
        self.ndim = ndim
        self.ignore_border = ignore_border
        if mode == 'average':
            mode = 'average_inc_pad'
        self.mode = mode
        CGpuKernelBase.__init__(self, ['pool.c'],
                                'APPLY_SPECIFIC(pool)')
        assert mode in ('max', 'sum', 'average_inc_pad', 'average_exc_pad')
        assert self.ndim in [2, 3]

    def c_headers(self):
        return ['gpuarray_api.h', 'gpuarray_helper.h', 'numpy_compat.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__), pygpu.get_include()]

    def make_node(self, inp, ws, stride=None, pad=None):
        ctx_name = infer_context_name(inp)
        inp = as_gpuarray_variable(inp, ctx_name)
        nd = self.ndim
        assert (inp.ndim == nd + 2)
        if stride is None:
            stride = ws
        if pad is None:
            pad = (0,) * nd
        elif isinstance(pad, (tuple, list)):
            if max(pad) != 0 and not self.ignore_border:
                raise ValueError('Padding works only with ignore_border=True')
            if isinstance(ws, (tuple, list)):
                if any(pad[i] >= ws[i] for i in range(nd)):
                    raise ValueError('Padding must be smaller than strides')

        ws = as_tensor_variable(ws)
        stride = as_tensor_variable(stride)
        pad = as_tensor_variable(pad)
        assert ws.ndim == stride.ndim and ws.ndim == pad.ndim
        assert ws.ndim == 1
        if not ws.dtype.startswith('int'):
            raise TypeError('Window shape parameters must be ints.')
        if not stride.dtype.startswith('int'):
            raise TypeError('Stride parameters must be ints.')
        if not pad.dtype.startswith('int'):
            raise TypeError('Padding parameters must be ints.')

        return Apply(self, [inp, ws, stride, pad], [inp.type()])

    def get_params(self, node):
        return node.inputs[0].type.context

    def get_op_params(self):
        ignore_border = int(self.ignore_border)
        max_pool = int(self.mode == 'max')
        inc_pad = int(self.mode != 'average_exc_pad')
        sum_mode = int(self.mode == 'sum')
        return [('IGNORE_BORDER', ignore_border),
                ('INC_PAD', inc_pad),
                ('MAX_POOL', max_pool),
                ('SUM_MODE', sum_mode)]

    def infer_shape(self, node, in_shapes):
        ws, stride, pad = [node.inputs[1], node.inputs[2], node.inputs[3]]
        shp = Pool.out_shape(in_shapes[0], ws, self.ignore_border, stride,
                             pad, self.ndim)
        return [shp]

    def grad(self, inp, grads):
        img, ws, stride, pad = inp
        grad, = grads

        grad = gpu_contiguous(grad)

        disc = [theano.gradient.DisconnectedType()() for i in inp[1:]]
        if self.mode == 'max':
            out = self(inp, ws, stride, pad)
            g_out = GpuMaxPoolGrad(ndim=self.ndim,
                                   ignore_border=self.ignore_border)(
                                       img, out, grad, ws, stride, pad)
            return [g_out] + disc
        else:
            g_out = GpuAveragePoolGrad(ndim=self.ndim,
                                       ignore_border=self.ignore_border,
                                       mode=self.mode)(
                                           img, grad, ws, stride, pad)
            return [g_out] + disc

    def connection_pattern(self, node):
        return [[1], [0], [0], [0]]


class GpuMaxPoolGrad(CGpuKernelBase):
    """
    Implement the grad of max pooling on the gpu.

    """
    __props__ = ('ignore_border', 'mode', 'ndim')

    def __init__(self, ignore_border, mode='max', ndim=2):
        self.ndim = ndim
        self.ignore_border = ignore_border
        self.mode = mode
        CGpuKernelBase.__init__(self, ['pool_max_grad.c'],
                                'APPLY_SPECIFIC(max_pool_grad)')
        assert mode == 'max'
        assert ndim in [2, 3]

    def c_headers(self):
        return ['gpuarray_api.h', 'gpuarray_helper.h', 'numpy_compat.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__), pygpu.get_include()]

    def make_node(self, inp, out, out_grad, ws, stride=None, pad=None):
        ctx_name = infer_context_name(inp, out, out_grad)
        nd = self.ndim
        inp = as_gpuarray_variable(inp, ctx_name)
        assert (inp.ndim == nd + 2)
        out = as_gpuarray_variable(out, ctx_name)
        assert (out.ndim == nd + 2)
        out_grad = as_gpuarray_variable(out_grad, ctx_name)
        assert (out_grad.ndim == nd + 2)

        assert (out_grad.ndim == inp.ndim)
        assert (inp.ndim == out.ndim)

        if stride is None:
            stride = ws
        if pad is None:
            pad = (0,) * nd
        ws = as_tensor_variable(ws)
        stride = as_tensor_variable(stride)
        pad = as_tensor_variable(pad)
        assert ws.ndim == stride.ndim and ws.ndim == pad.ndim
        assert ws.ndim == 1
        if not ws.dtype.startswith('int'):
            raise TypeError('Window shape parameters must be ints.')
        if not stride.dtype.startswith('int'):
            raise TypeError('Stride parameters must be ints.')
        if not pad.dtype.startswith('int'):
            raise TypeError('Padding parameters must be ints.')
        return Apply(self, [inp, out, out_grad, ws, stride, pad], [inp.type()])

    def get_params(self, node):
        return node.inputs[0].type.context

    def infer_shape(self, node, in_shapes):
        return [in_shapes[0]]

    def grad(self, inp, grads):
        x, maxout, gz, ws, stride, pad = inp
        ggx, = grads
        return ([theano.tensor.zeros_like(x),
                 theano.tensor.zeros_like(maxout),
                 GpuDownsampleFactorMaxGradGrad(ndim=self.ndim,
                                                ignore_border=self.ignore_border)(
                                                    x, maxout, ggx, ws, stride, pad)] +
                [theano.tensor.DisconnectedType()() for i in inp[3:]])

    def connection_pattern(self, node):
        return [[1], [1], [1], [0], [0], [0]]


class GpuAveragePoolGrad(CGpuKernelBase):
    """
    Implement the grad of average pooling on the gpu.

    """
    __props__ = ('ignore_border', 'mode', 'ndim')

    def __init__(self, ignore_border, mode='max', ndim=2):
        self.ndim = ndim
        self.ignore_border = ignore_border
        if mode == 'average':
            mode = 'average_inc_pad'
        self.mode = mode
        CGpuKernelBase.__init__(self, ['pool_ave_grad.c'],
                                'APPLY_SPECIFIC(ave_pool_grad)')
        assert mode in ('sum', 'average_inc_pad', 'average_exc_pad')
        assert ndim in [2, 3]

    def c_headers(self):
        return ['gpuarray_api.h', 'gpuarray_helper.h', 'numpy_compat.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__), pygpu.get_include()]

    def make_node(self, inp, out_grad, ws, stride=None, pad=None):
        ctx_name = infer_context_name(inp, out_grad)
        nd = self.ndim
        inp = as_gpuarray_variable(inp, ctx_name)
        assert (inp.ndim == nd + 2)
        out_grad = as_gpuarray_variable(out_grad, ctx_name)
        assert (out_grad.ndim == nd + 2)

        assert (out_grad.ndim == inp.ndim)

        if stride is None:
            stride = ws
        if pad is None:
            pad = (0,) * nd
        elif isinstance(pad, (tuple, list)):
            if max(pad) != 0 and not self.mode == 'average_exc_pad':
                raise ValueError('Padding must be zero for average_exc_pad')
        ws = as_tensor_variable(ws)
        stride = as_tensor_variable(stride)
        pad = as_tensor_variable(pad)
        assert ws.ndim == stride.ndim and ws.ndim == pad.ndim
        assert ws.ndim == 1
        if not ws.dtype.startswith('int'):
            raise TypeError('Window shape parameters must be ints.')
        if not stride.dtype.startswith('int'):
            raise TypeError('Stride parameters must be ints.')
        if not pad.dtype.startswith('int'):
            raise TypeError('Padding parameters must be ints.')
        return Apply(self, [inp, out_grad, ws, stride, pad], [inp.type()])

    def get_params(self, node):
        return node.inputs[0].type.context

    def get_op_params(self):
        inc_pad = int(self.mode == 'average_inc_pad')
        sum_mode = int(self.mode == 'sum')
        return [('INC_PAD', inc_pad),
                ('SUM_MODE', sum_mode)]

    def infer_shape(self, node, in_shapes):
        return [in_shapes[0]]

    def grad(self, inp, grads):
        x, gz, ws, stride, pad = inp
        ggx, = grads
        return ([theano.tensor.zeros_like(x),
                 GpuPool(ignore_border=self.ignore_border,
                         ndim=self.ndim, mode=self.mode)(
                             ggx, ws, stride, pad)] +
                [theano.gradient.DisconnectedType()() for i in inp[2:]])

    def connection_pattern(self, node):
        return [[1], [1], [0], [0], [0]]


class GpuDownsampleFactorMaxGradGrad(CGpuKernelBase):
    """
    Implement the grad of downsample with max on the gpu.

    """
    __props__ = ('ignore_border', 'mode', 'ndim')

    def __init__(self, ignore_border, mode='max', ndim=2):
        self.ndim = ndim
        self.ignore_border = ignore_border
        self.mode = mode
        CGpuKernelBase.__init__(self, ['pool_grad_grad.c'],
                                'APPLY_SPECIFIC(pool_grad_grad)')
        assert self.mode == 'max'
        assert self.ndim in [2, 3]

    def c_headers(self):
        return ['gpuarray_api.h', 'gpuarray_helper.h', 'numpy_compat.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__), pygpu.get_include()]

    def make_node(self, inp, out, out_grad, ws, stride=None, pad=None):
        ctx_name = infer_context_name(inp, out, out_grad)
        nd = self.ndim
        inp = as_gpuarray_variable(inp, ctx_name)
        assert (inp.ndim == nd + 2)
        out = as_gpuarray_variable(out, ctx_name)
        assert (out_grad.ndim == nd + 2)
        out_grad = as_gpuarray_variable(out_grad, ctx_name)
        assert (out.ndim == nd + 2)

        assert (out_grad.ndim == inp.ndim)
        assert (inp.ndim == out.ndim)

        if stride is None:
            stride = ws
        if pad is None:
            pad = (0,) * nd
        ws = as_tensor_variable(ws)
        stride = as_tensor_variable(stride)
        pad = as_tensor_variable(pad)
        assert ws.ndim == stride.ndim and ws.ndim == pad.ndim
        assert ws.ndim == 1
        if not ws.dtype.startswith('int'):
            raise TypeError('Window shape parameters must be ints.')
        if not stride.dtype.startswith('int'):
            raise TypeError('Stride parameters must be ints.')
        if not pad.dtype.startswith('int'):
            raise TypeError('Padding parameters must be ints.')
        return Apply(self, [inp, out, out_grad, ws, stride, pad], [inp.type()])

    def get_params(self, node):
        return node.inputs[0].type.context

    def infer_shape(self, node, in_shapes):
        return [in_shapes[1]]

    def grad(self, inp, grads):
        x, maxout, ggx, ws, stride, pad = inp
        gz, = grads
        return ([theano.tensor.zeros_like(x),
                theano.tensor.zeros_like(maxout),
                GpuMaxPoolGrad(ignore_border=self.ignore_border,
                               ndim=self.ndim)(
                                   x, maxout, gz, ws, stride, pad)] +
                [theano.gradient.DisconnectedType()() for i in inp[3:]])

    def connection_pattern(self, node):
        return [[1], [1], [1], [0], [0], [0]]
