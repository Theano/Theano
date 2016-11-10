from __future__ import absolute_import, print_function, division
import os.path

import pygpu
from theano import Apply
from theano.tensor.basic import as_tensor_variable

from .basic_ops import CGpuKernelBase, infer_context_name, as_gpuarray_variable


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

    def make_node(self, inp, ws, stride, pad):
        ctx_name = infer_context_name(inp)
        inp = as_gpuarray_variable(inp, ctx_name)
        assert (inp.ndim == self.ndim + 2)

        ws = as_tensor_variable(ws)
        stride = as_tensor_variable(stride)
        pad = as_tensor_variable(pad)
        assert ws.type.ndim == stride.type.ndim and ws.type.ndim == pad.type.ndim
        assert ws.type.ndim == 1

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

    def make_node(self, inp, out, out_grad, ws, stride, pad):
        ctx_name = infer_context_name(inp, out, out_grad)
        inp = as_gpuarray_variable(inp, ctx_name)
        assert (inp.ndim == self.ndim + 2)
        out = as_gpuarray_variable(out, ctx_name)
        assert (out.ndim == self.ndim + 2)
        out_grad = as_gpuarray_variable(out_grad, ctx_name)
        assert (out_grad.ndim in [4, 5])

        assert (out_grad.ndim == inp.ndim)
        assert (inp.ndim == out.ndim)

        ws = as_tensor_variable(ws)
        stride = as_tensor_variable(stride)
        pad = as_tensor_variable(pad)
        assert ws.type.ndim == stride.type.ndim and ws.type.ndim == pad.type.ndim
        assert ws.type.ndim == 1

        return Apply(self, [inp, out, out_grad, ws, stride, pad], [inp.type()])

    def get_params(self, node):
        return node.inputs[0].type.context


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

    def make_node(self, inp, out_grad, ws, stride, pad):
        ctx_name = infer_context_name(inp, out_grad)
        inp = as_gpuarray_variable(inp, ctx_name)
        assert (inp.ndim == self.ndim + 2)
        out_grad = as_gpuarray_variable(out_grad, ctx_name)
        assert (out_grad.ndim == self.ndim + 2)

        assert (out_grad.ndim == inp.ndim)

        ws = as_tensor_variable(ws)
        stride = as_tensor_variable(stride)
        pad = as_tensor_variable(pad)
        assert ws.type.ndim == stride.type.ndim and ws.type.ndim == pad.type.ndim
        assert ws.type.ndim == 1

        return Apply(self, [inp, out_grad, ws, stride, pad], [inp.type()])

    def get_params(self, node):
        return node.inputs[0].type.context

    def get_op_params(self):
        inc_pad = int(self.mode == 'average_inc_pad')
        sum_mode = int(self.mode == 'sum')
        return [('INC_PAD', inc_pad),
                ('SUM_MODE', sum_mode)]


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

    def make_node(self, inp, out, out_grad, ws, stride, pad):
        ctx_name = infer_context_name(inp, out, out_grad)
        inp = as_gpuarray_variable(inp, ctx_name)
        assert (inp.ndim == self.ndim + 2)
        out = as_gpuarray_variable(out, ctx_name)
        assert (out_grad.ndim == self.ndim + 2)
        out_grad = as_gpuarray_variable(out_grad, ctx_name)
        assert (out.ndim == self.ndim + 2)

        assert (out_grad.ndim == inp.ndim)
        assert (inp.ndim == out.ndim)

        ws = as_tensor_variable(ws)
        stride = as_tensor_variable(stride)
        pad = as_tensor_variable(pad)
        assert ws.type.ndim == stride.type.ndim and ws.type.ndim == pad.type.ndim
        assert ws.type.ndim == 1

        return Apply(self, [inp, out, out_grad, ws, stride, pad], [inp.type()])

    def get_params(self, node):
        return node.inputs[0].type.context
