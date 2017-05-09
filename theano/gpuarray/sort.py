from __future__ import absolute_import, print_function, division
import os

import theano
from theano import Apply
from theano.tensor import as_tensor_variable
from theano.tensor.sort import ArgTopKOp

from .basic_ops import (GpuKernelBase, Kernel, infer_context_name,
                        as_gpuarray_variable, gpu_contiguous)
from .type import GpuArrayType

try:
    import pygpu
except ImportError as e:
    # To make sure theano is importable
    pass


class GpuSortOp(object):
    # TODO
    pass

class GpuArgSortOp(object):
    # TODO
    pass

class GpuArgTopKOp(ArgTopKOp, GpuKernelBase):
    '''
    implement argtopk() on gpu

    '''
    __props__ = ArgTopKOp.__props__
    def __init__(self, axis=-1):
        ArgTopKOp.__init__(self, axis=axis)

    def c_headers(self):
        return ['gpuarray_api.h', 'gpuarray_helper.h', 'numpy_compat.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__), pygpu.get_include()]

    def gpu_kernels(self, node, nodename):
        device_type = str(node.inputs[0].type.context.kind)
        kernel_ext = dict(cuda='.cu', opencl='.cl')[device_type]
        flags = Kernel.get_flags(node.inputs[0].dtype)
        try:
            kernel_filename = 'topk_kernel%s' % kernel_ext
            with open(os.path.join(
                os.path.dirname(__file__), kernel_filename
            )) as f:
                kernel_src = f.read()
        except FileNotFoundError:
            raise RuntimeError(
                'Cannot find GPU kernel '
                'implementation for device "%s"' % device_type)
            return [Kernel(
                kernel_src,
                params='TODO_params',
                name='topk_kernel',
                flags=flags,
            )]

    def c_code(self, node, nodename, inps, outs, sub):
        if node.inputs[0].type.context.kind != b'cuda':
            raise NotImplementedError('We only have CUDA implementation so far.')
        x, k = inps
        y, = outs
        fail = sub['fail']
        ctx = sub['params']
        out_dtype = pygpu.dtypes.dtype_to_ctype(self.out_dtype).upper()
        MAX_TPB = 1024  # max thread per block
        WARP_SIZE = 32
        code = '''
{
    // prepare output
    const size_t *dims = PyGpuArray_DIMS(%(x)s);
    const size_t *odims[1] = {*((%(out_dtype)s)PyArray_DATA(%(k)s))};
    if (odims[0] > %(MAX_TPB)d) {
        PyErr_SetString(
            PyExc_ValueError,
            "topk: slice size larger than %(MAX_TPB)d is not supported");
        %(fail)s; }
    if (0 != theano_prep_output(
        &%(y)s, 1, odims,
        %(out_dtype)s, GA_C_ORDER, %(ctx)s)) {
        %(fail)s;
    }
    size_t blk[6] = ;
    size_t grd = blk+3;
    blk[1] = blk[2] = 1;
    grd[0] = grd[1] = grd[2] = 1;
    // round up to multiples of warp size
    blk[0] = (dims[0] + (%(WARP_SIZE)d - 1) / %(WARP_SIZE)d) * %(WARP_SIZE)d;

    void* args[] = {
        ((void*)(%(y)s->ga.data)),
        ((void*)(%(x)s->ga.data)),
        (void*)dims, (void*)odims
    };

    int err = GpuKernel_call(
        &topk_kernel, 3,
        grd, blk,
        blk[0] * gpuarray_get_elsize(%(x)s->ga.typecode),
        args);
    if (err != GA_NO_ERROR) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "gpu kernel topk_kernel failed to execute");
        %(fail)s;
    }
}
        '''
        return code % locals()

    def make_node(self, inp, k, out_dtype='int64'):
        ctx_name = infer_context_name(inp)
        inp = as_gpuarray_variable(inp, ctx_name)
        k = as_tensor_variable(k)
        bcast = inp.type.broadcastable
        return Apply(
            self, [inp, k],
            [GpuArrayType(
                dtype=out_dtype,
                broadcastable=bcast,
                context_name=ctx_name)()])

    def get_params(self, node):
        return node.inputs[0].type.context

    def get_op_params(self):
        return [('AXIS', self.axis)]

