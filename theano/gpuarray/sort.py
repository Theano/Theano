from __future__ import absolute_import, print_function, division
import os
from string import Template

import theano
from theano import Apply
from theano.tensor import as_tensor_variable
from theano.tensor.sort import TopKOp

from .basic_ops import (GpuKernelBase, Kernel, infer_context_name,
                        as_gpuarray_variable, gpu_contiguous)
from .opt import register_opt, op_lifter, register_opt2
from .type import GpuArrayType

try:
    import pygpu
    import pygpu.gpuarray as ga
except ImportError as e:
    # To make sure theano is importable
    pass


# TODO add support is slice size is larger than max allowed block size (1024)
# TODO add runtime opt, if k==1, use max/min reduce
# TODO sort / argsort

class GpuTopKOp(GpuKernelBase, TopKOp):
    '''
    Implements TopKOp() on gpu

    '''
    __props__ = TopKOp.__props__
    def __init__(self, axis=-1, return_indices=False, return_values=True):
        GpuKernelBase.__init__(self)
        TopKOp.__init__(
            self, axis=axis,
            return_values=return_values,
            return_indices=return_indices)

    def c_headers(self):
        return ['gpuarray_api.h', 'gpuarray_helper.h', 'numpy_compat.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__), pygpu.get_include()]

    '''
    def c_code_cache_version(self):
        return (1,)
    '''

    def gpu_kernels(self, node, nodename):
        # load kernel source
        device_type = node.inputs[0].type.context.kind
        kernel_ext = {b'cuda':'.cu', b'opencl':'.cl'}[device_type]
        try:
            kernel_filename = 'topk_kernel%s' % kernel_ext
            with open(os.path.join(
                os.path.dirname(__file__), kernel_filename
            ), 'r') as f:
                kernel_src = f.read()
        except FileNotFoundError:
            raise RuntimeError(
                'Cannot find GPU kernel '
                'implementation for device "%s"' % device_type)

        # prepare "$" macros
        ndim = node.inputs[0].ndim
        dstv_strides_code = ''.join('ga_ssize dstv_strides_%d, ' % i for i in range(ndim))
        dsti_strides_code = ''.join('ga_ssize dsti_strides_%d, ' % i for i in range(ndim))
        src_strides_code = ''.join('ga_ssize src_strides_%d, ' % i for i in range(ndim))
        set_slice_code = '''
    gidx = gid %% dims_%(i)d;
    gid /= dims_%(i)d;
    {dstv};
    {dsti};
    src = ptr_add(src, gidx*src_strides_%(i)d);\n'''.format(
            dstv='dstv = ptr_add(dstv, gidx*dstv_strides_%(i)d)' if self.return_values else '',
            dsti='dsti = ptr_add(dsti, gidx*dsti_strides_%(i)d)' if self.return_indices else '')
        set_slice_code = ''.join(
            set_slice_code % dict(i=j) for j in range(1, ndim))
        flags = Kernel.get_flags(node.inputs[0].dtype)
        dst = ''
        if self.return_values:
            dst += 'INPUT_TYPE *dstv, '
        if self.return_values:
            dst += 'INDEX_TYPE *dsti, '
        write_value = 'ptr_at(dstv, out_idx * dstv_strides_0) = xval' if self.return_values else ''
        write_index = 'ptr_at(dsti, out_idx * dsti_strides_0) = (INDEX_TYPE)idx' if self.return_indices else ''
        subs = dict(
            inp_t=ga.dtype_to_ctype(node.inputs[0].dtype),
            out_t=ga.dtype_to_ctype(node.outputs[0].dtype),
            dims=''.join('ga_size dims_%d, ' % i for i in range(1, ndim)),
            dstv='INPUT_TYPE *dstv,' if self.return_values else '',
            dsti='INDEX_TYPE *dsti,' if self.return_indices else '',
            dstv_strides=dstv_strides_code,
            dsti_strides=dsti_strides_code,
            src_strides=src_strides_code,
            set_slice=set_slice_code,
            write_value=write_value,
            write_index=write_index,
            ndim=str(ndim))

        # substitute "$" macros in kernel code
        kernel_src = Template(kernel_src).substitute(**subs)

        # compile kernel
        param_types = [ga.SIZE] * (ndim - 1)  # dims
        for _ in range(int(self.return_values) + int(self.return_indices)):
            param_types.append(ga.GpuArray)  # dst*
            param_types.extend([ga.SSIZE] * ndim)  # dst*_strides
        param_types.append(ga.SIZE)  # k
        param_types.append(ga.GpuArray)  # src
        param_types.extend([ga.SSIZE] * ndim)  # src_strides
        param_types.append(ga.SIZE) # size
        return [Kernel(
            code=kernel_src,
            name='k_topk_dense',
            params=param_types,
            flags=flags,
            objvar='k_topk_dense_' + nodename
        )]

    def c_code(self, node, nodename, inps, outs, sub):
        if node.inputs[0].type.context.kind != b'cuda':
            raise NotImplementedError('We only have CUDA implementation so far.')
        x, k = inps
        inp_dtc = pygpu.dtypes.dtype_to_ctype(node.inputs[0].dtype).upper()
        if not self.return_indices:
            yv, = outs
            out_dtype_s = ''
            out_dtc = ''
        else:
            if self.return_values:
                yv, yi = outs
            else:
                yi, = outs
            out_dtype_s = node.outputs[0].dtype
            out_dtc = pygpu.dtypes.dtype_to_ctype(out_dtype_s).upper()
        fail = sub['fail']
        ctx = sub['params']
        k_dtype = node.inputs[1].type.dtype_specs()[1]
        MAX_TPB = 1024  # max thread per block
        WARP_SIZE = 32

        ndim = node.inputs[0].ndim
        reordered_axes = list(range(ndim))
        axis = self.axis % ndim
        del(reordered_axes[axis])
        reordered_axes = [axis] + reordered_axes
        dims = ', '.join('(void*)(dims+%d)' % i for i in reordered_axes[1:])
        prep_output = ''
        if self.return_values:
            def_dvstrides = 'const ssize_t *dvstrides = PyGpuArray_STRIDES(%s)' % yv
            params_dv = '(void*)(%s->ga.data),\n' % yv
            params_dv += ''.join('(void*)(dvstrides+%d), ' % i for i in reordered_axes)
            prep_output += '''
    if (0 != theano_prep_output(
        &%(yv)s, %(ndim)d, odims,
        %(inp_dtc)s, GA_C_ORDER, %(ctx)s)) {
        %(fail)s;
    }\n''' % locals()
        else:
            def_dvstrides = params_dv = ''

        if self.return_indices:
            def_distrides = 'const ssize_t *distrides = PyGpuArray_STRIDES(%s)' % yi
            params_di = '(void*)(%s->ga.data),\n' % yi
            params_di += ''.join('(void*)(distrides+%d), ' % i for i in reordered_axes)
            prep_output += '''
    if (0 != theano_prep_output(
        &%(yi)s, %(ndim)d, odims,
        %(out_dtc)s, GA_C_ORDER, %(ctx)s)) {
        %(fail)s;
    }\n''' % locals()
        else:
            def_distrides = params_di = ''
        sstrides = ', '.join('(void*)(sstrides+%d)' % i for i in reordered_axes)
        code = '''
{
    const size_t *dims = PyGpuArray_DIMS(%(x)s);
    size_t odims[%(ndim)d];
    for (int i=0; i<%(ndim)d; i++) {
        odims[i] = dims[i];
    }
    odims[%(axis)d] = *((%(k_dtype)s*)(PyArray_DATA(%(k)s)));
    if (odims[0] > %(MAX_TPB)d) {
        PyErr_SetString(
            PyExc_ValueError,
            "topk: slice size larger than %(MAX_TPB)d is not supported");
        %(fail)s; }
    %(prep_output)s

    // TODO better scheduling?
    size_t blk[6];
    size_t *grd = blk+3;
    blk[1] = blk[2] = 1;
    grd[0] = grd[1] = grd[2] = 1;
    // round up to multiples of warp size
    blk[0] = ((dims[0] + %(WARP_SIZE)d - 1) / %(WARP_SIZE)d) * %(WARP_SIZE)d;
    for(int i=0; i<%(ndim)d; ++i) {
        if (i!=%(axis)d)
            grd[0] *= dims[i];
    }

    %(def_dvstrides)s;
    %(def_distrides)s;
    const ssize_t *sstrides = PyGpuArray_STRIDES(%(x)s);
    void* args[] = {
        %(dims)s
        %(params_dv)s
        %(params_di)s
        (void*)(odims+%(axis)d),
        (void*)(%(x)s->ga.data),
        %(sstrides)s,
        (void*)(dims+%(axis)d)
    };

    int err = GpuKernel_call(
        &k_topk_dense_%(nodename)s, 3,
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

    def make_node(self, inp, k, idx_dtype='int64'):
        ctx_name = infer_context_name(inp)
        inp = as_gpuarray_variable(inp, ctx_name)
        k = as_tensor_variable(k)
        bcast = inp.type.broadcastable
        outs = []
        if self.return_indices:
            outs.append(GpuArrayType(
                dtype=idx_dtype,
                broadcastable=bcast,
                context_name=ctx_name)())
        if self.return_values:
            outs.append(inp.type())
        return Apply(self, [inp, k], outs)

    def get_params(self, node):
        return node.inputs[0].type.context

    # def get_op_params(self):
        # return [('AXIS', self.axis)]

@register_opt('fast_compile')
@op_lifter([TopKOp])
@register_opt2([TopKOp], 'fast_compile')
def local_gpua_topkop(op, ctx_name, inputs, outputs):
    axis = op.axis
    rv = op.return_values
    ri = op.return_indices
    x, k = inputs
    x = as_gpuarray_variable(x, ctx_name)

    y = outputs[-1]
    return GpuTopKOp(
        axis=axis, return_values=rv, return_indices=ri)(x, k, idx_dtype=y.dtype)
