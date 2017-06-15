from __future__ import absolute_import, print_function, division
import os
from string import Template

import numpy as np
from theano import Apply
from theano.tensor import as_tensor_variable
from theano.tensor.sort import TopKOp

from .basic_ops import (GpuKernelBase, Kernel, infer_context_name,
                        as_gpuarray_variable)
from .opt import register_opt, op_lifter, register_opt2
from .type import GpuArrayType

try:
    import pygpu
    import pygpu.gpuarray as ga
except ImportError as e:
    # To make sure theano is importable
    pass

# TODO GPU sort / argsort
# TODO support when k >= 2^31


class GpuTopKOp(GpuKernelBase, TopKOp):
    '''
    Implements TopKOp on gpu

    '''
    __props__ = TopKOp.__props__
    _f16_ok = True

    def __init__(
        self, axis=-1,
        idx_dtype='int64',
        return_values=True,
        return_indices=True
    ):
        GpuKernelBase.__init__(self)
        TopKOp.__init__(
            self, axis=axis,
            idx_dtype=idx_dtype,
            return_values=return_values,
            return_indices=return_indices)

    def c_headers(self):
        return ['gpuarray_api.h', 'gpuarray_helper.h', 'numpy_compat.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__), pygpu.get_include()]

    def c_code_cache_version(self):
        return (1,)

    def gpu_kernels(self, node, nodename):
        # load kernel source
        device_type = node.inputs[0].type.context.kind
        knames = ['k_topk_dense', 'k_topk_dense_large']
        kernel_ext = {b'cuda': '.cu', b'opencl': '.cl'}[device_type]
        common_ext = {b'cuda': '.cuh', b'opencl': '.h'}[device_type]
        kernel_src = {}
        for kname in knames:
            with open(os.path.join(
                os.path.dirname(__file__), 'c_code', kname + kernel_ext
            ), 'r') as f:
                kernel_src[kname] = f.read()

        with open(os.path.join(
            os.path.dirname(__file__), 'c_code', 'k_topk_common' + common_ext
        ), 'r') as f:
            common_src = f.read()

        # prepare "$" macros
        if device_type == b'cuda':
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
            subs = dict(
                inp_t=ga.dtype_to_ctype(node.inputs[0].dtype),
                out_t=ga.dtype_to_ctype(self.idx_dtype),
                dims=''.join('ga_size dims_%d, ' % i for i in range(1, ndim)),
                dstv='INPUT_TYPE *dstv,' if self.return_values else '',
                dsti='INDEX_TYPE *dsti,' if self.return_indices else '',
                dstv_strides=dstv_strides_code if self.return_values else '',
                dsti_strides=dsti_strides_code if self.return_indices else '',
                src_strides=src_strides_code,
                set_slice=set_slice_code,
                write_value=int(self.return_values),
                write_index=int(self.return_indices),
                ndim=str(ndim),
                use_half=int(node.inputs[0].dtype == 'float16')
                )
        elif device_type == b'opencl':
            raise NotImplementedError()

        # compile kernels
        kernels = []
        param_types = [ga.SIZE] * (ndim - 1)  # dims
        for _ in range(int(self.return_values) + int(self.return_indices)):
            param_types.append(ga.GpuArray)  # dst*
            param_types.extend([ga.SSIZE] * ndim)  # dst*_strides
        param_types.append(ga.SIZE)  # k
        param_types.append(ga.GpuArray)  # src
        param_types.extend([ga.SSIZE] * ndim)  # src_strides
        param_types.append(ga.SIZE)  # size
        kernels.append(Kernel(
            code=Template(common_src + kernel_src['k_topk_dense']).substitute(**subs),
            name='k_topk_dense',
            params=param_types,
            flags=flags,
            objvar='k_topk_dense_' + nodename
            ))
        param_types.append(np.uint16)  # inp_per_thread
        kernels.append(Kernel(
            code=Template(common_src + kernel_src['k_topk_dense_large']).substitute(**subs),
            name='k_topk_dense_large',
            params=param_types,
            flags=flags,
            objvar='k_topk_dense_large_' + nodename
            ))
        return kernels

    def c_code(self, node, nodename, inps, outs, sub):
        if node.inputs[0].type.context.kind != b'cuda':
            raise NotImplementedError(
                '%s: We only have CUDA '
                'implementation so far.' % self.__class__.__name__)
        x, k = inps
        inp_dtc = ga.dtype_to_typecode(node.inputs[0].dtype)
        if not self.return_indices:
            yv, = outs
        elif self.return_values:
            yv, yi = outs
        else:
            yi, = outs
        out_dtype_s = self.idx_dtype
        out_dtc = ga.dtype_to_typecode(out_dtype_s)
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
        dims = ''.join('(void*)(dims+%d), ' % i for i in reordered_axes[1:])
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
    const ssize_t k_ = ((%(k_dtype)s*)(PyArray_DATA(%(k)s)))[0];
    const size_t *dims = PyGpuArray_DIMS(%(x)s);
    size_t odims[%(ndim)d];
    for (int i=0; i<%(ndim)d; i++)
        odims[i] = dims[i];


    odims[%(axis)d] = k_>=0 ? k_ : -k_;

    if (0 == odims[%(axis)d]) {
        PyErr_SetString(
            PyExc_ValueError,
            "topk: k must not be zero");
        %(fail)s;
    } else if (dims[%(axis)d] < odims[%(axis)d]){
        PyErr_SetString(
            PyExc_ValueError,
            "topk: k cannot larger than size on specified axis %(axis)d");
        %(fail)s;
    } else if (dims[%(axis)d] > INT_MAX) {
        PyErr_SetString(
            PyExc_ValueError,
            "topk: on GPU, array size on specified axis cannot larger or equal than 2^31");
        %(fail)s;
    }
    %(prep_output)s

    size_t blk[6];
    size_t *grd = blk+3;
    blk[0] = blk[1] = blk[2] = 1;
    grd[0] = grd[1] = grd[2] = 1;
    for(int i=0; i<%(ndim)d; ++i) {
        if (i!=%(axis)d)
            grd[0] *= dims[i];
        else
            blk[0] = dims[i];
    }
    // round up to multiples of warp size
    blk[0] = ((blk[0] + %(WARP_SIZE)d - 1) / %(WARP_SIZE)d) * %(WARP_SIZE)d;

    %(def_dvstrides)s;
    %(def_distrides)s;
    const ssize_t *sstrides = PyGpuArray_STRIDES(%(x)s);
    // inputs per thread
    unsigned short ipt = (dims[%(axis)d] + (%(MAX_TPB)d / 2)-1) / (%(MAX_TPB)d / 2);
    void* args[] = {
        %(dims)s
        %(params_dv)s
        %(params_di)s
        (void*)(&k_),
        (void*)(%(x)s->ga.data),
        %(sstrides)s,
        (void*)(dims+%(axis)d),
        (void*)(&ipt)
    };

    int err;
    if (blk[0] > %(MAX_TPB)d) {
        // LAUNCH_OUT_OF_RESOURCE if a 1024 sized block is used
        blk[0] = %(MAX_TPB)d / 2;
        err = GpuKernel_call(
            &k_topk_dense_large_%(nodename)s, 3,
            grd, blk, 0,
            args);
    } else {
        err = GpuKernel_call(
            &k_topk_dense_%(nodename)s, 3,
            grd, blk, 0,
            args);
    }
    if (err != GA_NO_ERROR) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "gpu kernel topk_kernel failed to execute");
        %(fail)s;
    }
}
        '''
        return code % locals()

    def make_node(self, inp, kth):
        ctx_name = infer_context_name(inp)
        inp = as_gpuarray_variable(inp, ctx_name)
        kth = as_tensor_variable(kth)
        bcast = inp.type.broadcastable
        outs = []
        if self.return_values:
            outs.append(inp.type())
        if self.return_indices:
            outs.append(GpuArrayType(
                dtype=self.idx_dtype,
                broadcastable=bcast,
                context_name=ctx_name)())
        return Apply(self, [inp, kth], outs)

    def get_params(self, node):
        return node.inputs[0].type.context


@register_opt('fast_compile')
@op_lifter([TopKOp], cuda_only=True)
@register_opt2([TopKOp], 'fast_compile')
def local_gpua_topkop(op, ctx_name, inputs, outputs):
    if isinstance(op, GpuTopKOp):
        return False

    axis = op.axis
    rv = op.return_values
    ri = op.return_indices
    x, k = inputs
    x = as_gpuarray_variable(x, ctx_name)

    gpu_op = GpuTopKOp(
        axis=axis,
        idx_dtype=op.idx_dtype,
        return_values=rv,
        return_indices=ri)
    rets = gpu_op(x, k)
    return rets
