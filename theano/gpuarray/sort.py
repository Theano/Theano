from __future__ import absolute_import, print_function, division
import os
from string import Template

from theano import Apply
from theano.tensor import as_tensor_variable
from theano.tensor.sort import TopKOp

from .basic_ops import (GpuKernelBase, Kernel, infer_context_name,
                        as_gpuarray_variable, gpuarray_helper_inc_dir)
from .opt import register_opt, op_lifter, register_opt2
from .type import GpuArrayType

try:
    import pygpu
    import pygpu.gpuarray as ga
except ImportError as e:
    # To make sure theano is importable
    pass


# TODO GPU sort / argsort
class GpuTopKOp(GpuKernelBase, TopKOp):
    '''
    Implements TopKOp on gpu

    '''
    __props__ = TopKOp.__props__
    _f16_ok = True

    def __init__(
        self, axis=-1,
        sorted=True,
        idx_dtype='int64',
        return_values=True,
        return_indices=True
    ):
        GpuKernelBase.__init__(self)
        TopKOp.__init__(
            self, axis=axis,
            sorted=sorted,
            idx_dtype=idx_dtype,
            return_values=return_values,
            return_indices=return_indices)

    def c_headers(self):
        return ['gpuarray_api.h', 'gpuarray_helper.h', 'numpy_compat.h']

    def c_header_dirs(self):
        return [
            os.path.dirname(__file__),
            gpuarray_helper_inc_dir(),
            pygpu.get_include()]

    def c_code_cache_version(self):
        return (4,)

    def gpu_kernels(self, node, nodename):
        # load kernel source
        device_type = node.inputs[0].type.context.kind
        kernel_ext = {b'cuda': '.cu', b'opencl': '.cl'}[device_type]
        common_ext = {b'cuda': '.cuh', b'opencl': '.h'}[device_type]

        # prepare "$" macros
        if device_type == b'cuda':
            ndim = node.inputs[0].ndim
            dstv_strides_code = ''.join('ssize_t dstv_strides_%d, ' % i for i in range(ndim))
            dsti_strides_code = ''.join('ssize_t dsti_strides_%d, ' % i for i in range(ndim))
            src_strides_code = ''.join('ssize_t src_strides_%d, ' % i for i in range(ndim))
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
            if self.return_values:
                set_slice_code += """
                dstv = ptr_add(dstv, dstv_offset);
                """
            if self.return_indices:
                set_slice_code += """
                dsti = ptr_add(dsti, dsti_offset);
                """
            set_slice_code += """
                src = ptr_add(src, src_offset);
            """
            flags = Kernel.get_flags(node.inputs[0].dtype)
            subs = dict(
                inp_t=ga.dtype_to_ctype(node.inputs[0].dtype),
                out_t=ga.dtype_to_ctype(self.idx_dtype),
                dims=''.join('size_t dims_%d, ' % i for i in range(1, ndim)),
                dstv='INPUT_TYPE *dstv,' if self.return_values else '',
                dstv_offset='size_t dstv_offset,' if self.return_values else '',
                dsti='INDEX_TYPE *dsti,' if self.return_indices else '',
                dsti_offset='size_t dsti_offset,' if self.return_indices else '',
                dstv_strides=dstv_strides_code if self.return_values else '',
                dsti_strides=dsti_strides_code if self.return_indices else '',
                src_strides=src_strides_code,
                set_slice=set_slice_code,
                write_value=int(self.return_values),
                write_index=int(self.return_indices),
                ndim=str(ndim)
                )
        elif device_type == b'opencl':
            raise NotImplementedError()

        # setup parameters
        param_types = [ga.SIZE] * (ndim - 1)  # dims
        for _ in range(self.return_values + self.return_indices):
            param_types.append(ga.GpuArray)  # dst*
            param_types.append(ga.SIZE)  # offset
            param_types.extend([ga.SSIZE] * ndim)  # dst*_strides
        param_types.append(ga.SIZE)  # k
        param_types.append(ga.GpuArray)  # src
        param_types.append(ga.SIZE)  # offset
        param_types.extend([ga.SSIZE] * ndim)  # src_strides
        param_types.append(ga.SIZE)  # size

        # load and compile kernels
        with open(os.path.join(
            os.path.dirname(__file__), 'c_code', 'topk_common' + common_ext
        )) as f:
            common_src = f.read()

        kernels = []

        def build_kernel(fname, kname, subs):
            with open(os.path.join(
                os.path.dirname(__file__), 'c_code', fname)
            ) as f:
                kernel_src = f.read()
            ker = Kernel(
                code=("#include <cluda.h>\n" +
                      Template(common_src + kernel_src).substitute(**subs)),
                name=kname,
                params=param_types,
                flags=flags,
                objvar=kname + nodename)
            return ker

        subs['count_t'] = 'int'
        kernels.append(
            build_kernel('topk_dense' + kernel_ext, 'k_topk_dense', subs))
        subs['kname'] = 'k_topk_dense_large'
        kernels.append(
            build_kernel('topk_dense_large' + kernel_ext, 'k_topk_dense_large', subs))
        subs['count_t'] = 'long long'
        subs['kname'] = 'k_topk_dense_xlarge'
        kernels.append(
            build_kernel('topk_dense_large' + kernel_ext, 'k_topk_dense_xlarge', subs))
        return kernels

    def c_code(self, node, nodename, inps, outs, sub):
        context = node.inputs[0].type.context
        if context.kind != b'cuda':
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
        # max threads per block
        MAX_TPB = context.maxlsize0
        # max blocks per grid
        MAX_BPG = context.maxgsize0
        WARP_SIZE = 32

        ndim = node.inputs[0].ndim
        reordered_axes = list(range(ndim))
        axis = self.axis % ndim
        del(reordered_axes[axis])
        reordered_axes = [axis] + reordered_axes
        dims = ''.join('dims[%d], ' % i for i in reordered_axes[1:])
        prep_output = ''
        if self.return_values:
            def_dvstrides = 'const ssize_t *dvstrides = PyGpuArray_STRIDES(%s)' % yv
            params_dv = '%s->ga.data, %s->ga.offset,\n' % (yv, yv)
            params_dv += ''.join('dvstrides[%d], ' % i for i in reordered_axes)
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
            params_di = '%s->ga.data, %s->ga.offset,\n' % (yi, yi)
            params_di += ''.join('distrides[%d], ' % i for i in reordered_axes)
            prep_output += '''
    if (0 != theano_prep_output(
        &%(yi)s, %(ndim)d, odims,
        %(out_dtc)s, GA_C_ORDER, %(ctx)s)) {
        %(fail)s;
    }\n''' % locals()
        else:
            def_distrides = params_di = ''
        sstrides = ', '.join('sstrides[%d]' % i for i in reordered_axes)
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
            "topk: kth must not be zero");
        %(fail)s;
    } else if (dims[%(axis)d] < odims[%(axis)d]) {
        PyErr_SetString(
            PyExc_ValueError,
            "topk: kth cannot be larger than the size of specified axis %(axis)d");
        %(fail)s;
    }
    %(prep_output)s

    size_t grid_size=1, block_size=1;
    for (int i=0; i<%(ndim)d; ++i) {
        if (i!=%(axis)d)
            grid_size *= dims[i];
        else
            block_size = dims[i];
    }
    // round up to multiples of warp size
    block_size = ((block_size + %(WARP_SIZE)d - 1) / %(WARP_SIZE)d) * %(WARP_SIZE)d;

    if (grid_size > %(MAX_BPG)d) {
        PyErr_SetString(
            PyExc_ValueError,
            "topk: too many slices to work with, expected <= %(MAX_BPG)d");
        %(fail)s;
    }

    %(def_dvstrides)s;
    %(def_distrides)s;
    const ssize_t *sstrides = PyGpuArray_STRIDES(%(x)s);

    int err;
    if (dims[%(axis)d] > (1u << 31)) {
        block_size = %(MAX_TPB)d;
        err = k_topk_dense_xlarge_call(
                1, &grid_size, &block_size, 0,
                %(dims)s
                %(params_dv)s
                %(params_di)s
                k_,
                %(x)s->ga.data,
                %(x)s->ga.offset,
                %(sstrides)s,
                dims[%(axis)d]
        );
    } else if (block_size > %(MAX_TPB)d) {
        block_size = %(MAX_TPB)d;
        err = k_topk_dense_large_call(
                1, &grid_size, &block_size, 0,
                %(dims)s
                %(params_dv)s
                %(params_di)s
                k_,
                %(x)s->ga.data,
                %(x)s->ga.offset,
                %(sstrides)s,
                dims[%(axis)d]
        );
    } else {
        err = k_topk_dense_call(
                1, &grid_size, &block_size, 0,
                %(dims)s
                %(params_dv)s
                %(params_di)s
                k_,
                %(x)s->ga.data,
                %(x)s->ga.offset,
                %(sstrides)s,
                dims[%(axis)d]
        );
    }
    if (err != GA_NO_ERROR) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "topk: gpu kernel failed to execute");
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
    axis = op.axis
    rv = op.return_values
    ri = op.return_indices
    x, k = inputs
    x = as_gpuarray_variable(x, ctx_name)

    gpu_op = GpuTopKOp(
        axis=axis,
        sorted=op.sorted,
        idx_dtype=op.idx_dtype,
        return_values=rv,
        return_indices=ri)
    rets = gpu_op(x, k)
    return rets
