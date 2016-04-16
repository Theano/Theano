from __future__ import absolute_import, print_function, division
import os.path

from theano import Apply, config, Op

from theano.compile import optdb
from theano.gof import LocalOptGroup
from theano.tensor.basic import as_tensor_variable
from theano.tensor.opt import in2out

from .basic_ops import as_gpuarray_variable, infer_context_name
from .opt_util import inplace_allocempty

try:
    import pygpu
    from pygpu import blas
except ImportError as e:
    # To make sure theano is importable
    pass


class BlasOp(Op):
    def c_headers(self):
        return ['<blas_api.h>', '<numpy_compat.h>', '<gpuarray_helper.h>']

    def c_header_dirs(self):
        return [pygpu.get_include(), os.path.dirname(__file__)]

    def c_init_code(self):
        return ['import_pygpu__blas();']


class GpuGemv(BlasOp):
    """
    Gemv on the GPU.

    """
    __props__ = ('inplace',)

    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def make_node(self, y, alpha, A, x, beta):
        ctx_name = infer_context_name(y, A, x)
        A = as_gpuarray_variable(A, ctx_name)
        x = as_gpuarray_variable(x, ctx_name)
        y = as_gpuarray_variable(y, ctx_name)
        alpha = as_tensor_variable(alpha)
        beta = as_tensor_variable(beta)
        assert alpha.ndim == 0
        assert beta.ndim == 0
        assert A.ndim == 2
        assert x.ndim == 1
        assert y.ndim == 1
        assert A.dtype == x.dtype == y.dtype
        return Apply(self, [y, alpha, A, x, beta], [y.type()])

    def perform(self, node, inputs, out_storage):
        y, alpha, A, x, beta = inputs
        inplace = self.inplace
        if inplace and y.strides[0] < 0:
            inplace = False
        out_storage[0][0] = blas.gemv(alpha, A, x, beta, y,
                                      overwrite_y=inplace)

    def c_code(self, node, name, inp, out, sub):
        vars = dict(out=out[0], y=inp[0], alpha=inp[1], A=inp[2], x=inp[3],
                    beta=inp[4], fail=sub['fail'], name=name)
        if self.inplace:
            code = """
                   if (%(y)s->ga.strides[0] <= 0) {
                     %(out)s = theano_try_copy(%(out)s, %(y)s);
                     if (%(out)s == NULL) {
                       %(fail)s
                     }
                   } else {
                     Py_XDECREF(%(out)s);
                     %(out)s = %(y)s;
                     Py_INCREF(%(out)s);
                   }
                   """ % vars
        else:
            code = """
                   %(out)s = theano_try_copy(%(out)s, %(y)s);
                   if (%(out)s == NULL) {
                       %(fail)s
                   }
                   """ % vars
        code += """
        if (pygpu_blas_rgemv(cb_no_trans,
                             ((dtype_%(alpha)s *)PyArray_DATA(%(alpha)s))[0],
                             %(A)s, %(x)s,
                             ((dtype_%(beta)s *)PyArray_DATA(%(beta)s))[0],
                             %(out)s, 0) == -1) {
            %(fail)s
        }
        """ % vars
        if config.gpuarray.sync:
            code += """
            GpuArray_sync(&%(out)s->ga);
            """ % vars
        return code

    def c_code_cache_version(self):
        return (4,)

gpugemv_no_inplace = GpuGemv(inplace=False)
gpugemv_inplace = GpuGemv(inplace=True)


class GpuGemm(BlasOp):
    """
    Gemm on the GPU.

    """
    __props__ = ('inplace',)
    _f16_ok = True

    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def make_node(self, C, alpha, A, B, beta):
        ctx_name = infer_context_name(C, A, B)
        A = as_gpuarray_variable(A, ctx_name)
        B = as_gpuarray_variable(B, ctx_name)
        C = as_gpuarray_variable(C, ctx_name)
        alpha = as_tensor_variable(alpha)
        beta = as_tensor_variable(beta)
        assert alpha.ndim == 0
        assert beta.ndim == 0
        assert A.ndim == 2
        assert B.ndim == 2
        assert C.ndim == 2
        assert A.dtype == B.dtype == C.dtype
        return Apply(self, [C, alpha, A, B, beta], [C.type()])

    def perform(self, node, inputs, outputs):
        C, alpha, A, B, beta = inputs
        inplace = self.inplace
        if inplace and not C.flags.forc:
            inplace = False
        outputs[0][0] = blas.gemm(alpha, A, B, beta, C,
                                  overwrite_c=inplace)

    def c_code(self, node, name, inp, out, sub):
        vars = dict(out=out[0], C=inp[0], alpha=inp[1], A=inp[2], B=inp[3],
                    beta=inp[4], fail=sub['fail'], name=name)
        if self.inplace:
            code = """
                   if (!GpuArray_ISONESEGMENT(&%(C)s->ga)) {
                     %(out)s = theano_try_copy(%(out)s, %(C)s);
                     if (%(out)s == NULL) {
                       %(fail)s
                     }
                   } else {
                     Py_XDECREF(%(out)s);
                     %(out)s = %(C)s;
                     Py_INCREF(%(out)s);
                   }
                   """ % vars
        else:
            code = """
                   %(out)s = theano_try_copy(%(out)s, %(C)s);
                   if (%(out)s == NULL) {
                       %(fail)s
                   }
                   """ % vars
        code += """
        if (pygpu_blas_rgemm(cb_no_trans, cb_no_trans,
                             ((dtype_%(alpha)s *)PyArray_DATA(%(alpha)s))[0],
                             %(A)s, %(B)s,
                             ((dtype_%(beta)s *)PyArray_DATA(%(beta)s))[0],
                             %(out)s, 0) == -1) {
            %(fail)s
        }
        """ % vars
        if config.gpuarray.sync:
            code += """
            GpuArray_sync(&%(out)s->ga);
            """ % vars
        return code

    def c_code_cache_version(self):
        return (5,)

gpugemm_no_inplace = GpuGemm(inplace=False)
gpugemm_inplace = GpuGemm(inplace=True)


class GpuGer(BlasOp):
    """
    Ger on the GPU.

    """
    __props__ = ('inplace',)

    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def make_node(self, A, alpha, x, y):
        ctx_name = infer_context_name(A, x, y)
        A = as_gpuarray_variable(A, ctx_name)
        x = as_gpuarray_variable(x, ctx_name)
        y = as_gpuarray_variable(y, ctx_name)
        alpha = as_tensor_variable(alpha)
        assert alpha.ndim == 0
        assert A.ndim == 2
        assert x.ndim == 1
        assert y.ndim == 1
        assert A.dtype == x.dtype == y.dtype
        return Apply(self, [A, alpha, x, y], [A.type()])

    def perform(self, node, inp, out):
        A, alpha, x, y = inp
        inplace = self.inplace
        if inplace and not A.flags.forc:
            inplace = False
        out[0][0] = blas.ger(alpha, x, y, A,
                             overwrite_a=inplace)

    def c_code(self, node, name, inp, out, sub):
        vars = dict(out=out[0], A=inp[0], alpha=inp[1], x=inp[2], y=inp[3],
                    fail=sub['fail'], name=name)
        if self.inplace:
            code = """
                   if (!GpuArray_ISONESEGMENT(&%(A)s->ga)) {
                     %(out)s = theano_try_copy(%(out)s, %(A)s);
                     if (%(out)s == NULL) {
                       %(fail)s
                     }
                   } else {
                     Py_XDECREF(%(out)s);
                     %(out)s = %(A)s;
                     Py_INCREF(%(out)s);
                   }
                   """ % vars
        else:
            code = """
                   %(out)s = theano_try_copy(%(out)s, %(A)s);
                   if (%(out)s == NULL) {
                       %(fail)s
                   }
                   """ % vars
        code += """
        if (pygpu_blas_rger(((dtype_%(alpha)s *)PyArray_DATA(%(alpha)s))[0],
                            %(x)s, %(y)s, %(out)s, 0) == -1) {
            %(fail)s
        }
        """ % vars
        if config.gpuarray.sync:
            code += """
            GpuArray_sync(&%(out)s->ga);
            """ % vars
        return code

    def c_code_cache_version(self):
        return (3,)


gpuger_no_inplace = GpuGer(inplace=False)
gpuger_inplace = GpuGer(inplace=True)


class GpuDot22(BlasOp):
    """
    Dot22 on the GPU.

    """
    __props__ = ()

    def make_node(self, x, y):
        ctx_name = infer_context_name(x, y)
        x = as_gpuarray_variable(x, ctx_name)
        y = as_gpuarray_variable(y, ctx_name)
        assert x.ndim == 2
        assert y.ndim == 2
        assert x.dtype == y.dtype
        otype = x.type.clone(
            broadcastable=(x.type.broadcastable[0], y.type.broadcastable[1]))
        return Apply(self, [x, y], [otype()])

    def perform(self, node, inputs, outputs):
        x, y = inputs

        out = pygpu.empty((x.shape[0], y.shape[1]), dtype=x.dtype,
                          context=x.context)
        outputs[0][0] = blas.gemm(1., x, y, 0., out,
                                  overwrite_c=True)

    def c_code(self, node, name, inputs, outputs, sub):
        dtype = node.inputs[0].dtype
        typecode = pygpu.gpuarray.dtype_to_typecode(dtype)
        vars = dict(A=inputs[0], B=inputs[1], dtype=dtype, out=outputs[0],
                    typecode=typecode,
                    fail=sub['fail'], name=name)
        code = """
        double one = 1.;
        double zero = 0.;

        size_t dims[] = {0, 0};
        dims[0] = PyGpuArray_DIMS(%(A)s)[0];
        dims[1] = PyGpuArray_DIMS(%(B)s)[1];

        if (theano_prep_output(&%(out)s, 2, dims, %(typecode)s, GA_C_ORDER,
                               %(A)s->context)) {
            %(fail)s
        }

        if (pygpu_blas_rgemm(cb_no_trans, cb_no_trans,
                             one,
                             %(A)s, %(B)s,
                             zero,
                             %(out)s, 0) == -1) {
            %(fail)s
        }
        """ % vars
        if config.gpuarray.sync:
            code += """
            GpuArray_sync(&%(out)s->ga);
            """ % vars
        return code

    def c_code_cache_version(self):
        return (4,)

gpu_dot22 = GpuDot22()


@inplace_allocempty(GpuGemv, 0)
def local_inplace_gpuagemv(node, inputs):
    return [gpugemv_inplace(*inputs)]


@inplace_allocempty(GpuGemm, 0)
def local_inplace_gpuagemm(node, inputs):
    return [gpugemm_inplace(*inputs)]


@inplace_allocempty(GpuGer, 0)
def local_inplace_gpuager(node, inputs):
    return [gpuger_inplace(*inputs)]

gpuablas_opt_inplace = in2out(LocalOptGroup(local_inplace_gpuagemv,
                                            local_inplace_gpuagemm,
                                            local_inplace_gpuager),
                              name='gpuablas_opt_inplace')

optdb.register('InplaceGpuaBlasOpt',
               gpuablas_opt_inplace,
               70.0, 'fast_run', 'inplace', 'gpuarray')
