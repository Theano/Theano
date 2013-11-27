from theano import Op, Apply, config

from theano.tensor.blas import Dot22, Gemv, Gemm
from theano.sandbox.gpuarray.basic_ops import (HideC, as_gpuarray_variable)

try:
    import pygpu
    from pygpu import blas
except ImportError, e:
    # To make sure theano is importable
    pass


class BlasOp(HideC):
    def c_headers(self):
        return ['<blas_api.h>']

    def c_header_dirs(self):
        return [pygpu.get_include()]

    def c_init_code(self):
        return ['import_pygpu__blas();']


class GpuGemv(BlasOp, Gemv):
    def make_node(self, y, alpha, A, x, beta):
        res = Gemv.make_node(self, y, alpha, A, x, beta)
        A = as_gpuarray_variable(A)
        x = as_gpuarray_variable(x)
        y = as_gpuarray_variable(y)
        assert A.dtype == x.dtype == y.dtype == alpha.dtype == beta.dtype
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
                   Py_XDECREF(%(out)s);
                   %(out)s = %(y)s;
                   Py_INCREF(%(out)s);
                   """ % vars
        else:
            code = """
                   Py_XDECREF(%(out)s);
                   %(out)s = pygpu_copy(%(y)s, GA_ANY_ORDER);
                   if (%(out)s == NULL) {
                       %(fail)s
                   }
                   """ % vars
        code += """
        if (pygpu_blas_rgemv(cb_no_trans,
                             ((dtype_%(alpha)s *)PyArray_DATA(%(alpha)s))[0],
                             %(A)s, %(x)s,
                             ((dtype_%(beta)s *)PyArray_DATA(%(beta)s))[0],
                             %(out)s) == NULL) {
            %(fail)s
        }
        """ % vars
        if config.gpuarray.sync:
            code += """
            GpuArray_sync(&%(out)s->ga);
            """ % vars
        return code

    def c_code_cache_version(self):
        return (0,)

gpugemv_no_inplace = GpuGemv(inplace=False)
gpugemv_inplace = GpuGemv(inplace=True)


class GpuGemm(BlasOp, Gemm):
    def make_node(self, C, alpha, A, B, beta):
        res = Gemm.make_node(self, C, alpha, A, B, beta)
        A = as_gpuarray_variable(A)
        B = as_gpuarray_variable(B)
        C = as_gpuarray_variable(C)
        assert A.dtype == B.dtype == C.dtype == alpha.dtype == beta.dtype
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
                   Py_XDECREF(%(out)s);
                   %(out)s = %(C)s;
                   Py_INCREF(%(out)s);
                   """ % vars
        else:
            code = """
                   Py_XDECREF(%(out)s);
                   %(out)s = pygpu_copy(%(C)s, GA_ANY_ORDER);
                   if (%(out)s == NULL) {
                       %(fail)s
                   }
                   """ % vars
        code += """
        if (pygpu_blas_rgemm(cb_no_trans, cb_no_trans,
                             ((dtype_%(alpha)s *)PyArray_DATA(%(alpha)s))[0],
                             %(A)s, %(B)s,
                             ((dtype_%(beta)s *)PyArray_DATA(%(beta)s))[0],
                             %(out)s) == NULL) {
            %(fail)s
        }
        """ % vars
        if config.gpuarray.sync:
            code += """
            GpuArray_sync(&%(out)s->ga);
            """ % vars
        return code

    def c_code_cache_version(self):
        return (0,)


gpugemm_no_inplace = GpuGemm(inplace=False)
gpugemm_inplace = GpuGemm(inplace=True)


class GpuDot22(BlasOp, Dot22):
    def make_node(self, x, y):
        res = Dot22.make_node(self, x, y)
        x = as_gpuarray_variable(x)
        y = as_gpuarray_variable(y)
        assert x.dtype == y.dtype
        return Apply(self, [x, y], [x.type()])

    def perform(self, node, inputs, outputs):
        x, y = inputs

        out = pygpu.empty((x.shape[0], y.shape[1]), dtype=x.dtype)
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

        %(out)s = pygpu_empty(2, dims,
                            %(typecode)s,
                            GA_C_ORDER,
                            pygpu_default_context(), Py_None);
        if (!%(out)s) {
            %(fail)s
        }

        if (pygpu_blas_rgemm(cb_no_trans, cb_no_trans,
                             one,
                             %(A)s, %(B)s,
                             zero,
                             %(out)s) == NULL) {
            %(fail)s
        }
        """ % vars
        if config.gpuarray.sync:
            code += """
            GpuArray_sync(&%(out)s->ga);
            """ % vars
        return code

    def c_code_cache_version(self):
        return (0,)

    def c_headers(self):
        ret = super(GpuDot22, self).c_headers()
        return ret + ['<compyte/numpy_compat.h>']

gpu_dot22 = GpuDot22()

from theano.compile import optdb
from theano.gof import local_optimizer, LocalOptGroup
from theano.tensor.opt import in2out


@local_optimizer([gpugemv_no_inplace])
def local_inplace_gpuagemv(node):
    if node.op == gpugemv_no_inplace:
        return [gpugemv_inplace(*node.inputs)]


@local_optimizer([gpugemm_no_inplace])
def local_inplace_gpuagemm(node):
    if node.op == gpugemm_no_inplace:
        return [gpugemm_inplace(*node.inputs)]

gpuablas_opt_inplace = in2out(LocalOptGroup(
        local_inplace_gpuagemv, local_inplace_gpuagemm),
                              name='gpuablas_opt_inplace')
optdb.register('InplaceGpuaBlasOpt',
               gpuablas_opt_inplace,
               70.0, 'fast_run', 'inplace', 'gpuarray')
