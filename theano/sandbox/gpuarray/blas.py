from theano import Apply, config

from theano.tensor.blas import Gemv
from theano.sandbox.gpuarray.basic_ops import (HideC, as_gpuarray_variable)

class GpuGemv(HideC, Gemv):
    def make_node(self, y, alpha, A, x, beta):
        res = Gemv.make_node(self, y, alpha, A, x, beta)
        A = as_gpuarray_variable(A)
        x = as_gpuarray_variable(x)
        y = as_gpuarray_variable(y)
        return Apply(self, [y, alpha, A, x, beta], [y.type()])

    def perform(*args):
        raise NotImplementedError

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
        if (pygpu_blas_sgemv(cb_no_trans,
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
            """
        return code

gpugemv_no_inplace = GpuGemv(inplace=False)
gpugemv_inplace = GpuGemv(inplace=True)

from theano.compile import optdb
from theano.gof import local_optimizer, EquilibriumOptimizer

@local_optimizer([gpugemv_no_inplace])
def local_inplace_gpuagemv(node):
    if node.op == gpugemv_no_inplace:
        return [gpugemv_inplace(*node.inputs)]

#gpuablas_opt_inplace = EquilibriumOptimzer(
#    [local_inplace_gpuagemv],
#    failure_callback=EquilibriumOptimizer.warn_inplace,
#    max_use_ratio=5)
#optdb.register('InplaceGpuaBlasOpt',
#               gpuablas_opt_inplace,
#               70.0, 'fast_run', 'inplace', 'gpuarray')
