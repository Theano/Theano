import os

import pygpu

import theano
import theano.sandbox.multinomial
from theano import Apply
from theano.gof import COp, local_optimizer
from .basic_ops import as_gpuarray_variable, infer_context_name
from .type import gpu_context_type, GpuArrayType
from theano.sandbox.cuda.nvcc_compiler import NVCC_compiler
from theano.sandbox import gpuarray
from theano.sandbox.gpuarray.opt import register_opt, op_lifter
from theano.tensor import NotScalarConstantError, get_scalar_constant_value


class GPUAMultinomialFromUniform(COp):
    __props__ = ("odtype",)
    params_type = gpu_context_type

    def __init__(self, odtype):
        COp.__init__(self, ['multinomial.c'], 'APPLY_SPECIFIC(multinomial)')
        self.odtype = odtype

    def get_params(self, node):
        return node.outputs[0].type.context

    def c_compiler(self):
        # TODO: get rid of this
        return NVCC_compiler

    def c_headers(self):
        return ['<numpy_compat.h>', 'gpuarray_helper.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__), pygpu.get_include()]

    def make_node(self, pvals, unis):
        assert pvals.dtype == 'float32'
        assert unis.dtype == 'float32'
        ctx_name = infer_context_name(pvals, unis)

        pvals = as_gpuarray_variable(pvals, ctx_name)
        unis = as_gpuarray_variable(unis, ctx_name)

        if pvals.ndim != 2:
            raise NotImplementedError('pvals ndim should be 2', pvals.ndim)
        if unis.ndim != 1:
            raise NotImplementedError('unis ndim should be 1', unis.ndim)
        if self.odtype == 'auto':
            odtype = pvals.dtype
        else:
            odtype = self.odtype
        if odtype != pvals.dtype:
            raise NotImplementedError(
                'GpuMultinomialFromUniform works only if '
                'self.odtype == pvals.dtype', odtype, pvals.dtype)
        br = (pvals.broadcastable[1], pvals.broadcastable[0])
        out = GpuArrayType(broadcastable=br, dtype=odtype)()

        return Apply(self, [pvals, unis], [out])

 #   def c_code_cache_version(self):
 #       return
 #       return (8,)


@register_opt()
@op_lifter([theano.sandbox.multinomial.MultinomialFromUniform])
def local_gpua_multinomial(node, context_name):
    # TODO : need description for function

    if len(node.inputs) == 2:
        p, u = node.inputs
        n_samples = 1
    else:
        p, u, n_samples = node.inputs
    try:
        if get_scalar_constant_value(n_samples) != 1:
            return None
    except NotScalarConstantError:
        return None
    m, = node.outputs
    if (p.dtype == u.dtype == m.dtype == 'float32'):
        gpu_op = GPUAMultinomialFromUniform(node.op.odtype)
        return gpuarray.elemwise.GpuDimShuffle([False, False], [1, 0])(
            gpu_op(p, u))
